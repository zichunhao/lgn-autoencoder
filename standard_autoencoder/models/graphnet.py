import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float

class GraphNet(nn.Module):

    def __init__(
        self, num_nodes, input_node_size, output_node_size, node_sizes, edge_sizes,
        num_mps, dropout=0.1, alphas=0.1, batch_norm=False, 
        device=None, dtype=None
    ):
        """
        A fully connected message-passing standard graph neural network
        with the distance as edge features.

        Parameters
        ----------
        num_nodes : int
            Number of nodes for the graph.
        input_node_size : int
            Size/dimension of input node feature vectors.
        output_node_size : int
            Size/dimension of output node feature vectors.
        node_sizes : list of list of int
            Sizes/dimensions of hidden node 
            in each layer in each iteration of message passing.
        edge_sizes : list of list of int
            Sizes/dimensions of edges networks 
            in each layer  in each iteration of message passing.
        num_mps : int
            The number of message passing step.
        dropout : float
            Dropout momentum for edge features 
            in each iteration of message passing.
        alphas: array-like
            Alpha value for the leaky relu layer for edge features 
            in each iteration of message passing.
        batch_norm : bool, default: `True`
            Whether to use batch normalization 
            in the edge and node features.
        device: torch.device, default: `'cuda'` if gpu is available and `'cpu'` otherwise
            The device on which the model is run.
        dtype: torch.dtype, default: torch.float
            The data type of the model.
        """

        node_sizes = _adjust_var_list(node_sizes, num_mps)
        edge_sizes = _adjust_var_list(edge_sizes, num_mps)
        alphas = _adjust_var_list(alphas, num_mps)

        if device is None:
            device = DEFAULT_DEVICE
        if dtype is None:
            dtype = DEFAULT_DTYPE

        super(GraphNet, self).__init__()
        self.device = device
        self.dtype = dtype
        self.eps = 1e-16 if self.dtype in [torch.float64, torch.double] else 1e-12

        # Node networks
        self.num_nodes = num_nodes  # Number of nodes in graph
        self.input_node_size = input_node_size  # Dimension of input node features
        self.node_sizes = node_sizes  # List of dimensions of hidden node features
        self.node_net = nn.ModuleList()
        self.output_node_size = output_node_size  # Dimension of output node features
        self.output_layer = nn.Linear(self.node_sizes[-1][-1], self.output_node_size)

        # Edge networks
        self.edge_sizes = edge_sizes  # Output sizes in edge networks
        self.input_edge_sizes = [2 * s[0] + 1 for s in self.node_sizes]  # mij = xi ⊕ xj ⊕ d(xi, xj)
        self.edge_net = nn.ModuleList()

        self.num_mps = num_mps  # Number of message passing
        self.batch_norm = batch_norm  # Use batch normalization (bool)
        if self.batch_norm:
            self.bn_node = nn.ModuleList()
            self.bn_edge = nn.ModuleList()

        self.alphas = alphas  # For leaky relu layer for edge features
        self.dropout_p = dropout  # Dropout layer for edge features

        for i in range(self.num_mps):
            # Edge layers
            edge_layers = _create_dnn(layer_sizes=self.edge_sizes[i], input_size=self.input_edge_sizes[i])
            self.edge_net.append(edge_layers)
            if self.batch_norm:
                bn_edge_i = nn.ModuleList()
                for j in range(len(edge_layers)):
                    bn_edge_i.append(nn.BatchNorm1d(self.edge_sizes[i][j]))
                self.bn_edge.append(bn_edge_i)

            # Node layers
            node_layers = _create_dnn(layer_sizes=self.node_sizes[i])
            node_layers.insert(0, nn.Linear(self.edge_sizes[i][-1] + self.node_sizes[i][0], self.node_sizes[i][0]))
            if i+1 < self.num_mps:
                node_layers.append(nn.Linear(node_sizes[i][-1], self.node_sizes[i+1][0]))
            else:
                node_layers.append(nn.Linear(node_sizes[i][-1], self.output_node_size))
            self.node_net.append(node_layers)
            if self.batch_norm:
                bn_node_i = nn.ModuleList()
                for j in range(len(self.node_net[i])):
                    bn_node_i.append(nn.BatchNorm1d(self.node_net[i][j].out_features))
                self.bn_node.append(bn_node_i)

        self.to(self.device)

    def forward(self, x, metric='cartesian'):
        """
        Parameter
        ----------
        x : torch.Tensor
            The input node features.
        metric : str or function
            The metric for computing distances between nodes.
            Choices:
                - 'cartesian': diag(+, +, +, +)
                - 'minkowskian': diag(+, -, -, -), which is used only if x.shape[-1] == 4
                - A mapping to R.
            Default: 'cartesian'

        Return
        ------
        x : torch.Tensor
            Updated node features.
        """
        self.metric = metric
        batch_size = x.shape[0]

        x = F.pad(x, (0, self.node_sizes[0][0] - self.input_node_size, 0, 0, 0, 0)).to(self.device)

        for i in range(self.num_mps):
            metric = _get_metric_func(self.metric if x.shape[-1] == 4 else 'cartesian')
            A = self._getA(x, batch_size, self.input_edge_sizes[i], hidden_node_size=self.node_sizes[i][0], metric=metric)
            A = self._edge_conv(A, i)
            x = self._aggregate(x, A, i, batch_size)
            x = x.view(batch_size, self.num_nodes, -1)

        x = x.view(batch_size, self.num_nodes, self.output_node_size)
        return x

    def _getA(self, x, batch_size, input_edge_size, hidden_node_size, metric):
        """
        Parameters
        ----------
        x: torch.Tensor
            Node features with shape (batch_size, num_particles, 4) or (batch_size, num_particles, 3)
        batch_size: int
            Batch size.

        Return
        ------
        A: torch.Tensor with shape (batch_size * self.num_nodes * self.num_nodes, input_edge_size)
            Adjacency matrix that stores distances among nodes.
        """
        x1 = x.repeat(1, 1, self.num_nodes).view(batch_size, self.num_nodes*self.num_nodes, hidden_node_size)
        x2 = x.repeat(1, self.num_nodes, 1)  # 1*(self.num_nodes)*1 tensor with repeated x along axis=1
        dists = metric(x2 - x1 + self.eps)
        A = torch.cat((x1, x2, dists), 2).view(batch_size*self.num_nodes*self.num_nodes, input_edge_size)
        return A

    def _concat(self, A, x, batch_size, edge_size, node_size):
        A = A.view(batch_size, self.num_nodes, self.num_nodes, edge_size)
        A = torch.sum(A, 2)
        x = torch.cat((A, x), 2)
        x = x.view(batch_size * self.num_nodes, edge_size + node_size)
        return x

    def _dropout(self, x):
        dropout = nn.Dropout(p=self.dropout_p)
        return dropout(x)

    def _aggregate(self, x, A, i, batch_size):
        x = self._concat(A, x, batch_size, self.edge_sizes[i][-1], self.node_sizes[i][0])
        for j in range(len(self.node_net[i])):
            x = self.node_net[i][j](x)
            x = F.leaky_relu(x, negative_slope=self.alphas[i])
            if self.batch_norm:
                x = self.bn_node[i][j](x)
            x = self._dropout(x)
        return x

    def _edge_conv(self, A, i):
        for j in range(len(self.edge_net[i])):
            A = self.edge_net[i][j](A)
            A = F.leaky_relu(A, negative_slope=self.alphas[i])
            if self.batch_norm:
                A = self.bn_edge[i][j](A)
            A = self._dropout(A)
        return A


def _create_dnn(layer_sizes, input_size=-1):
    dnn = nn.ModuleList()
    if input_size >= 0:
        sizes = layer_sizes.copy()
        sizes.insert(0, input_size)
        for i in range(len(layer_sizes)):
            dnn.append(nn.Linear(sizes[i], sizes[i+1]))
    else:
        for i in range(len(layer_sizes) - 1):
            dnn.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    return dnn


def _adjust_var_list(data, num):
    try:
        if len(data) < num:
            data = data + [data[-1]] * (num - len(data))
            return data
    except TypeError:  # Non interable
        data = [data] * num
    return data[:num]


def _get_metric_func(metric):
    if isinstance(metric, types.FunctionType):
        return metric
    if metric.lower() == 'cartesian':
        return lambda x: torch.norm(x, dim=2).unsqueeze(2)
    if metric.lower() == 'minkowskian':
        return lambda x: (2 * torch.pow(x[..., 0], 2) - 2 * torch.sum(torch.pow(x, 2), dim=-1)).unsqueeze(2)
    else:
        logging.warning(f"Metric ({metric}) for adjacency matrix is not implemented. Use 'cartesian' instead.")
        return lambda x: torch.norm(x, dim=2).unsqueeze(2)
