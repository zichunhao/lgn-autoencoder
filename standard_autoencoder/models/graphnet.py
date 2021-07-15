import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphNet(nn.Module):
    """
    The basic graph neural network in the autoencoder.

    Parameters
    ----------
    num_nodes: int
        Number of nodes for the graph.
    input_node_size: int
        Dimension of input node feature vectors.
    output_node_size: int
        Dimension of output node feature vectors.
    num_hidden_node_layers: int
        Number of layers of hidden nodes.
    hidden_edge_size: int
        Dimension of hidden edges before message passing.
    output_edge_size: int
        Dimension of output edges for message passing.
    num_mps: int
        The number of message passing step.
    dropout: float
        Dropout value for edge features.
    alpha: float
        Alpha value for the leaky relu layer for edge features.
    batch_norm: bool (default: True)
        Whether to use batch normalization.
    """

    def __init__(self, num_nodes, input_node_size, output_node_size, num_hidden_node_layers,
                 hidden_edge_size, output_edge_size, num_mps, dropout, alpha, batch_norm=True, 
                 device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super(GraphNet, self).__init__()

        # Nodes
        self.num_nodes = num_nodes  # Number of nodes in graph
        self.input_node_size = input_node_size  # Dimension of input node features
        self.hidden_node_size = output_node_size  # Dimension of hidden node features
        self.num_hidden_node_layers = num_hidden_node_layers  # Node layers in node networks

        # Edges
        self.hidden_edge_size = hidden_edge_size  # Hidden size in edge networks
        self.output_edge_size = output_edge_size  # Output size in edge networks
        self.input_edge_size = 2 * self.hidden_node_size + 1

        self.num_mps = num_mps  # Number of message passing
        self.batch_norm = batch_norm  # Use batch normalization (bool)

        self.alpha = alpha  # For leaky relu layer for edge features
        self.dropout = nn.Dropout(p=dropout)  # Dropout layer for edge features

        # AGGREGATE function
        self.aggregate_hidden = nn.ModuleList()
        self.aggregate = nn.ModuleList()

        # UPDATE function
        self.update_hidden = nn.ModuleList()
        self.update = nn.ModuleList()

        # Batch normalization layers
        self.bn_edge_hidden = nn.ModuleList()
        self.bn_edge = nn.ModuleList()
        self.bn_node = nn.ModuleList()

        self.device = device

        for i in range(self.num_mps):
            # Edge feature layers
            self.aggregate_hidden.append(nn.Linear(self.input_edge_size, self.hidden_edge_size))
            self.aggregate.append(nn.Linear(self.hidden_edge_size, self.output_edge_size))

            if batch_norm:
                self.bn_edge_hidden.append(nn.BatchNorm1d(self.hidden_edge_size))
                self.bn_edge.append(nn.BatchNorm1d(self.output_edge_size))

            # Node feature layers
            node_layers = nn.ModuleList()
            node_layers.append(nn.Linear(self.output_edge_size + self.hidden_node_size, self.hidden_node_size))

            for j in range(self.num_hidden_node_layers - 1):
                node_layers.append(nn.Linear(self.hidden_node_size, self.hidden_node_size))

            self.update_hidden.append(node_layers)  # Layer for message Aggregation of hidden layer
            self.update.append(nn.Linear(self.hidden_node_size, self.hidden_node_size))  # Layer for message Aggregation

            if batch_norm:
                bn_node_i = nn.ModuleList()
                for i in range(num_hidden_node_layers):
                    bn_node_i.append(nn.BatchNorm1d(self.hidden_node_size))
                self.bn_node.append(bn_node_i)

    def forward(self, x, eps=1e-12):
        """
        Parameter
        ----------
        x: torch.Tensor
            The input node features.

        Return
        ------
        x: torch.Tensor
            Node features after message passing.
        """
        self.x = x
        batch_size = x.shape[0]

        x = F.pad(x, (0, self.hidden_node_size - self.input_node_size, 0, 0, 0, 0)).to(self.device)

        for i in range(self.num_mps):
            # Edge features
            A = self.getA(x, batch_size, eps=eps)

            # Edge layer 1
            A = F.leaky_relu(self.aggregate_hidden[i](A), negative_slope=self.alpha)
            if self.batch_norm:
                A = self.bn_edge_hidden[i](A)
            A = self.dropout(A)

            # Edge layer 2
            A = F.leaky_relu(self.aggregate[i](A), negative_slope=self.alpha)
            if self.batch_norm:
                A = self.bn_edge[i](A)
            A = self.dropout(A)

            # Concatenation
            A = A.view(batch_size, self.num_nodes, self.num_nodes, self.output_edge_size)
            A = torch.sum(A, 2)
            x = torch.cat((A, x), 2)
            x = x.view(batch_size * self.num_nodes, self.output_edge_size + self.hidden_node_size)

            # Aggregation
            for j in range(self.num_hidden_node_layers):
                x = F.leaky_relu(self.update_hidden[i][j](x), negative_slope=self.alpha)
                if self.batch_norm:
                    x = self.bn_node[i][j](x)
                x = self.dropout(x)

            x = self.dropout(self.update[i](x))
            x = x.view(batch_size, self.num_nodes, self.hidden_node_size)

        return x

    def getA(self, x, batch_size, eps=1e-12):
        """
        Parameters
        ----------
        x: torch.Tensor
            The node features.
        batch_size: int
            The batch size.

        Return
        ------
        A: torch.Tensor with shape (batch_size * self.num_nodes * self.num_nodes, self.input_edge_size)
            The adjacency matrix that stores the message m = MESSAGE(hv, hw).
        """
        x1 = x.repeat(1, 1, self.num_nodes).view(batch_size, self.num_nodes*self.num_nodes, self.hidden_node_size).to(self.device)
        x2 = x.repeat(1, self.num_nodes, 1).to(self.device)  # 1*(self.num_nodes)*1 tensor with repeated x along axis=1
        dists = torch.norm(x2 - x1 + eps, dim=2).unsqueeze(2)
        A = torch.cat((x1, x2, dists), 2).view(batch_size * self.num_nodes * self.num_nodes, self.input_edge_size)
        return A
