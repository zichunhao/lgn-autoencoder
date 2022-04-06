import torch
import torch.nn as nn
import logging

from .graphnet import GraphNet

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float
LOCAL_MIXING_CHOICES = ('node', 'local')

class Decoder(nn.Module):
    def __init__(
        self, num_nodes, latent_node_size, output_node_size, node_sizes, edge_sizes,
        num_mps, dropout, alphas, batch_norm=False, latent_map='mix', normalize_output=False,
        device=None, dtype=None
    ):
        '''
        The graph decoder of the autoencoder built on `GraphNet`
        
        Parameters
        ----------
        num_nodes: int
            Number of nodes in the decoder.
        latent_node_size: int 
            Size/dimension of the latent feature vectors.
            If the latent map is 'mix', this is the size of the latent feature vectors per node.
        output_node_size: int
            Size/dimension of the output/reconstructed node feature vectors.
        node_sizes: array-like
            List of sizes/dimensions of the node feature vectors in each massage passing step.
        edge_sizes: array-like
            List of sizes/dimensions of the edge feature vectors in each massage passing step.
        num_mps: int
            Number of message passing steps.
        dropout: float
            Dropout rate.
        alphas: array-like
            Alpha value for the leaky relu layer for edge features 
            in each iteration of message passing.
        batch_norm: bool
            Whether to use batch normalization 
            in the edge and node features.
        latent_map: str, default: `'mix'`
            The choice of mapping to latent space. The choices are ('mix', 'mean', 'node', 'local'). 
            If `'mix'`, a linear layer is used to map the node features to the latent space.
            If `'mean'`, the mean is taken across the node features in the graph.
            If `'local'` or `'node'`, linear layers are applied per node.
        device: torch.device, default: `'cuda'` if gpu is available and `'cpu'` otherwise
            The device on which the model is run.
        dtype: torch.dtype, default: torch.float
            The data type of the model.
        '''
        if device is None:
            device = DEFAULT_DEVICE
        if dtype is None:
            dtype = DEFAULT_DTYPE

        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.latent_map = latent_map
        self.latent_node_size = latent_node_size
        self.output_node_size = output_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps
        self.normalize_output = normalize_output

        self.device = device
        self.dtype = dtype

        # layers
        if self.latent_map.lower() in LOCAL_MIXING_CHOICES:
            latent_space_size = latent_node_size * num_nodes
        else:
            latent_space_size = latent_node_size
        self.linear = nn.Linear(
            latent_space_size,
            self.num_nodes*self.latent_node_size
        ).to(self.device).to(self.dtype)

        self.decoder = GraphNet(
            num_nodes=self.num_nodes, 
            input_node_size=self.latent_node_size,
            output_node_size=self.output_node_size, 
            node_sizes=self.node_sizes,
            edge_sizes=self.edge_sizes, 
            num_mps=self.num_mps,
            dropout=dropout, 
            alphas=alphas, 
            batch_norm=batch_norm,
            dtype=self.dtype, 
            device=self.device
        ).to(self.device).to(self.dtype)

        num_params = sum(p.nelement()for p in self.parameters() if p.requires_grad)
        logging.info(f"Decoder initialized. Number of parameters: {num_params}")

    def forward(self, x, metric='cartesian'):
        x = x.to(self.device).to(self.dtype)
        x = self.linear(x).view(-1, self.num_nodes, self.latent_node_size)
        x = self.decoder(x, metric=metric)
        if self.normalize_output:
            x = torch.tanh(x)
        return x
