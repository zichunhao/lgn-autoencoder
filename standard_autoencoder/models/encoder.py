import torch
import torch.nn as nn
import logging

from .graphnet import GraphNet

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.float
LOCAL_MIXING_CHOICES = ('node', 'local')

class Encoder(nn.Module):
    def __init__(
        self, num_nodes, input_node_size, latent_node_size, node_sizes, edge_sizes,
        num_mps, dropout, alphas, batch_norm=False, latent_map='mix', 
        device=None, dtype=None
    ):
        """
        The graph encoder of the autoencoder built on `GraphNet`

        Parameters
        ----------
        num_nodes: int
            Number of nodes in the graph.
        input_node_size: int
            Size/dimension of the input feature vectors. 
        latent_node_size: int 
            Size/dimension of the latent feature vectors.
            If `encoder_level` is `'global'`, this is the size of the global latent vector.
            If `encoder_level` is `'local'` or '`node'`, this is the size of the local latent vectors
            so that the total size is `latent_node_size * num_nodes`.
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
        """
        if device is None:
            device = DEFAULT_DEVICE
        if dtype is None:
            dtype = DEFAULT_DTYPE

        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.input_node_size = input_node_size
        self.latent_node_size = latent_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps
        self.latent_map = latent_map
        if self.latent_map.lower() in LOCAL_MIXING_CHOICES:
            self.latent_space_size = latent_node_size * num_nodes
        else:
            self.latent_space_size = latent_node_size

        self.device = device
        self.dtype = dtype

        # layers
        if self.latent_map.lower() in LOCAL_MIXING_CHOICES:
            encoder_out_size = self.node_sizes[-1][-1]
        else:
            encoder_out_size = self.latent_node_size
        
        self.encoder = GraphNet(
            num_nodes=self.num_nodes, 
            input_node_size=input_node_size,
            output_node_size=encoder_out_size,
            node_sizes=self.node_sizes,
            edge_sizes=self.edge_sizes, 
            num_mps=self.num_mps,
            dropout=dropout, 
            alphas=alphas, 
            batch_norm=batch_norm,
            dtype=self.dtype, 
            device=self.device
        ).to(self.device).to(self.dtype)
        
        if self.latent_map.lower() == 'mix':
            self.mix_layer = nn.Linear(
                self.latent_node_size*self.num_nodes,
                self.latent_node_size, 
                bias=False
            ).to(self.device).to(self.dtype)
        elif self.latent_map.lower() in ['local', 'node', 'conv']:
            self.mix_layer = nn.Linear(
                encoder_out_size, latent_node_size
            ).to(self.device).to(self.dtype)
        else:
            pass

        num_params = sum(
            p.nelement() 
            for p in self.parameters() 
            if p.requires_grad
        )

        logging.info(
            f"Encoder initialized. Number of parameters: {num_params}"
        )

    def forward(self, x, metric='cartesian'):
        bs = x.shape[0]
        x = x.to(self.device).to(self.dtype)
        x = self.encoder(x, metric=metric)
        
        if self.latent_map.lower() == 'mean':
            x = torch.mean(x, dim=-2).unsqueeze(dim=0)  # Latent vector
        elif self.latent_map.lower() == 'mix':
            x = self.mix_layer(x.view(bs, -1)).unsqueeze(dim=0)
        elif self.latent_map.lower() in LOCAL_MIXING_CHOICES:
            x = self.mix_layer(x).view(bs, -1)
        else:
            logging.warning(f"Unknown latent map {self.latent_map} in Encoder. Using mean.")
            x = torch.mean(x, dim=-2).unsqueeze(dim=0)
        
        return x
