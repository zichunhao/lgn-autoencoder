import torch
import torch.nn as nn
import logging

from models.graphnet import GraphNet


class Encoder(nn.Module):
    def __init__(self, num_nodes, input_node_size, latent_node_size, node_sizes, edge_sizes,
                 num_mps, dropout, alphas, batch_norm=True, latent_map='mix', device=None, dtype=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float

        super(Encoder, self).__init__()

        self.num_nodes = num_nodes
        self.input_node_size = input_node_size
        self.latent_node_size = latent_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps
        self.latent_map = latent_map

        self.device = device
        self.dtype = dtype

        # layers
        self.encoder = GraphNet(num_nodes=self.num_nodes, input_node_size=input_node_size,
                                output_node_size=self.latent_node_size, node_sizes=self.node_sizes,
                                edge_sizes=self.edge_sizes, num_mps=self.num_mps,
                                dropout=dropout, alphas=alphas, batch_norm=batch_norm,
                                dtype=self.dtype, device=self.device).to(self.device).to(self.dtype)
        
        if self.latent_map.lower() == 'mix':
            self.mix_layer = nn.Linear(self.latent_node_size*self.num_nodes,
                                       self.latent_node_size, 
                                       bias=False).to(self.device).to(self.dtype)

        logging.info(f"Encoder initialized. Number of parameters: {sum(p.nelement() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x, metric='cartesian'):
        bs = x.shape[0]
        x = x.to(self.device).to(self.dtype)
        x = self.encoder(x, metric=metric)
        if self.latent_map.lower() == 'mean':
            x = torch.mean(x, dim=-2).unsqueeze(dim=0)  # Latent vector
        elif self.latent_map.lower() == 'mix':
            x = self.mix_layer(x.view(bs, -1)).unsqueeze(dim=0)
        return x
