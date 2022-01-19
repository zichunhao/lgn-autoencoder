import torch
import torch.nn as nn
import logging

from .graphnet import GraphNet


class Decoder(nn.Module):
    def __init__(self, num_nodes, latent_node_size, output_node_size, node_sizes, edge_sizes,
                 num_mps, dropout, alphas, batch_norm=True, device=None, dtype=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.float

        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.latent_node_size = latent_node_size
        self.output_node_size = output_node_size
        self.node_sizes = node_sizes
        self.edge_sizes = edge_sizes
        self.num_mps = num_mps

        self.device = device
        self.dtype = dtype

        # layers
        self.linear = nn.Linear(self.latent_node_size, self.num_nodes*self.latent_node_size).to(self.device).to(self.dtype)

        self.decoder = GraphNet(num_nodes=self.num_nodes, input_node_size=self.latent_node_size,
                                output_node_size=self.output_node_size, node_sizes=self.node_sizes,
                                edge_sizes=self.edge_sizes, num_mps=self.num_mps,
                                dropout=dropout, alphas=alphas, batch_norm=batch_norm,
                                dtype=self.dtype, device=self.device).to(self.device).to(self.dtype)

        logging.info(f"Decoder initialized. Number of parameters: {sum(p.nelement() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x, metric='cartesian'):
        x = x.to(self.device).to(self.dtype)
        x = self.linear(x).view(-1, self.num_nodes, self.latent_node_size)
        x = self.decoder(x, metric=metric)
        return x
