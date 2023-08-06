import torch
import torch_geometric as pyg

from torch import nn


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, out_channels, standard_scaler):
        super(GCN, self).__init__()
        self.standard_scaler = standard_scaler

        self.graph_encoder = pyg.nn.GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels
        )

        self.linear = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, data):
        out = self.standard_scaler(data.x)
        out = self.graph_encoder(x=out, edge_index=data.edge_index)
        out = self.linear(out)[~data.PQVA_mask].reshape(data.target_vector.shape)

        return out
