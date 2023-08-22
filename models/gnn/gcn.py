import torch
import torch_geometric as pyg

from torch import nn

from mlpf.utils.standard_scaler import StandardScaler


class GCN(torch.nn.Module):
    """
    GCN model:

    x(Nxf_in)->input_scaler->GCN->linear->output(Nxf_out)->output_scaler->PQVA_prediction

    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int,
                 input_scaler: StandardScaler,
                 output_scaler: StandardScaler,
                 jumping_knowledge: str = None):
        super(GCN, self).__init__()
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        self.graph_encoder = pyg.nn.GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=hidden_channels,
            jk=jumping_knowledge
        )

        self.linear = nn.Linear(in_features=hidden_channels, out_features=out_channels)

    def forward(self, data):
        out = self.input_scaler(data.x)
        out = self.graph_encoder(x=out, edge_index=data.edge_index)
        out = self.linear(out)

        out = self.output_scaler.inverse(out)

        out[data.PQVA_mask] = data.PQVA_matrix[data.PQVA_mask]

        return out
