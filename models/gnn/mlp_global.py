import torch
from torch import Tensor

from torch_geometric.data import Data

from mlpf.utils.standard_scaler import StandardScaler


class LinearGlobal(torch.nn.Module):
    """
    A purely linear layer model.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 input_scaler: StandardScaler,
                 output_scaler: StandardScaler):
        super(LinearGlobal, self).__init__()

        self.input_scaler = input_scaler
        self.output_scaler = output_scaler

        self.linear = torch.nn.Linear(in_features=input_size, out_features=output_size)

    def scale_output(self, output: Tensor, data: Data) -> Tensor:
        """
        Scale the given output with the model output scaler.

        :param output: Either out prediction or target of shape Nxf
        :param data: Corresponding data batch.
        :return: Scaled version.
        """
        batch_size = data.num_graphs
        output_size = data.feature_vector.shape[1]
        return self.output_scaler(output[~data.PQVA_mask].reshape(batch_size, output_size))

    def forward(self, data: Data) -> Tensor:
        out = self.input_scaler(data.feature_vector)
        out = self.linear(out)
        out = self.output_scaler.inverse(out)

        PQVA_prediction = torch.zeros_like(data.PQVA_matrix)
        PQVA_prediction[data.PQVA_mask] = data.PQVA_matrix[data.PQVA_mask]
        PQVA_prediction[~data.PQVA_mask] = out.flatten()

        return PQVA_prediction
