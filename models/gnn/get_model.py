from typing import List

import torch

from mlpf.utils.standard_scaler import StandardScaler
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data

from models.gnn.gcn import GCN


def get_model(model_cfg: DictConfig, data_train: List[Data]) -> nn.Module:
    """
    Return a model constructed from a config and additional parameters.

    :param model_cfg: Hydra model config.
    :param data_train: List of PyG Data objects.
    :return: Model.
    """
    node_features_stacked = torch.vstack([data.x for data in data_train])

    if model_cfg.name == "gcn":
        return GCN(
            in_channels=data_train[0].x.shape[1],
            hidden_channels=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            out_channels=data_train[0].PQVA_matrix.shape[1],
            standard_scaler=StandardScaler(node_features_stacked)
        )

    else:
        raise ValueError(f"Model '{model_cfg.name}' is not supported.")
