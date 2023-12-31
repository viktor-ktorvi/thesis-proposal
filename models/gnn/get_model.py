from typing import List

import torch

from mlpf.utils.standard_scaler import StandardScaler
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data

from models.gnn.gcn import GCN
from models.gnn.mlp_global import LinearGlobal


def get_model(model_cfg: DictConfig, data_train: List[Data]) -> nn.Module:
    """
    Return a model constructed from a config and additional parameters.

    :param model_cfg: Hydra model config.
    :param data_train: List of PyG Data objects.
    :return: Model.
    """
    node_features_stacked = torch.vstack([data.x for data in data_train])
    node_targets_stacked = torch.vstack([data.PQVA_matrix for data in data_train])

    if model_cfg.name == "gcn":
        return GCN(
            in_channels=data_train[0].x.shape[1],
            hidden_channels=model_cfg.hidden_channels,
            num_layers=model_cfg.num_layers,
            out_channels=data_train[0].PQVA_matrix.shape[1],
            input_scaler=StandardScaler(node_features_stacked),
            output_scaler=StandardScaler(node_targets_stacked),
            jumping_knowledge=model_cfg.jumping_knowledge
        )

    elif model_cfg.name == "linear":
        global_features_stacked = torch.vstack([data.feature_vector for data in data_train])
        global_targets_stacked = torch.vstack([data.target_vector for data in data_train])

        return LinearGlobal(
            input_size=global_features_stacked.shape[1],
            output_size=global_targets_stacked.shape[1],
            input_scaler=StandardScaler(global_features_stacked),
            output_scaler=StandardScaler(global_targets_stacked)
        )

    else:
        raise ValueError(f"Model '{model_cfg.name}' is not supported.")
