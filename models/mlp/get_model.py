from typing import List

import torch

from mlpf.utils.standard_scaler import StandardScaler
from omegaconf import DictConfig
from torch import nn
from torch_geometric.data import Data


def get_model(model_cfg: DictConfig, data_train: List[Data]) -> nn.Module:
    """
    Return a model constructed from a config and additional parameters.

    :param model_cfg: Hydra model config.
    :param data_train: List of PyG Data.
    :return: Model.
    """
    features_train = torch.vstack([data.feature_vector for data in data_train])
    if model_cfg.name == "linear":
        backbone = nn.Linear(in_features=features_train.shape[1], out_features=data_train[0].target_vector.shape[1])

    else:
        raise ValueError(f"Model '{model_cfg.name}' is not supported.")

    return nn.Sequential(
        StandardScaler(features_train),
        backbone
    )
