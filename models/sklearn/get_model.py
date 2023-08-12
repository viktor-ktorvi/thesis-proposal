from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_model(model_cfg: DictConfig):
    """
    Return a model constructed from a config and additional parameters.

    :param model_cfg: Hydra model config.
    :return: Model.
    """
    if model_cfg.name == "ridge":
        backbone = Ridge(alpha=model_cfg.alpha)

    else:
        raise ValueError(f"Model '{model_cfg.name}' is not supported.")

    return make_pipeline(StandardScaler(), backbone)
