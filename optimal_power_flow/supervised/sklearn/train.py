import hydra
import os
import random

import numpy as np
import wandb
from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.loading.load_data import load_data
from mlpf.loss.numpy.metrics.active import ActivePowerError, RelativeActivePowerError
from mlpf.loss.numpy.metrics.bounds.active import UpperActivePowerError, LowerActivePowerError
from mlpf.loss.numpy.metrics.bounds.reactive import UpperReactivePowerError, LowerReactivePowerError
from mlpf.loss.numpy.metrics.bounds.voltage import UpperVoltageError, LowerVoltageError
from mlpf.loss.numpy.metrics.costs import ActivePowerCost, RelativeActivePowerCost
from mlpf.loss.numpy.metrics.metrics import MultipleMetrics
from mlpf.loss.numpy.metrics.reactive import ReactivePowerError, RelativeReactivePowerError
from mlpf.utils.description_format import format_description

from data.download import download
from models.sklearn.get_model import get_model


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode)

    cfg = OmegaConf.merge(cfg, dict(wandb.config))
    wandb.config.update({"config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}, allow_val_change=True)

    # Random seeds
    np.random.seed(cfg.general.random_seed)
    random.seed(cfg.general.random_seed)

    download(path=cfg.data.path, google_drive_url=cfg.data.google_drive_url)
    solved_ppc_list = load_data(cfg.data.path, max_samples=cfg.general.max_samples)

    # ppc -> Data
    opf_data = [OptimalPowerFlowData(ppc) for ppc in tqdm(solved_ppc_list, desc="Converting ppcs to data")]

    data_train, data_val = train_test_split(opf_data, test_size=cfg.general.validation_split, random_state=cfg.general.random_seed)

    features_train = np.vstack([data.feature_vector for data in data_train])
    targets_train = np.vstack([data.target_vector for data in data_train])

    features_val = np.vstack([data.feature_vector for data in data_val])
    targets_val = np.vstack([data.target_vector for data in data_val])

    # Model

    model = get_model(cfg.model)
    model.fit(features_train, targets_train)

    # Evaluation

    predictions_val = model.predict(features_val)

    power_metrics = MultipleMetrics(
        ActivePowerError(),
        RelativeActivePowerError(),
        ReactivePowerError(),
        RelativeReactivePowerError(),
        ActivePowerCost(),
        RelativeActivePowerCost(),
        UpperVoltageError(),
        LowerVoltageError(),
        UpperActivePowerError(),
        LowerActivePowerError(),
        UpperReactivePowerError(),
        LowerReactivePowerError()
    )

    # TODO potentially have the same thing for train/

    for i in tqdm(range(predictions_val.shape[0]), desc="Calculating metrics"):
        power_metrics.update(predictions_val[i], data_val[i])

    print(f"\nR2 score: {'train':>10} = {model.score(features_train, targets_train):3.4f}\nR2 score: {'validation':>10} = {model.score(features_val, targets_val):3.4f}\n")

    description = power_metrics.compute().describe()
    description = format_description(description, power_metrics)

    log = {}
    aggregation = "mean"
    metric_dir = "val"
    log[f"{metric_dir}/r2 score"] = model.score(features_val, targets_val)
    for metric_name in description.index:
        unit = description["unit"][metric_name]
        log[f"{metric_dir}/{aggregation} {metric_name.strip()} {unit.strip()}"] = description[aggregation][metric_name]

    wandb.log(log)

    print(description)


if __name__ == "__main__":
    main()
