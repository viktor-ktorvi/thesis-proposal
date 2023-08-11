import hydra
import os
import random

import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
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

    backbone = Ridge()
    model = make_pipeline(StandardScaler(), backbone)
    model.fit(features_train, targets_train)

    # Evaluation

    predictions_val = model.predict(features_val)

    power_metrics = MultipleMetrics(
        ActivePowerError(),
        ReactivePowerError(),
        RelativeActivePowerError(),
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

    for i in tqdm(range(predictions_val.shape[0]), desc="Calculating metrics"):
        power_metrics.update(predictions_val[i], data_val[i])

    print(f"\nR2 score: {'train':>10} = {model.score(features_train, targets_train):3.4f}\nR2 score: {'validation':>10} = {model.score(features_val, targets_val):3.4f}\n")

    description = power_metrics.compute().describe()
    description = format_description(description, power_metrics)

    print(description)


if __name__ == "__main__":
    main()
