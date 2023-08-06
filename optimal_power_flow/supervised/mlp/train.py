import hydra
import torch
import wandb

import torch.nn as nn
import torch_geometric as pyg

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection, MeanSquaredError, R2Score
from tqdm import tqdm

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.loading.load_data import load_data, load_solved_from_tuple
from mlpf.loss.torch.metrics.active import MeanActivePowerError, MeanRelativeActivePowerError
from mlpf.loss.torch.metrics.bounds.active import MeanUpperActivePowerError, MeanLowerActivePowerError
from mlpf.loss.torch.metrics.bounds.reactive import MeanUpperReactivePowerError, MeanLowerReactivePowerError
from mlpf.loss.torch.metrics.bounds.voltage import MeanUpperVoltageError, MeanLowerVoltageError
from mlpf.loss.torch.metrics.costs import MeanActivePowerCost, MeanRelativeActivePowerCost
from mlpf.loss.torch.metrics.reactive import MeanReactivePowerError, MeanRelativeReactivePowerError
from mlpf.utils.standard_scaler import StandardScaler


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    wandb.init(project="thesis proposal", mode="online")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seeds
    pyg.seed_everything(cfg.random_seed)

    solved_ppc_list = load_data(cfg.data_path, max_samples=cfg.max_samples)

    # ppc -> Data
    opf_data_list = []
    for solved_ppc in tqdm(solved_ppc_list, ascii=True, desc="Converting ppcs to data"):
        opf_data_list.append(OptimalPowerFlowData(solved_ppc).to_pyg_data())

    data_train, data_val = train_test_split(opf_data_list, test_size=cfg.validation_split, random_state=cfg.random_seed)

    # Torch dataloaders

    train_loader = DataLoader(data_train, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=cfg.model.batch_size, shuffle=False)

    train_features = torch.vstack([data.feature_vector for data in data_train])
    train_targets = torch.vstack([data.target_vector for data in data_train])

    # scale the targets because the different grid state values are at different orders of magnitude
    output_scaler = StandardScaler(train_targets)
    output_scaler.to(device)

    input_size = data_train[0].feature_vector.shape[1]
    output_size = data_train[0].target_vector.shape[1]

    # Model
    model = nn.Sequential(
        StandardScaler(train_features),
        nn.Linear(in_features=input_size, out_features=output_size),
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = nn.MSELoss()

    # Metrics

    metrics_train = MetricCollection(
        MeanSquaredError(),
        R2Score(num_outputs=output_size),
        MeanActivePowerError(),
        MeanRelativeActivePowerError(),
        MeanReactivePowerError(),
        MeanRelativeReactivePowerError(),
        MeanActivePowerCost(),
        MeanRelativeActivePowerCost(),
        MeanUpperVoltageError(),
        MeanLowerVoltageError(),
        MeanUpperActivePowerError(),
        MeanLowerActivePowerError(),
        MeanUpperReactivePowerError(),
        MeanLowerReactivePowerError()
    ).to(device)

    metrics_val = MetricCollection(
        MeanSquaredError(),
        R2Score(num_outputs=output_size),
        MeanActivePowerError(),
        MeanRelativeActivePowerError(),
        MeanReactivePowerError(),
        MeanRelativeReactivePowerError(),
        MeanActivePowerCost(),
        MeanRelativeActivePowerCost(),
        MeanUpperVoltageError(),
        MeanLowerVoltageError(),
        MeanUpperActivePowerError(),
        MeanLowerActivePowerError(),
        MeanUpperReactivePowerError(),
        MeanLowerReactivePowerError()
    ).to(device)

    progress_bar = tqdm(range(cfg.model.num_epochs), ascii=True, desc="Training | Validation:")

    for epoch in progress_bar:

        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            features, targets = batch.feature_vector, batch.target_vector

            optimizer.zero_grad()

            predictions = model(features)
            loss = criterion(predictions, output_scaler(targets))
            loss.backward()

            metrics_train(preds=predictions, target=output_scaler(targets), power_flow_predictions=output_scaler.inverse(predictions), batch=batch)

            optimizer.step()

        # Validation
        with torch.no_grad():

            model.eval()
            for batch in val_loader:
                batch = batch.to(device)
                features, targets = batch.feature_vector, batch.target_vector

                predictions = model(features)

                metrics_val(preds=predictions, target=output_scaler(targets), power_flow_predictions=output_scaler.inverse(predictions), batch=batch)

        overall_metrics_train = metrics_train.compute()
        overall_metrics_val = metrics_val.compute()

        # logging
        description = "Training | Validation:"
        logs = {}
        for key in overall_metrics_train.keys():
            unit = getattr(metrics_train[key], 'unit', None)  # get metric unit if it exists
            unit_in_brackets = f" [{unit}]" if unit is not None else ''
            log_unit = unit_in_brackets.replace("/", " per ")

            logs[f"train/{key}{log_unit}"] = overall_metrics_train[key]
            logs[f"val/{key}{log_unit}"] = overall_metrics_val[key]

            description += f" {key}{unit_in_brackets}: ({overall_metrics_train[key]:2.4f} | {overall_metrics_val[key]:2.4f});"

        wandb.log(logs)
        progress_bar.set_description(description)

        metrics_train.reset()
        metrics_val.reset()


if __name__ == '__main__':
    main()
