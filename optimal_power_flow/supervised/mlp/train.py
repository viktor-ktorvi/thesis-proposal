import hydra
import os
import torch
import wandb

import torch.nn as nn
import torch_geometric as pyg
from omegaconf import OmegaConf

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.loading.load_data import load_data
from mlpf.utils.standard_scaler import StandardScaler

from data.download import download
from utils.logging import collect_log, clean_metric_name
from utils.metrics import optimal_power_flow_metrics_with_mse_and_r2score


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
    # TODO tags
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode)

    # update hydra config with wandb config
    cfg = OmegaConf.merge(cfg, dict(wandb.config))

    # update wandb config with hydra config so that the complete config goes to the run info
    wandb.config.update({"config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}, allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seeds
    pyg.seed_everything(cfg.general.random_seed)

    download(path=cfg.data.path, google_drive_url=cfg.data.google_drive_url)
    solved_ppc_list = load_data(cfg.data.path, max_samples=cfg.general.max_samples)

    # ppc -> Data
    opf_data_list = [OptimalPowerFlowData(solved_ppc).to_pyg_data() for solved_ppc in tqdm(solved_ppc_list, ascii=True, desc="Converting ppcs to data")]

    data_train, data_val = train_test_split(opf_data_list, test_size=cfg.general.validation_split, random_state=cfg.general.random_seed)

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

    metrics_train = optimal_power_flow_metrics_with_mse_and_r2score(output_size).to(device)
    metrics_val = optimal_power_flow_metrics_with_mse_and_r2score(output_size).to(device)

    metric_value_width = 15
    metric_name_width = 50
    metric_value_decimals = 7
    description_width = 100
    # TODO all this formatting will have to be its own function;
    print(f"\n{'Training':^{metric_value_width}} | {'Validation':^{metric_value_width}} {'Metric':{metric_name_width}} Unit\n")

    # if running from the IDE console, make sure to select 'emulate terminal' in the run configuration, otherwise the output will look bad
    progress_bars: dict = {
        key: value for key, value in zip(
            metrics_train,
            [tqdm(total=cfg.model.num_epochs, position=i) for i in range(len(metrics_train))]
        )
    }
    for epoch in range(cfg.model.num_epochs):

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
        for key in overall_metrics_train.keys():
            unit = getattr(metrics_train[key], 'unit', None)  # get metric unit if it exists
            unit_in_brackets = f" [{unit}]" if unit is not None else ''

            description = f"{overall_metrics_train[key]:^{metric_value_width}.{metric_value_decimals}f} | {overall_metrics_val[key]:^{metric_value_width}.{metric_value_decimals}f} {clean_metric_name(key):{metric_name_width}}{unit_in_brackets}"

            progress_bars[key].set_description(f"{description:{description_width}}")
            progress_bars[key].update(1)

        wandb.log(collect_log(metrics_train, metrics_val))

        metrics_train.reset()
        metrics_val.reset()


if __name__ == '__main__':
    main()
