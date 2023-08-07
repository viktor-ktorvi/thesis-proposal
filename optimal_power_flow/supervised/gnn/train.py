import hydra
import os
import torch
import wandb

import torch.nn as nn
import torch_geometric as pyg

from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.loading.load_data import load_data
from mlpf.utils.standard_scaler import StandardScaler

from data.download import download
from models.gcn import GCN
from utils.metrics import optimal_power_flow_metrics_with_mse_and_r2score


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
    # TODO tags
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seeds
    pyg.seed_everything(cfg.general.random_seed)

    download(path=cfg.data.path, google_drive_url=cfg.data.google_drive_url)
    solved_ppc_list = load_data(cfg.data.path, max_samples=cfg.general.max_samples)

    # ppc -> Data
    opf_data_list = [OptimalPowerFlowData(solved_ppc).to_pyg_data() for solved_ppc in tqdm(solved_ppc_list, ascii=True, desc="Converting ppcs to data")]

    for data in opf_data_list:
        data.x[~data.PQVA_mask] = 0.0  # delete the target info from the input features

    data_train, data_val = train_test_split(opf_data_list, test_size=cfg.general.validation_split, random_state=cfg.general.random_seed)

    # Torch dataloaders

    train_loader = DataLoader(data_train, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=cfg.model.batch_size, shuffle=False)

    node_features_stacked = torch.vstack([data.x for data in data_train])
    train_targets = torch.vstack([data.target_vector for data in data_train])

    output_size = train_targets.shape[1]

    output_scaler = StandardScaler(train_targets)
    output_scaler.to(device)

    # Model

    standard_scaler = StandardScaler(node_features_stacked)
    model = GCN(in_channels=node_features_stacked.shape[1],
                hidden_channels=cfg.model.hidden_channels,
                num_layers=cfg.model.num_layers,
                out_channels=data_train[0].PQVA_matrix.shape[1],
                standard_scaler=standard_scaler)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = nn.MSELoss()

    # Metrics

    metrics_train = optimal_power_flow_metrics_with_mse_and_r2score(output_size).to(device)
    metrics_val = optimal_power_flow_metrics_with_mse_and_r2score(output_size).to(device)

    progress_bar = tqdm(range(cfg.model.num_epochs), ascii=True, desc="Training | Validation:")

    for epoch in progress_bar:

        # Training
        model.train()
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            predictions = model(batch)
            loss = criterion(predictions, output_scaler(batch.target_vector))
            loss.backward()

            metrics_train(preds=predictions, target=output_scaler(batch.target_vector), power_flow_predictions=output_scaler.inverse(predictions), batch=batch)

            optimizer.step()

        # Validation
        with torch.no_grad():

            model.eval()
            for batch in val_loader:
                batch = batch.to(device)

                predictions = model(batch)

                metrics_val(preds=predictions, target=output_scaler(batch.target_vector), power_flow_predictions=output_scaler.inverse(predictions), batch=batch)

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
