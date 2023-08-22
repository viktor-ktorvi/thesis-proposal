import os

import hydra
import torch
import wandb

import torch_geometric as pyg

from mlpf.data.data.optimal_power_flow import OptimalPowerFlowData
from mlpf.data.loading.load_data import load_data
from mlpf.utils.progress_bar import CustomProgressBar
from mlpf.utils.standard_scaler import StandardScaler
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torchmetrics import MetricCollection
from tqdm import tqdm

from data.download import download
from models.gnn.get_model import get_model
from utils.logging import collect_log
from utils.metric_lists.supervised import train_metrics, val_metrics, full_evaluation_metrics


# TODO for multi-topology, it's probably gonna be crucial to inherit from the OPFData class and
#  include the limits into data.x. While overwriting data.x we could also delete the target info.

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
    # TODO tags
    wandb.init(project=cfg.wandb.project, mode=cfg.wandb.mode)

    # cross-update hydra and wandb configs
    cfg = OmegaConf.merge(cfg, dict(wandb.config))
    wandb.config.update({"config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)}, allow_val_change=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # TODO device from config

    # random seeds
    pyg.seed_everything(cfg.general.random_seed)

    # load data
    download(path=cfg.data.path, google_drive_url=cfg.data.google_drive_url)
    opf_data_list = [
        OptimalPowerFlowData(solved_ppc).to_pyg_data() for solved_ppc in tqdm(load_data(cfg.data.path, max_samples=cfg.general.max_samples), desc="Converting ppcs to data")
    ]

    data_train, data_val = train_test_split(opf_data_list, test_size=cfg.general.validation_split, random_state=cfg.general.random_seed)

    # delete the target info from the input features
    for data in data_train:
        data.x[~data.PQVA_mask] = 0.0

    for data in data_val:
        data.x[~data.PQVA_mask] = 0.0

    # dataloaders
    train_loader = DataLoader(data_train, batch_size=cfg.model.batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=cfg.model.batch_size, shuffle=False)

    # model
    model = get_model(cfg.model, data_train)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    criterion = torch.nn.MSELoss()

    output_size = data_train[0].x.shape[1]

    metrics_train = train_metrics(output_size).to(device)
    metrics_val = val_metrics(output_size).to(device)
    metrics_final_evaluation = full_evaluation_metrics(output_size).to(device)

    # if running from the IDE console, make sure to select 'emulate terminal' in the run configuration, otherwise the output will look bad
    progress_bar = CustomProgressBar(metrics_train.keys(), total=cfg.model.num_epochs)

    def validation(metrics: MetricCollection):
        with torch.no_grad():
            model.eval()
            for batch in val_loader:
                batch = batch.to(device)

                PQVA_prediction = model(batch)

                metrics(preds=PQVA_prediction,
                        target=batch.PQVA_matrix,
                        batch=batch)

    for epoch in range(cfg.model.num_epochs):

        # train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            PQVA_prediction = model(batch)

            # optimizing the scaled results difference could lead to nicer gradients in my experience
            loss = criterion(
                model.scale_output(PQVA_prediction, batch),
                model.scale_output(batch.PQVA_matrix, batch)
            )
            loss.backward()

            metrics_train(preds=PQVA_prediction,
                          target=batch.PQVA_matrix,
                          batch=batch)

            optimizer.step()

        # validate
        validation(metrics_val)

        progress_bar.update(metrics_train, metrics_val)

        wandb.log(collect_log(metrics_train, subcategory="train") | collect_log(metrics_val, subcategory="val"), step=epoch)

        metrics_train.reset()
        metrics_val.reset()

        # Evaluation on more metrics from time to time
        if epoch % round(cfg.model.num_epochs * cfg.general.full_evaluation_ratio) == 0:
            validation(metrics_final_evaluation)
            wandb.log(collect_log(metrics_final_evaluation, subcategory="eval"), step=epoch)
            metrics_final_evaluation.reset()

    # one final evaluation
    validation(metrics_final_evaluation)
    wandb.log(collect_log(metrics_final_evaluation, subcategory="eval"), step=cfg.model.num_epochs)


if __name__ == '__main__':
    main()
