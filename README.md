# thesis-proposal

Masters thesis proposal

```
git clone https://github.com/viktor-ktorvi/thesis-proposal.git
cd thesis-proposal

conda env create -f environment.yml
conda activate thesis_proposal_env

wandb login
```

Follow login instructions.

Train a linear model:

```
python -m optimal_power_flow.supervised.train model=linear
```

Train a GCN:

```
python -m optimal_power_flow.supervised.train model=gcn
```

Run a wandb sweep:

```
wandb sweep ./optimal_power_flow/supervised/sweep_configs/gcn.yaml
```