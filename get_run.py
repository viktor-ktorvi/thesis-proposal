import os
import hydra
import json
import wandb

from pprint import pprint
from matplotlib import pyplot as plt


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "configs"), config_name="default")
def main(cfg):
    api = wandb.Api()

    sweep_path = "viktor-ktorvi/thesis-proposal/r4ised6c"
    sweep = api.sweep(sweep_path)
    xy_values = {}

    config_parameter_key = "num_layers"
    metric_key = "val/r2 score"
    for run in sweep.runs:
        config = json.loads(run.json_config)
        config_parameter = config['config']['value']['model'][config_parameter_key]
        metric_value = run.summary[metric_key]

        if config_parameter not in xy_values:
            xy_values[config_parameter] = [metric_value]
        else:
            xy_values[config_parameter].append(metric_value)

    pprint(xy_values)

    for x in xy_values:
        xy_values[x] = max(xy_values[x])

    pprint(xy_values)

    xy_values = dict(sorted(xy_values.items()))

    plt.figure()
    plt.plot(xy_values.keys(), xy_values.values())
    plt.title(f"{metric_key} VS {config_parameter_key}")
    plt.xlabel(config_parameter_key)
    plt.ylabel(metric_key)
    plt.show()
    kjkszpj = None


if __name__ == "__main__":
    main()
