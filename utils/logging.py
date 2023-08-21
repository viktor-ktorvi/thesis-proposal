from typing import Dict

from torchmetrics import MetricCollection

from mlpf.utils.logging import clean_metric_name, get_unit


def collect_log(metrics: MetricCollection, subcategory: str) -> Dict:
    """
    Collect metrics in a dictionary, generate clean names for the metrics and add the units.
    :param metrics: Metric collection.
    :param subcategory: The subcategory of logs in wandb to be appended to the name like: 'subcategory/name'.
    :return:
    """
    logs = {}

    overall_metrics = metrics.compute()
    for key in overall_metrics.keys():
        log_unit = get_unit(metrics[key]).replace("/", " per ")

        logs[f"{subcategory}/{clean_metric_name(key)}{log_unit}"] = overall_metrics[key]

    return logs
