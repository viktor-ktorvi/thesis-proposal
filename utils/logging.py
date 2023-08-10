from typing import Dict

from torchmetrics import MetricCollection

from mlpf.utils.logging import clean_metric_name, get_unit


def collect_log(metrics_train: MetricCollection, metrics_val: MetricCollection) -> Dict:
    """
    Collect the metric values for logging.

    :param metrics_train: Train metric collection.
    :param metrics_val: Validation metric collection.
    :return: Dictionary of logs.
    """
    logs = {}

    overall_metrics_train = metrics_train.compute()
    overall_metrics_val = metrics_val.compute()
    for key in overall_metrics_train.keys():
        log_unit = get_unit(metrics_train[key]).replace("/", " per ")

        logs[f"train/{clean_metric_name(key)}{log_unit}"] = overall_metrics_train[key]
        logs[f"val/{clean_metric_name(key)}{log_unit}"] = overall_metrics_val[key]

    return logs
