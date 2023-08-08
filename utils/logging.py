import re

from typing import Any, Dict

from torchmetrics import MetricCollection


def get_unit(metric: Any) -> str:
    """
    Extract the unit of the metric as a string if it exists and wrap it up in square brackets.

    :param metric: Metric.
    :return: String of [unit]
    """
    unit = getattr(metric, 'unit', None)  # get metric unit if it exists
    return f" [{unit}]" if unit is not None else ''


def clean_metric_name(metric_name: str) -> str:
    """
    Add a space before capital letters and then turn everything to lowercase.

    :param metric_name: Name of the metric. Usually the string of the class name.
    :return: Clean name.
    """
    return re.sub(r"(\w)([A-Z])", r"\1 \2", metric_name).lower()


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
