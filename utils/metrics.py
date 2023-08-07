from typing import List

from mlpf.loss.torch.metrics.active import MeanActivePowerError, MeanRelativeActivePowerError
from mlpf.loss.torch.metrics.bounds.active import MeanUpperActivePowerError, MeanLowerActivePowerError
from mlpf.loss.torch.metrics.bounds.reactive import MeanUpperReactivePowerError, MeanLowerReactivePowerError
from mlpf.loss.torch.metrics.bounds.voltage import MeanUpperVoltageError, MeanLowerVoltageError
from mlpf.loss.torch.metrics.costs import MeanActivePowerCost, MeanRelativeActivePowerCost
from mlpf.loss.torch.metrics.reactive import MeanReactivePowerError, MeanRelativeReactivePowerError
from torchmetrics import MetricCollection, MeanSquaredError, R2Score, Metric


def _get_pf_metrics() -> List:
    """
    :return: Return just power flow metrics.
    """
    return [
        MeanActivePowerError(),
        MeanRelativeActivePowerError(),
        MeanReactivePowerError(),
        MeanRelativeReactivePowerError(),
    ]


def _get_opf_metrics() -> List:
    """
    :return: Return just optimal power flow metrics.
    """
    return [
        MeanActivePowerCost(),
        MeanRelativeActivePowerCost(),
        MeanUpperVoltageError(),
        MeanLowerVoltageError(),
        MeanUpperActivePowerError(),
        MeanLowerActivePowerError(),
        MeanUpperReactivePowerError(),
        MeanLowerReactivePowerError()
    ]


def _get_pf_and_opf_metrics() -> List:
    """
    :return: Return combined PF and OPF metrics.
    """
    return _get_pf_metrics() + _get_opf_metrics()


def optimal_power_flow_metrics() -> MetricCollection:
    """
    :return: Return combined PF and OPF metric collection.
    """
    return MetricCollection(*_get_pf_and_opf_metrics())


def optimal_power_flow_metrics_with_mse() -> MetricCollection:
    """
    :return: Return combined PF and OPF + MSE metric collection.
    """
    combined_metrics = [MeanSquaredError()] + _get_pf_and_opf_metrics()
    return MetricCollection(*combined_metrics)


def optimal_power_flow_metrics_with_mse_and_r2score(num_outputs: int):
    """
    :return: Return combined PF and OPF + MSE + R2Score metric collection.
    """
    combined_metrics = [MeanSquaredError(), R2Score(num_outputs=num_outputs)] + _get_pf_and_opf_metrics()
    return MetricCollection(*combined_metrics)
