from torchmetrics import MeanSquaredError, R2Score, MetricCollection

from mlpf.loss.torch.metrics.active import (
    MeanActivePowerError,
    MeanRelativeActivePowerError,
    MaxActivePowerError,
    MaxRelativeActivePowerError
)

from mlpf.loss.torch.metrics.reactive import (
    MeanReactivePowerError,
    MeanRelativeReactivePowerError,
    MaxReactivePowerError,
    MaxRelativeReactivePowerError
)

from mlpf.loss.torch.metrics.bounds.active import (
    MeanUpperActivePowerError,
    MeanLowerActivePowerError,
    MaxUpperActivePowerError,
    MinLowerActivePowerError
)

from mlpf.loss.torch.metrics.bounds.reactive import (
    MeanUpperReactivePowerError,
    MeanLowerReactivePowerError,
    MaxUpperReactivePowerError,
    MinLowerReactivePowerError
)

from mlpf.loss.torch.metrics.bounds.voltage import (
    MeanUpperVoltageError,
    MeanLowerVoltageError,
    MaxUpperVoltageError,
    MinLowerVoltageError
)

from mlpf.loss.torch.metrics.costs import (
    MeanActivePowerCost,
    MeanRelativeActivePowerCost,
    MaxActivePowerCost,
    MaxRelativeActivePowerCost,
    MinActivePowerCost,
    MinRelativeActivePowerCost
)


def train_metrics(output_size: int) -> MetricCollection:
    """
    A few metrics for training so that it's not too slow.
    :param output_size: Target dimension.
    :return:
    """
    return MetricCollection(
        [
            MeanSquaredError(),
            R2Score(output_size),
            MeanActivePowerError(),
            MeanRelativeActivePowerError(),
            MeanRelativeActivePowerCost()
        ]
    )


def val_metrics(output_size: int) -> MetricCollection:
    """
    A few metrics for training so that it's not too slow.
    :param output_size: Target dimension.
    :return:
    """
    return train_metrics(output_size)


def full_evaluation_metrics(output_size: int) -> MetricCollection:
    """
    A comprehensive list of metrics.
    :param output_size: Target dimension.
    :return:
    """
    return MetricCollection(
        [
            MeanSquaredError(),
            R2Score(output_size),
            MeanActivePowerError(),
            MeanRelativeActivePowerError(),
            MaxActivePowerError(),
            MaxRelativeActivePowerError(),

            MeanReactivePowerError(),
            MeanRelativeReactivePowerError(),
            MaxReactivePowerError(),
            MaxRelativeReactivePowerError(),

            MeanUpperActivePowerError(),
            MeanLowerActivePowerError(),
            MaxUpperActivePowerError(),
            MinLowerActivePowerError(),

            MeanUpperReactivePowerError(),
            MeanLowerReactivePowerError(),
            MaxUpperReactivePowerError(),
            MinLowerReactivePowerError(),

            MeanUpperVoltageError(),
            MeanLowerVoltageError(),
            MaxUpperVoltageError(),
            MinLowerVoltageError(),

            MeanActivePowerCost(),
            MeanRelativeActivePowerCost(),
            MaxActivePowerCost(),
            MaxRelativeActivePowerCost(),
            MinActivePowerCost(),
            MinRelativeActivePowerCost()
        ]
    )
