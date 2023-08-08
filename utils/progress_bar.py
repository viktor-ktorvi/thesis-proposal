from typing import List

from torchmetrics import MetricCollection
from tqdm import tqdm

from utils.logging import clean_metric_name


class CustomProgressBar:
    def __init__(self,
                 metric_keys: List[str],
                 total: int,
                 metric_value_width: int = 15,
                 metric_name_width: int = 50,
                 metric_value_decimals: int = 4,
                 description_width: int = 100
                 ):
        """
        Create the custom progress bar and print its header. The class will create a horizontally parallel
        progress bar for each of the metric keys and update its description with the corresponding metric.

        :param metric_keys: Keys of the metric collection.
        :param total: Total number of iterations in the progress bar.
        :param metric_value_width:
        :param metric_name_width:
        :param metric_value_decimals:
        :param description_width:
        """
        self.metric_value_width = metric_value_width
        self.metric_name_width = metric_name_width
        self.metric_value_decimals = metric_value_decimals
        self.description_width = description_width

        print(f"\n{self.header()}\n")

        self.progress_bars: dict = {
            key: value for key, value in zip(
                metric_keys,
                [tqdm(total=total, position=i) for i in range(len(metric_keys))]
            )
        }

    def header(self) -> str:
        return f"{'Training':^{self.metric_value_width}} | {'Validation':^{self.metric_value_width}} {'Metric':{self.metric_name_width}} Unit"

    def update(self, metrics_train: MetricCollection, metrics_val: MetricCollection, how_much: int = 1):
        """
        Update each of the progress bars.

        :param metrics_train:
        :param metrics_val:
        :param how_much: By how much to update the progress bar.
        :return:
        """
        overall_metrics_train = metrics_train.compute()
        overall_metrics_val = metrics_val.compute()

        for key in overall_metrics_train.keys():
            unit = getattr(metrics_train[key], 'unit', None)  # get metric unit if it exists
            unit_in_brackets = f" [{unit}]" if unit is not None else ''

            description = f"{overall_metrics_train[key]:^{self.metric_value_width}.{self.metric_value_decimals}f} | {overall_metrics_val[key]:^{self.metric_value_width}.{self.metric_value_decimals}f} {clean_metric_name(key):{self.metric_name_width}}{unit_in_brackets}"

            self.progress_bars[key].set_description(f"{description:{self.description_width}}")
            self.progress_bars[key].update(how_much)
