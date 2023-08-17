from typing import List

import numpy as np

from numpy.typing import NDArray


def multiheader_table(large_column_names: List[str],
                      small_column_names: List[str],
                      metric_names: List[str],
                      first_column_name: str,
                      data_tables: List[NDArray],
                      num_decimals: int = 3) -> str:
    num_large_columns = len(large_column_names)
    num_small_columns = len(small_column_names)
    num_metrics = len(metric_names)
    assert num_small_columns // num_large_columns == num_small_columns / num_large_columns

    alignment = ""
    for i in range(num_small_columns):
        alignment += "c|"
        if (i + 1) % (num_small_columns // num_large_columns) == 0 and i != num_small_columns - 1:
            alignment += "|"
    begin = f"\\makegapedcells\\begin{{tabular}}{{|c||{alignment}}}"

    first_column = f"\\multirow{{2}}{{*}}{{{first_column_name}}}"

    large_columns = ""
    for i in range(num_large_columns):
        if i == 0:
            additional_vline = "|"
        else:
            additional_vline = ""

        large_columns += f"\\multicolumn{{3}}{{c|{additional_vline}}}{{{large_column_names[i]}}}"

        if i != num_large_columns - 1:
            large_columns += " & "
        else:
            large_columns += " \\\\"

    large_columns += f" \\cline{{2-{num_small_columns + 1}}}"

    small_columns = ""
    for i in range(num_small_columns):
        small_columns += f"& {small_column_names[i]} "

    small_columns += "\\\\ \\hline"
    header = first_column + " & " + large_columns + "\n" + small_columns

    content = ""
    for i in range(num_metrics):
        content += metric_names[i]
        for j in range(num_large_columns):
            for k in range(data_tables[j].shape[1]):
                content += f" & {data_tables[j][i, k]:.{num_decimals}f}"

        content += " \\\\ \\hline\n"

    end = "\\end{tabular}"
    table = begin + "\n" + "\\hline\n" + header + "\n" + content + "\n" + end

    return table


def main():
    # TODO docs
    #  functional decomposition
    #  bold certain values

    first_column_name = "Metric"

    large_column_names = ["single-topology", "multi-topology"]
    small_column_names = ["MLP$_{global}$", "GCN", "GAT", "MLP$_{local}$", "GCN", "GAT"]

    # metric_names = [
    #     "$\\mathbb{E} \\ \\frac{\\Delta P}{P}$",
    #     "$max \\ \\frac{\\Delta P}{P}$",
    #     "$\\mathbb{E} \\ \\frac{\\Delta Q}{Q}$",
    #     "$max \\ \\frac{\\Delta Q}{Q}$",
    #     "$\\mathbb{E} \\ \\Delta V_{max}$",
    #     "$max \\ \\Delta V_{max}$",
    # ]

    def error(symbol: str) -> str:
        return f"\\Delta {symbol}"

    def expectation() -> str:
        return "\\mathbb{E}"

    def relative_absolute_error(symbol: str) -> str:
        return f"|\\frac{{{error(symbol)}}}{{{symbol}}}|"

    def expected_relative_absolute_error(symbol: str) -> str:
        return f"${expectation()} \\ {relative_absolute_error(symbol)}$"

    def expected_error(symbol: str) -> str:
        return f"${expectation()} \\ {error(symbol)}$"

    def max_relative_absolute_error(symbol: str) -> str:
        return f"$max \\ {relative_absolute_error(symbol)}$"

    def max_error(symbol: str) -> str:
        return f"$max \\ {error(symbol)}$"

    metric_names = [
        expected_relative_absolute_error("P"),
        expected_relative_absolute_error("Q"),
        expected_error("V_{max}"),
        max_relative_absolute_error("P"),
        max_relative_absolute_error("Q"),
        max_error("V_{max}"),
    ]

    num_decimals = 3

    data_tables = [np.random.randn(len(metric_names), len(small_column_names) // len(large_column_names)) for _ in range(len(large_column_names))]
    latex_table = multiheader_table(large_column_names=large_column_names,
                                    small_column_names=small_column_names,
                                    metric_names=metric_names,
                                    first_column_name=first_column_name,
                                    data_tables=data_tables,
                                    num_decimals=num_decimals)
    print(latex_table)

    with open("latex_example/table_with_two_headers/main.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
