from typing import List

import numpy as np

from numpy.typing import NDArray

from pylatex import MultiColumn, MultiRow, Tabular, NoEscape

from utils.latex.metrics import expected_relative_absolute_error, expected_error, max_relative_absolute_error, max_error, inline_math


def middle_columns_alignment(num_small_columns: int, num_large_columns: int) -> str:
    """
    Create the table alignment argument for tabular by centering each small column and adding double vertical lines where the large columns end.

    :param num_small_columns:
    :param num_large_columns:
    :return: Alignment argument for tabular.
    """

    alignment = "|"
    for i in range(num_small_columns):
        alignment += "c|"
        if (i + 1) % (num_small_columns // num_large_columns) == 0 and i != num_small_columns - 1:
            alignment += "|"

    return alignment


def first_columns_alignment(num_first_columns: int):
    alignment = "|"
    for _ in range(num_first_columns):
        alignment += "c|"

    return alignment


def last_columns_alignment(num_last_columns: int):
    alignment = "|"
    for _ in range(num_last_columns):
        alignment += "c|"

    return alignment


def double_header_table(first_columns_names: List[str],
                        first_columns_content: List[List[str]],
                        large_column_names: List[str],
                        small_column_names: List[str],
                        content: List[NDArray],
                        last_columns_names: List[str] = (),
                        last_columns_contents: List[List[str]] = (),
                        num_decimals: int = 3) -> str:
    # first columns
    num_first_columns = len(first_columns_names)
    assert len(first_columns_content) == num_first_columns

    # middle columns
    num_large_columns = len(large_column_names)
    num_small_columns = len(small_column_names)

    assert num_small_columns // num_large_columns == num_small_columns / num_large_columns

    large_column_size = num_small_columns // num_large_columns

    # last columns
    num_last_columns = len(last_columns_names)
    assert len(last_columns_contents) == num_last_columns

    # height
    # TODO could check content as well
    content_height = len(first_columns_content[0])

    for column in first_columns_content:
        assert len(column) == content_height

    first_alignment = first_columns_alignment(num_first_columns)
    middle_alignment = middle_columns_alignment(num_small_columns, num_large_columns)
    last_alignment = last_columns_alignment(num_last_columns)
    table = Tabular(f"{first_alignment}{middle_alignment}{last_alignment}")

    # header
    table.add_hline()
    table.add_row(
        (
            *[MultiRow(size=2, data=first_columns_name) for first_columns_name in first_columns_names],
            *[MultiColumn(size=large_column_size,
                          align="c||" if i != num_large_columns - 1 or (i == num_large_columns - 1 and num_last_columns > 0) else "c|",
                          data=large_column_names[i]) for i in range(num_large_columns)],
            *[MultiRow(size=2, data=last_columns_name) for last_columns_name in last_columns_names],
        )
    )
    table.add_hline(num_first_columns + 1, num_small_columns + num_first_columns)
    table.add_row(["" for _ in range(num_first_columns)] + small_column_names + ["" for _ in range(num_last_columns)])
    table.add_hline()

    # content
    for i in range(content_height):

        first_columns_row_values = [first_columns_content[j][i] for j in range(num_first_columns)]

        content_row_values = []
        for j in range(num_large_columns):
            content_row_values += [f"{content[j][i, k]:.{num_decimals}f}" for k in range(content[j].shape[1])]

        last_columns_row_values = [last_columns_contents[j][i] for j in range(num_last_columns)]

        table.add_row((*first_columns_row_values, *content_row_values, *last_columns_row_values))

        table.add_hline()

    return "\\makegapedcells" + table.dumps()


def main():
    # TODO docs
    #  functional decomposition
    #  bold certain values

    large_column_names = ["single-topology", "multi-topology"]
    small_column_names = [NoEscape("MLP$_{global}$"), "GCN", "GAT",
                          NoEscape("MLP$_{local}$"), "GCN", "GAT"]

    metric_names = [
        inline_math(expected_relative_absolute_error("P")),
        inline_math(expected_relative_absolute_error("Q")),
        inline_math(expected_error("V_{max}")),
        inline_math(max_relative_absolute_error("P")),
        inline_math(max_relative_absolute_error("Q")),
        inline_math(max_error("V_{max}")),
    ]

    num_decimals = 3

    data_tables = [np.random.randn(len(metric_names), len(small_column_names) // len(large_column_names)) for _ in range(len(large_column_names))]

    first_column_names = ["Metric"]
    first_columns_content = [metric_names]
    last_columns_names = ["Unit"]
    last_columns_contents = [[inline_math("\\cdot"),
                              inline_math("\\cdot"),
                              "p.u.",
                              inline_math("\\cdot"),
                              inline_math("\\cdot"),
                              "p.u."]]
    latex_table = double_header_table(first_columns_names=first_column_names,
                                      first_columns_content=first_columns_content,
                                      large_column_names=large_column_names,
                                      small_column_names=small_column_names,
                                      last_columns_names=last_columns_names,
                                      last_columns_contents=last_columns_contents,
                                      content=data_tables,
                                      num_decimals=num_decimals)
    print(latex_table)

    with open("latex_example/table_with_two_headers/main.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    main()
