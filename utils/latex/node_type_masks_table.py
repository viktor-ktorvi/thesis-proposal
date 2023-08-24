from mlpf.data.masks.optimal_power_flow import BusTypeMasks
from pylatex import Tabular, NoEscape, Table, Label, Marker

from utils.latex.metrics import inline_math


def node_type_mask_table(marker: Marker) -> Table:
    node_type_masks = Tabular("|c|c|c|c|c|")
    node_type_masks.add_hline()
    node_type_masks.add_row(("Node type", "P", "Q", "V", NoEscape(inline_math("\\theta"))))
    node_type_masks.add_hline()

    for node_type, mask in vars(BusTypeMasks).items():
        if not node_type.startswith('_'):
            node_type_masks.add_row((node_type, *map(int, mask)))
            node_type_masks.add_hline()

    table = Table(position="h")
    table.append(NoEscape(r'\centering'))
    table.append(node_type_masks)
    table.add_caption("Value mask in relation to the node type.")
    table.append(Label(marker))
    return table
