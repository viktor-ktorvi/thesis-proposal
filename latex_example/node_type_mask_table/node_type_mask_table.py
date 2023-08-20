import random

from pylatex import Tabular, NoEscape
from mlpf.data.masks.power_flow import BusTypeMasks
from utils.latex.metrics import inline_math

table_alignment = "|c|c|c|c|c|"

mask_rule_table = Tabular(table_alignment)
mask_rule_table.add_hline()
mask_rule_table.add_row(("Node type", "P", "Q", "V", NoEscape(inline_math("\\theta"))))
mask_rule_table.add_hline()

for node_type, mask in vars(BusTypeMasks).items():
    if not node_type.startswith('_'):
        mask_rule_table.add_row((node_type, *map(int, mask)))
        mask_rule_table.add_hline()

node_types = [key for key in vars(BusTypeMasks).keys() if not key.startswith("_")]

print(node_types)
mask_example_table = Tabular(table_alignment)
mask_example_table.add_hline()
mask_example_table.add_row(("Node type", "P", "Q", "V", NoEscape(inline_math("\\theta"))))
mask_example_table.add_hline()

for i in range(6):
    random_node_type = random.choice(node_types)
    mask_example_table.add_row((random_node_type, *map(int, vars(BusTypeMasks)[random_node_type])))
    mask_example_table.add_hline()

mask_example_table.add_row(*(NoEscape(inline_math("\\vdots")) for _ in range(mask_example_table.width)))

mask_example_table.add_hline()

for i in range(5):
    random_node_type = random.choice(node_types)
    mask_example_table.add_row((random_node_type, *map(int, vars(BusTypeMasks)[random_node_type])))
    mask_example_table.add_hline()

mother_table = Tabular("cc")
mother_table.add_row((mask_rule_table, mask_example_table))

with open("latex_example/node_type_mask_table/main.tex", "w") as f:
    content = mother_table.dumps()
    print(content)
    f.write(content)
