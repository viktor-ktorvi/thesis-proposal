from pylatex import Document, Tabular, MultiColumn, LineBreak, NewLine
from utils.latex.double_header_table import middle_columns_alignment

doc = Document()

table1 = Tabular('|c|c|c|c|')
table1.add_hline()
table1.add_row((MultiColumn(4, align='|c|', data='Multicolumn'),))
table1.add_hline()
table1.add_row((1, 2, 3, 4))
table1.add_hline()
table1.add_row((5, 6, 7, 8))
table1.add_hline()
row_cells = ('9', MultiColumn(3, align='|c|', data='Multicolumn not on left'))
table1.add_row(row_cells)
table1.add_hline()

num_large_columns = 2
num_small_columns_per_large = 3
table2 = Tabular("|c|" + middle_columns_alignment(num_small_columns=num_large_columns * num_small_columns_per_large, num_large_columns=num_large_columns))
table2.add_row(("Metric", MultiColumn(num_small_columns_per_large, data='Multicolumn1'), MultiColumn(num_small_columns_per_large, data='Multicolumn2')))

doc.append(table1)
doc.append(LineBreak())
doc.append(table2)
doc.generate_tex(filepath="table")

import os

pdf_path = os.path.join(os.getcwd(), "out", "table.pdf")
if os.path.exists(pdf_path):
    os.remove(pdf_path)
os.system(
    "pdflatex -file-line-error -interaction=nonstopmode -synctex=1 -output-format=pdf -output-directory=C:/Users/todos/PycharmProjects/thesis_proposal/pylatex_examples/out -aux-directory=C:/Users/todos/PycharmProjects/thesis_proposal/pylatex_examples/auxil table.tex")

os.startfile(os.path.join(os.getcwd(), "out", "table.pdf"))
