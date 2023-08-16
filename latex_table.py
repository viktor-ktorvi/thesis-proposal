import pandas as pd

df = pd.DataFrame(dict(name=['Raphael', 'Donatello'],
                       age=[26, 45],
                       height=[181.23, 177.65]))
table_text = df.to_latex(
    index=False,
    formatters={"name": str.upper},
    float_format="{:.1f}".format,
)

with open("latex_example/table.tex", "w") as file:
    file.write(table_text)
