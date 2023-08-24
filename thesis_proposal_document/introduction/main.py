import os

from pylatex import Command, Subsection, NoEscape, NewLine, Marker, Ref

from utils.latex.node_type_masks_table import node_type_mask_table


def main():
    current_directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    introduction = Subsection("Introduction", numbering=False)
    introduction.append(Command("input", "introduction/basic_problem_statement"))
    introduction.append(Command("input", "introduction/motivation"))
    introduction.append(Command("input", "introduction/mathematical_formulation"))

    introduction.append(NewLine())
    introduction.append(NewLine())

    introduction.append(Command("input", "introduction/node_types_and_masks"))

    introduction.append(NewLine())
    introduction.append(NewLine())

    introduction.append(node_type_mask_table(Marker("nodemasks", prefix="tab")))

    with open(os.path.join(current_directory, "main.tex"), "w") as f:
        f.write(introduction.dumps())


if __name__ == "__main__":
    main()
