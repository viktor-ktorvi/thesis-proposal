import os

from pylatex import Command, Subsection, NoEscape


def main():
    current_directory = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

    introduction = Subsection("Introduction", numbering=False)
    problem_formulation = Subsection("Problem formulation", numbering=False)
    problem_formulation.append(Command("input", "introduction/problem_formulation"))

    motivation = Subsection("Motivation", numbering=False)
    motivation.append(Command("input", "introduction/motivation"))

    introduction.append(problem_formulation)
    introduction.append(motivation)
    with open(os.path.join(current_directory, "main.tex"), "w") as f:
        f.write(introduction.dumps())


if __name__ == "__main__":
    main()
