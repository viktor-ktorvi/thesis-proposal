from pylatex import Document, Command, NoEscape, Package

from introduction.main import main as introduction_main


def main():
    doc = Document(documentclass="report")

    doc.preamble.append(Package("amsmath"))
    doc.preamble.append(Package("amssymb"))
    doc.preamble.append(Package("hyperref"))
    doc.preamble.append(Package("indentfirst"))

    doc.preamble.append(Package("biblatex", options=["sorting=none"]))
    doc.preamble.append(Command("addbibresource", arguments=["references.bib"]))

    doc.preamble.append(Command("title", "Master thesis proposal"))
    doc.preamble.append(Command("author", "Viktor Todosijevic"))
    doc.preamble.append(Command("date", "August 2023"))
    doc.append(NoEscape(r"\maketitle"))

    introduction_main()
    doc.append(Command("input", NoEscape("introduction/main")))
    doc.append(Command("input", NoEscape("literature_overview/main")))
    doc.append(Command("input", NoEscape("methodology/main")))
    doc.append(Command("input", NoEscape("conclusion/main")))

    doc.append(Command("printbibliography"))

    with open("main.tex", "w") as f:
        f.write(doc.dumps())


if __name__ == "__main__":
    main()
