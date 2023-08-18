from pylatex import NoEscape, Command


def abs_brackets(symbol: str) -> NoEscape:
    return NoEscape(f"\\Bigl|{symbol}\\Bigr|")


def error(symbol: str) -> NoEscape:
    return NoEscape(f"\\Delta {symbol}")


def expectation() -> str:
    return Command("mathbb", "E").dumps()


def relative_absolute_error(symbol: str) -> NoEscape:
    return abs_brackets(Command("frac", [error(symbol), symbol]).dumps())


def expected_relative_absolute_error(symbol: str) -> NoEscape:
    return NoEscape(f"{expectation()} \\ {relative_absolute_error(symbol)}")


def expected_error(symbol: str) -> NoEscape:
    return NoEscape(f"{expectation()} \\ {error(symbol)}")


def max_relative_absolute_error(symbol: str) -> NoEscape:
    return NoEscape(f"max \\ {relative_absolute_error(symbol)}")


def max_error(symbol: str) -> NoEscape:
    return NoEscape(f"max \\ {error(symbol)}")


def inline_math(symbol: str) -> NoEscape:
    return NoEscape(f"${symbol}$")
