"""Debug script for testing linters and type checkers."""

from __future__ import annotations


def add_numbers(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def greet(name: str, excited: bool = False) -> str:
    """Return a greeting for the given name."""
    suffix = "!" if excited else "."
    return f"Hello, {name}{suffix}"


def main() -> None:
    """Run the debug script."""
    print(add_numbers(2, 3))
    print(greet("world", excited=True))


if __name__ == "__main__":
    main()