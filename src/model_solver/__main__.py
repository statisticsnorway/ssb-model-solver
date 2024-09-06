"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Model Solver."""


if __name__ == "__main__":
    main(prog_name="ssb-model-solver")  # pragma: no cover
