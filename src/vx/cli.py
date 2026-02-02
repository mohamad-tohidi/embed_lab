from __future__ import annotations

from importlib import resources
from importlib.abc import Traversable
from pathlib import Path
from typing import Annotated, Iterable, Iterator

import typer

import template

EXCLUDES: frozenset[str] = frozenset(
    {"__pycache__", "__init__.py"}
)


app = typer.Typer(
    name="vx",
    add_completion=True,
    no_args_is_help=True,
    help="Initialize an Embed Lab project from the bundled template.",
    context_settings={
        "help_option_names": ["-h", "--help"]
    },
)


def iter_template_files(
    root: Traversable,
    prefix: Path = Path(),
) -> Iterator[tuple[Traversable, Path]]:
    for item in root.iterdir():
        if item.name in EXCLUDES:
            continue

        rel = prefix / item.name
        if item.is_dir():
            yield from iter_template_files(item, rel)
        elif item.is_file():
            yield item, rel


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_template_files(
    items: Iterable[tuple[Traversable, Path]],
    dest_root: Path,
) -> tuple[int, int]:
    written = 0
    skipped = 0

    for src, rel in items:
        dst = dest_root / rel
        ensure_dir(dst.parent)

        if dst.exists():
            skipped += 1
            typer.secho(
                f"  . Skipped {rel} (exists)",
                fg=typer.colors.YELLOW,
            )
            continue

        dst.write_bytes(src.read_bytes())

        written += 1
        verb = "Created"
        typer.secho(
            f"  + {verb} {rel}", fg=typer.colors.GREEN
        )

    return written, skipped


@app.command(
    "init",
    help="Create/refresh a directory from the bundled template.",
)
def init(
    path: Annotated[
        Path,
        typer.Argument(
            help="Target directory to initialize."
        ),
    ] = Path("."),
) -> None:
    """
    Initialize an Embed Lab directory.

    What happens:
    - Copies the bundled `template` package contents into PATH.
    - Skips existing files by default (use --overwrite to replace them).
    - Excludes __pycache__ and __init__.py.
    """
    dest = path.expanduser().resolve()
    ensure_dir(dest)

    action = "Initializing"
    typer.secho(
        f"{action} Embed Lab in: {dest}",
        fg=typer.colors.BLUE,
    )

    template_root = resources.files(template)
    items = list(iter_template_files(template_root))

    typer.echo(f"This will process {len(items)} file(s).")
    typer.echo(
        "Existing files will be skipped unless --overwrite is set."
    )

    written, skipped = copy_template_files(
        items,
        dest,
    )

    suffix = "written"
    typer.secho(
        f"\nDone. {written} file(s) {suffix}, {skipped} skipped.",
        fg=typer.colors.BLUE,
        bold=True,
    )
    typer.echo("Try (from inside the directory):")
    typer.echo("  python experiments/exp_01_baseline.py")


if __name__ == "__main__":
    app()
