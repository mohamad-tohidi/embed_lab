import typer
from pathlib import Path
from typing import Annotated
from importlib import resources
from importlib.abc import Traversable


import template

app = typer.Typer(name="emb", add_completion=True)


def copy_recursive(
    source: Traversable, dest: Path, base_path: Path
) -> None:
    """
    Recursively copies files from a Traversable (package resource) to a destination Path.
    """
    for item in source.iterdir():
        # Skip package internals and cache
        if item.name in ["__pycache__", "__init__.py"]:
            continue

        target_path = dest / item.name

        if item.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            copy_recursive(item, target_path, base_path)
        elif item.is_file():
            # Check if exists to skip
            if target_path.exists():
                typer.secho(
                    f"  . Skipped {target_path.relative_to(base_path)} (exists)",
                    fg=typer.colors.YELLOW,
                )
                continue

            # Copy content
            target_path.write_bytes(item.read_bytes())
            typer.secho(
                f"  + Created {target_path.relative_to(base_path)}",
                fg=typer.colors.GREEN,
            )


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory to initialize the lab in"
        ),
    ] = Path("."),
) -> None:
    base_path: Path = path.resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    typer.secho(
        f"🧪 Initializing Embed Lab in {base_path.name}...",
        fg=typer.colors.BLUE,
    )

    # Get the root Traversable object for the template package [cite:web:1]
    template_root = resources.files(template)

    # Start the recursive copy
    copy_recursive(template_root, base_path, base_path)

    typer.secho(
        "\n✨ Done! Try:", fg=typer.colors.BLUE, bold=True
    )
    typer.echo("   python experiments/exp_01_baseline.py")


if __name__ == "__main__":
    app()
