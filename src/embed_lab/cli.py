import typer
from pathlib import Path
from typing import Annotated
from importlib import resources
from importlib.abc import Traversable

# from template import baseline

TEMPLATES = {
    "baseline": {
        "name": "Baseline",
        "description": "A minimal starting point with a single baseline experiment script. Perfect for quick prototyping.",
        # "module": baseline,
    },
}

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
        f"🧪 Initializing Embed Lab in {base_path}...",
        fg=typer.colors.BLUE,
        bold=True,
    )

    # Show template selection menu
    options = list(TEMPLATES.keys())
    typer.secho("\nPlease select a template:", bold=True)

    for i, key in enumerate(options, start=1):
        tmpl = TEMPLATES[key]
        typer.secho(
            f"{i}. {tmpl['name']}",
            fg=typer.colors.CYAN,
            bold=True,
        )
        typer.echo(f"   {tmpl['description']}\n")

    # Interactive selection with validation
    while True:
        try:
            choice_input = typer.prompt(
                "Enter the number of your chosen template",
                default="1",
            )
            choice = int(choice_input)
            if 1 <= choice <= len(options):
                break
            typer.secho(
                f"Please enter a number between 1 and {len(options)}.",
                fg=typer.colors.RED,
            )
        except ValueError:
            typer.secho(
                "Invalid input – please enter a number.",
                fg=typer.colors.RED,
            )

    selected_key = options[choice - 1]
    selected = TEMPLATES[selected_key]

    typer.secho(
        f"\nSelected template: {selected['name']}",
        fg=typer.colors.GREEN,
        bold=True,
    )

    # Get the root Traversable for the selected template
    template_root = resources.files(selected["module"])

    # Perform the copy
    copy_recursive(template_root, base_path, base_path)

    typer.secho(
        "\n✨ Done! Your Embed Lab is initialized.",
        fg=typer.colors.BLUE,
        bold=True,
    )
    typer.echo("\nNext steps:")
    typer.echo("   • Explore the created files.")
    typer.echo(
        "   • Check the experiments/ directory for runnable scripts."
    )
    typer.echo(
        "   • Example command: python experiments/exp_01_baseline.py"
    )
    typer.echo(
        "     (Script names may vary depending on the chosen template.)"
    )


if __name__ == "__main__":
    app()
