import typer
from pathlib import Path
from typing import Annotated

from embed_lab import templates

app = typer.Typer(name="emb", add_completion=True)


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

    structure: dict[Path, str | None] = {
        # Directories
        base_path / "inventory": None,
        base_path / "experiments": None,
        base_path / "results": None,
        # Inventory
        base_path / "inventory" / "__init__.py": "",
        base_path
        / "inventory"
        / "datasets.py": templates.TEMPLATE_DATASETS,
        base_path
        / "inventory"
        / "preprocess.py": templates.TEMPLATE_PREPROCESS,
        base_path
        / "inventory"
        / "train.py": templates.TEMPLATE_TRAIN,
        # Experiments
        base_path / "experiments" / "__init__.py": "",
        base_path
        / "experiments"
        / "exp_01_baseline.py": templates.TEMPLATE_EXP_01_BASELINE,
        # Misc
        base_path / "results" / ".gitkeep": "",
        base_path
        / ".gitignore": templates.TEMPLATE_GITIGNORE,
        base_path / "main.py": templates.TEMPLATE_MAIN,
    }

    typer.secho(
        f"🧪 Initializing Embed Lab in {base_path.name}...",
        fg=typer.colors.BLUE,
    )

    for file_path, content in structure.items():
        if content is None:
            file_path.mkdir(parents=True, exist_ok=True)
            continue

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists():
            typer.secho(
                f"  . Skipped {file_path.relative_to(base_path)} (exists)",
                fg=typer.colors.YELLOW,
            )
            continue

        file_path.write_text(
            content, encoding="utf-8"
        )  # pathlib.Path.write_text creates/writes text files. [web:32]
        typer.secho(
            f"  + Created {file_path.relative_to(base_path)}",
            fg=typer.colors.GREEN,
        )

    typer.secho(
        "\n✨ Done! Try:", fg=typer.colors.BLUE, bold=True
    )
    typer.echo("   python experiments/exp_01_baseline.py")
