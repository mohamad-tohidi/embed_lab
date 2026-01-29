# Embed Lab (`embed_lab`)

Embed Lab is a small Python tool that scaffolds a repeatable workspace for fine-tuning information retrieval (IR) embedding models. It gives you a clean project layout, runnable experiment files, and an “inventory” layer where you centralize reusable training/evaluation code instead of rewriting it per experiment.

The goal is **consistency**: one place to define datasets, preprocessing, training, evaluation, and plotting, and many small experiment scripts that compose those building blocks.

## Why we created this

Fine-tuning IR embedding models tends to drift into ad-hoc notebooks and one-off scripts: inconsistent data loading, hard-to-reproduce runs, and experiments that are difficult to compare. Embed Lab exists to:

- Centralize the core pipeline (datasets, preprocessing, training, evaluation) in `inventory/`
- Keep experiments as small, explicit Python files in `experiments/`
- Make results reproducible by writing artifacts to `results/<experiment_name>/`
- Reduce setup time by generating a working template with `emb init`

Although the starter template ships with a Sentence-Transformers (SBERT) example, the structure is intended to generalize to other training stacks and tasks.

## What you get

Running `emb init` generates a “lab” folder with:

- `inventory/`: reusable modules (datasets, preprocess, train, evaluate, plotting)
- `experiments/`: runnable experiment scripts (`exp_01_baseline.py`, etc.)
- `data/`: JSONL splits (train/dev/gold) with a tiny sample dataset
- `results/`: where all run artifacts are written (git-ignored)

This separation is deliberate: `inventory/` changes slowly, `experiments/` grows over time as your research log.

## Install

Embed Lab is designed to be used with `uv`, but it works with any environment manager.

```bash
uv add embed-lab
```

Confirm the CLI is available:

```bash
emb --help
```

## Quickstart

Create a new lab in the current directory:

```bash
emb init .
```

Install the example pipeline dependencies (the generated template uses these):

```bash
uv add sentence-transformers datasets plotly
```

Run the baseline experiment:

```bash
uv run experiments/exp_01_baseline.py
```

You should see artifacts under:

- `results/exp_01_baseline/final/` (saved model)
- `results/exp_01_baseline/eval/` (metrics + evaluator CSV)
- `results/exp_01_baseline/plots/` (interactive HTML charts)

## How it works

Embed Lab itself is a CLI that writes a curated project skeleton to disk. The generated code is intentionally simple and editable, so teams can “own” their lab and adapt it.

The default workflow looks like:

1. Load examples from `data/*.jsonl`
2. Preprocess (optionally)
3. Train and save a model
4. Evaluate on a gold split
5. Plot training curves and metrics

A minimal example of the generated experiment entrypoint:

```python
from pathlib import Path

from inventory.datasets import load_splits
from inventory.preprocess import preprocess
from inventory.train import train

def main() -> None:
    run_dir = Path("results") / "exp_01_baseline"
    train_raw, dev_raw, gold_raw = load_splits(Path("data"))
    train_dir = train(preprocess(train_raw), preprocess(dev_raw), run_dir)

if __name__ == "__main__":
    main()
```

## Project status and roadmap

Embed Lab is intentionally small today: it scaffolds a working lab and stays out of your way. The long-term plan is to make the CLI smarter and the templates more diverse.

Planned directions:

- Data validation helpers in the CLI: duplicate detection across splits, overlap/leak checks, schema validation, basic stats
- Multiple templates: pairwise contrastive, in-batch negatives (e.g., MultipleNegativesRankingLoss), hard-negative mining layouts, cross-encoder reranking experiments
- Extensible template registry: add templates without touching core CLI logic
- Better run metadata: automatic run manifests (model, loss, hyperparams, git commit, dataset hash)

If you want to contribute, these are great entry points.

## Contributing

Contributions are welcome: templates, CLI improvements, docs, and bug fixes.

Suggested contribution workflow:

1. Fork the repo
2. Create a feature branch
3. Add tests or a small reproducible example if applicable
4. Open a PR with a clear description and rationale

Design preferences:

- Keep code minimal and readable
- Favor type annotations
- Avoid adding heavy dependencies to the core package (templates can require extras)

## Development notes

This repo is packaged as `embed_lab` and exposes the `emb` command via:

- `emb = "embed_lab.cli:app"`

The CLI is built with Typer, and templates are stored as strings in `embed_lab/templates.py`. The `emb init` command writes those templates into the target directory if files don’t already exist.

### Repository layout

- `src/embed_lab/cli.py`: Typer CLI entrypoint
- `src/embed_lab/templates.py`: template content written by `emb init`
- `pyproject.toml`: packaging + `emb` script definition


