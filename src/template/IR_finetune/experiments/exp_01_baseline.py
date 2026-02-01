from __future__ import annotations
from pathlib import Path

# Import the implementations
from inventory.datasets import load_splits
from inventory.preprocess import preprocess
from inventory.train import train
from inventory.evaluate import evaluate_on_gold
from inventory.plotting import (
    plot_gold_metrics,
    plot_training_curves,
)

# Import the SPECS (The new part)
from inventory.specs import (
    TrainFn,
    EvaluateFn,
    PreprocessFn,
)


def main() -> None:
    exp_name = "exp_01_baseline"
    run_dir = Path("results") / exp_name
    data_dir = Path("data")

    # 1. Enforce the Conventions
    # By assigning the imported functions to typed variables,
    # we guarantee they match the spec.
    train_step: TrainFn = train
    eval_step: EvaluateFn = evaluate_on_gold
    prep_step: PreprocessFn = preprocess

    # 2. Run the pipeline
    train_raw, dev_raw, gold_raw = load_splits(data_dir)

    train_data = prep_step(train_raw)
    dev_data = prep_step(dev_raw)
    gold_data = prep_step(gold_raw)

    final_dir = train_step(
        train_examples=train_data,
        dev_examples=dev_data,
        run_dir=run_dir,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    eval_step(
        model_dir=final_dir,
        gold_examples=gold_data,
        output_dir=run_dir / "eval",
    )

    plot_training_curves(run_dir)
    plot_gold_metrics(run_dir)


if __name__ == "__main__":
    main()
