from __future__ import annotations

from pathlib import Path

from inventory.datasets import load_splits
from inventory.preprocess import preprocess
from inventory.train import train
from inventory.evaluate import evaluate_on_gold
from inventory.plotting import plot_gold_metrics, plot_training_curves


def main() -> None:
    exp_name = "exp_01_baseline"
    run_dir = Path("results") / exp_name
    data_dir = Path("data")

    train_raw, dev_raw, gold_raw = load_splits(data_dir)

    train_data = preprocess(train_raw)
    dev_data = preprocess(dev_raw)
    gold_data = preprocess(gold_raw)

    final_dir = train(
        train_examples=train_data,
        dev_examples=dev_data,
        run_dir=run_dir,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Later:
        # model_name="BAAI/bge-m3",
    )

    evaluate_on_gold(
        model_dir=final_dir,
        gold_examples=gold_data,
        output_dir=run_dir / "eval",
    )

    plot_training_curves(run_dir)
    plot_gold_metrics(run_dir)


if __name__ == "__main__":
    main()
