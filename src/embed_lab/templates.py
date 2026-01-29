from typing import Final


# inventory/datasets.py
TEMPLATE_DATASETS: Final[str] = """\
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class PairExample:
    sentence1: str
    sentence2: str
    label: int  # 1=similar, 0=dissimilar


def load() -> List[PairExample]:
    # Tiny toy dataset; small enough to run on CPU quickly.
    positives = [
        ("how to reset my password", "i forgot my password, how do i change it"),
        ("what is machine learning", "explain machine learning in simple terms"),
        ("best way to cook rice", "how do i cook rice properly"),
        ("python type annotations", "how to use type hints in python"),
    ]
    negatives = [
        ("how to reset my password", "best way to cook rice"),
        ("python type annotations", "i forgot my password, how do i change it"),
        ("what is machine learning", "how do i cook rice properly"),
        ("best way to cook rice", "explain machine learning in simple terms"),
    ]

    out: List[PairExample] = []
    out += [PairExample(a, b, 1) for a, b in positives]
    out += [PairExample(a, b, 0) for a, b in negatives]
    return out
"""


# inventory/preprocess.py
TEMPLATE_PREPROCESS: Final[str] = """\
from __future__ import annotations

from typing import List

from inventory.datasets import PairExample


def preprocess(examples: List[PairExample]) -> List[PairExample]:
    # No-op placeholder; add cleaning/filtering/prompting here if needed.
    return examples
"""


# inventory/train.py
TEMPLATE_TRAIN: Final[str] = """\
from __future__ import annotations

from pathlib import Path
from typing import List

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from inventory.datasets import PairExample


def train(
    examples: List[PairExample],
    output_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = Dataset.from_dict(
        {
            "sentence1": [e.sentence1 for e in examples],
            "sentence2": [e.sentence2 for e in examples],
            "label": [e.label for e in examples],
        }
    )

    model = SentenceTransformer(model_name)

    # ContrastiveLoss expects (sentence1, sentence2) + label in {0,1}.
    loss = losses.ContrastiveLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()

    model.save_pretrained(str(output_dir / "final"))
"""


# experiments/exp_01_baseline.py
TEMPLATE_EXP_01_BASELINE: Final[str] = """\
from __future__ import annotations

from pathlib import Path

from inventory.datasets import load
from inventory.preprocess import preprocess
from inventory.train import train


def main() -> None:
    exp_name = "exp_01_baseline"
    results_dir = Path("results") / exp_name

    examples = preprocess(load())
    train(
        examples=examples,
        output_dir=results_dir,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Try later:
        # model_name="BAAI/bge-m3",
    )


if __name__ == "__main__":
    main()
"""


# main.py
TEMPLATE_MAIN: Final[str] = """\
def main() -> None:
    print("Run an experiment with: emb train experiments/exp_01_baseline.py")


if __name__ == "__main__":
    main()
"""


TEMPLATE_GITIGNORE: Final[str] = """\
__pycache__/
*.py[cod]
.venv/
results/
.env
"""
