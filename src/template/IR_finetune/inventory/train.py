from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from datasets import Dataset  # type: ignore
from sentence_transformers import (  # type: ignore
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from .datasets import PairExample


def _to_hf_dataset(examples: List[PairExample]) -> Dataset:
    return Dataset.from_dict(
        {
            "query": [e.query for e in examples],
            "passage": [e.passage for e in examples],
            "label": [e.label for e in examples],
        }
    )


def train(
    train_examples: List[PairExample],
    dev_examples: List[PairExample],
    run_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _to_hf_dataset(train_examples)
    eval_dataset = _to_hf_dataset(dev_examples)

    model = SentenceTransformer(model_name)
    loss = losses.ContrastiveLoss(model)

    ckpt_dir = run_dir / "checkpoints"

    args = SentenceTransformerTrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=5,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    final_dir = run_dir / "final"
    model.save_pretrained(str(final_dir))

    best_ckpt: Optional[str] = getattr(
        trainer.state, "best_model_checkpoint", None
    )
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "final_dir": str(final_dir),
                "best_checkpoint": best_ckpt,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return final_dir
