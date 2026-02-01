from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer  # type: ignore
from sentence_transformers.evaluation import (  # type: ignore
    BinaryClassificationEvaluator,
)

from inventory.datasets import PairExample  # type: ignore


def evaluate_on_gold(
    model_dir: Path,
    gold_examples: List[PairExample],
    output_dir: Path,
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(str(model_dir))

    evaluator = BinaryClassificationEvaluator(
        sentences1=[e.query for e in gold_examples],
        sentences2=[e.passage for e in gold_examples],
        labels=[e.label for e in gold_examples],
        name="gold",
        write_csv=True,
        show_progress_bar=False,
    )

    metrics = evaluator(model, output_path=str(output_dir))

    (output_dir / "gold_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics
