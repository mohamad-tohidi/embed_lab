from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True, slots=True)
class PairExample:
    query: str
    passage: str
    label: int  # 1=similar, 0=dissimilar


def _read_jsonl(path: Path) -> List[PairExample]:
    out: List[PairExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out.append(
                PairExample(
                    query=str(obj["query"]),
                    passage=str(obj["passage"]),
                    label=int(obj["label"]),
                )
            )
    return out


def load_splits(data_dir: Path) -> Tuple[List[PairExample], List[PairExample], List[PairExample]]:
    train = _read_jsonl(data_dir / "train.jsonl")
    dev = _read_jsonl(data_dir / "dev.jsonl")
    gold = _read_jsonl(data_dir / "gold.jsonl")
    return train, dev, gold
