from __future__ import annotations

import json
from pydantic import BaseModel
from pathlib import Path
from typing import List, Tuple


class PairExample(BaseModel):
    query: str
    passage: str


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
                )
            )
    return out


def load_splits(
    data_dir: Path,
) -> Tuple[
    List[PairExample], List[PairExample], List[PairExample]
]:
    train = _read_jsonl(data_dir / "train.jsonl")
    dev = _read_jsonl(data_dir / "dev.jsonl")
    gold = _read_jsonl(data_dir / "gold.jsonl")
    return train, dev, gold
