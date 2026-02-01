"""
inventory/specs.py
Formal definitions of the pipeline steps using Python Protocols.
"""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Protocol,
    runtime_checkable,
)

from inventory.datasets import PairExample


@runtime_checkable
class PreprocessFn(Protocol):
    """
    Convention: Takes a list of examples, returns a modified list of examples.
    """

    def __call__(
        self, examples: List[PairExample]
    ) -> List[PairExample]: ...


@runtime_checkable
class TrainFn(Protocol):
    """
    Convention: Takes train/dev data + run_dir, returns the Path to the saved model.
    Keyword arguments (model_name, batch_size) are allowed but optional in the contract.
    """

    def __call__(
        self,
        train_examples: List[PairExample],
        dev_examples: List[PairExample],
        run_dir: Path,
        **kwargs: Any,
    ) -> Path: ...


@runtime_checkable
class EvaluateFn(Protocol):
    """
    Convention: Takes model_dir + gold data + output_dir, returns metrics dict.
    """

    def __call__(
        self,
        model_dir: Path,
        gold_examples: List[PairExample],
        output_dir: Path,
    ) -> Dict[str, float]: ...
