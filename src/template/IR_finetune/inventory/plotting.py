from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.express as px  # type: ignore


def plot_training_curves(run_dir: Path) -> None:
    ckpt_dir = run_dir / "checkpoints"
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return

    state: Dict[str, Any] = json.loads(
        state_path.read_text(encoding="utf-8")
    )
    log_history: List[Dict[str, Any]] = state.get(
        "log_history", []
    )

    steps_loss: List[int] = []
    loss_vals: List[float] = []
    steps_lr: List[int] = []
    lr_vals: List[float] = []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        step_i = int(step)

        if "loss" in row:
            steps_loss.append(step_i)
            loss_vals.append(float(row["loss"]))

        if "learning_rate" in row:
            steps_lr.append(step_i)
            lr_vals.append(float(row["learning_rate"]))

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if steps_loss:
        fig = px.line(
            x=steps_loss,
            y=loss_vals,
            labels={"x": "step", "y": "loss"},
            title="Training loss",
        )
        fig.write_html(out_dir / "loss.html")

    if steps_lr:
        fig = px.line(
            x=steps_lr,
            y=lr_vals,
            labels={"x": "step", "y": "learning_rate"},
            title="Learning rate",
        )
        fig.write_html(out_dir / "lr.html")


def plot_gold_metrics(
    run_dir: Path,
    keys: Tuple[str, ...] = (
        "gold_cosine_accuracy",
        "gold_cosine_f1",
    ),
) -> None:
    metrics_path = run_dir / "eval" / "gold_metrics.json"
    if not metrics_path.exists():
        return

    metrics: Dict[str, Any] = json.loads(
        metrics_path.read_text(encoding="utf-8")
    )

    xs: List[str] = []
    ys: List[float] = []
    for k in keys:
        if k in metrics:
            xs.append(k)
            ys.append(float(metrics[k]))

    if not xs:
        return

    out_dir = run_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = px.bar(
        x=xs,
        y=ys,
        labels={"x": "metric", "y": "value"},
        title="Gold metrics",
    )
    fig.write_html(out_dir / "gold_metrics.html")
