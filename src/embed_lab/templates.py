from typing import Final

# -------------------------
# data/*.jsonl
# -------------------------

TEMPLATE_DATA_TRAIN_JSONL: Final[str] = """\
{"query":"how to reset my password","passage":"i forgot my password, how do i change it","label":1}
{"query":"what is machine learning","passage":"explain machine learning in simple terms","label":1}
{"query":"best way to cook rice","passage":"how do i cook rice properly","label":1}
{"query":"python type annotations","passage":"how to use type hints in python","label":1}
{"query":"how to reset my password","passage":"best way to cook rice","label":0}
{"query":"python type annotations","passage":"i forgot my password, how do i change it","label":0}
{"query":"what is machine learning","passage":"how do i cook rice properly","label":0}
{"query":"best way to cook rice","passage":"explain machine learning in simple terms","label":0}
"""

TEMPLATE_DATA_DEV_JSONL: Final[str] = """\
{"query":"how to center a div","passage":"css center div horizontally and vertically","label":1}
{"query":"how to center a div","passage":"what is machine learning","label":0}
"""

TEMPLATE_DATA_GOLD_JSONL: Final[str] = """\
{"query":"install python on ubuntu","passage":"how do i install python in ubuntu","label":1}
{"query":"git undo last commit","passage":"how to revert the latest git commit","label":1}
{"query":"install python on ubuntu","passage":"best way to cook rice","label":0}
{"query":"git undo last commit","passage":"how to reset my password","label":0}
"""

# -------------------------
# inventory/datasets.py
# -------------------------

TEMPLATE_DATASETS: Final[str] = """\
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
"""

# -------------------------
# inventory/preprocess.py
# -------------------------

TEMPLATE_PREPROCESS: Final[str] = """\
from __future__ import annotations

from typing import List

from inventory.datasets import PairExample


def preprocess(examples: List[PairExample]) -> List[PairExample]:
    # No-op placeholder; add cleaning/filtering/prompting here if needed.
    return examples
"""

# -------------------------
# inventory/train.py
# -------------------------

TEMPLATE_TRAIN: Final[str] = """\
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from inventory.datasets import PairExample


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

    best_ckpt: Optional[str] = getattr(trainer.state, "best_model_checkpoint", None)
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
"""

# -------------------------
# inventory/evaluate.py
# -------------------------

TEMPLATE_EVALUATE: Final[str] = """\
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator

from inventory.datasets import PairExample


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
"""

# -------------------------
# inventory/plotting.py (Plotly -> HTML)
# -------------------------

TEMPLATE_PLOTTING: Final[str] = """\
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import plotly.express as px


def plot_training_curves(run_dir: Path) -> None:
    ckpt_dir = run_dir / "checkpoints"
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return

    state: Dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
    log_history: List[Dict[str, Any]] = state.get("log_history", [])

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
    keys: Tuple[str, ...] = ("gold_cosine_accuracy", "gold_cosine_f1"),
) -> None:
    metrics_path = run_dir / "eval" / "gold_metrics.json"
    if not metrics_path.exists():
        return

    metrics: Dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))

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
"""

# -------------------------
# experiments/exp_01_baseline.py
# -------------------------

TEMPLATE_EXP_01_BASELINE: Final[str] = """\
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
"""

# -------------------------
# main.py
# -------------------------

TEMPLATE_MAIN: Final[str] = """\
def main() -> None:
    print(
        "Run the baseline experiment with: uv run experiments/exp_01_baseline.py"
    )


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


TEMPLATE_README_MD: Final[str] = """\
# Embed Lab

Embed Lab is a small template for fine-tuning text embedding models with **convention over configuration**.

You define reusable code once in `inventory/`, and you define experiments as runnable Python files in `experiments/`.
Each experiment produces its own artifacts under `results/<experiment_name>/`.

This template ships with a working, end-to-end example based on Sentence-Transformers (SBERT) so you can run a complete pipeline quickly.

---

## Quickstart

i assume you are using `uv` as your environment manager. If not, adapt accordingly 
(and ask your self ,why ?).



1) Create a new lab project:

    when you are reading this, it means you have already installed `embed-lab` package in your environment.

    by running `uv add embed-lab` 

    and ran `emb init .` to create a new lab project.


2) Install runtime deps for the example pipeline:

    ```bash
    uv add sentence-transformers datasets plotly
    ```

3) Run the baseline experiment:

    ```bash
    uv run experiments/exp_01_baseline.py
    ```

Artifacts will be written to:

- `results/exp_01_baseline/final/` (the saved model)
- `results/exp_01_baseline/eval/` (gold evaluation metrics + CSV)
- `results/exp_01_baseline/plots/` (interactive Plotly HTML charts)

---

## Project layout

```text
my_lab/
├── data/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── gold.jsonl
│
├── inventory/
│   ├── datasets.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── plotting.py
│
├── experiments/
│   └── exp_01_baseline.py
│
└── results/               # generated artifacts (git-ignored)
```

---

## The pipeline

Every experiment follows the same “3 steps”:

1) **Load**: read JSONL files from `data/`
2) **Preprocess**: optional transformation (can be a no-op)
3) **Train**: fine-tune a model and save it

Then (optional but recommended):

4) **Evaluate** on a gold dataset  
5) **Plot** metrics + training curves

This template keeps these steps in separate files so you can swap parts without rewriting everything.

---

## File-by-file explanation

### [`data/*.jsonl`](./data/validation.jsonl)
Your datasets live here.

Each line is a JSON object with:

```json
{"query": "...", "passage": "...", "label": 1}
```

- `label=1` means the pair is similar / positive
- `label=0` means dissimilar / negative

<details>
<summary><strong>Key notes for creating a good dataset</strong></summary>

- Make sure **no data is leaked** from the train set into the validation or gold data. (you can run a dedup in the preprocess.)
- We call the test set **`gold`** here because it must be **high-quality and pure**.
- If you can **label the gold set manually your self**, do it!

</details>


### [`inventory/datasets.py`](./inventory/datasets.py)
- Loads `train.jsonl`, `dev.jsonl`, `gold.jsonl`
- Returns lists of examples for each split


### [`inventory/preprocess.py`](./inventory/preprocess.py)
A hook for anything you want:
- cleaning text
- filtering low-quality samples
- adding prompts / instructions
- balancing labels
- **deduplication** (i highly recommend this between the train and gold anb val sets!)

It is intentionally a no-op by default.

### [`inventory/train.py`](./inventory/train.py)
Trains a Sentence-Transformers embedding model using a pairwise training loss and saves the model to `results/<exp>/final/`. 

This file is “just an example backend”.
You can rewrite it using pure Transformers, PyTorch Lightning, JAX, etc.

<details>
<summary><strong>Key notes for writing a good training script</strong></summary>

if you are fine tuning a pre trained IR model, like `bge-m3` or the `e5` model or anything else.

first read their technichal report and see what loss functions and method thye used.

it is recommended to stick to their loss function and data structure and argumants (not all args though!) for a good fine tune!

then you need to define a loss function here.

the loss function is VERY important here and it MUST align with your data (or your data must align with your loss) and also for the task that you want to fine tune the model on.

i recommend you to check the http://sbert.net website for detailed informatioin


</details>


### [`inventory/evaluate.py`](./inventory/evaluate.py)
Evaluates the trained model on the `gold` split and saves:
- `results/<exp>/eval/gold_metrics.json`
- a CSV log file produced by the evaluator (useful for tracking runs) [attached_file:1]

### [`inventory/plotting.py`](./inventory/plotting.py)
Creates interactive Plotly charts and writes them as HTML files in:

- `results/<exp>/plots/loss.html`
- `results/<exp>/plots/lr.html`
- `results/<exp>/plots/gold_metrics.html`

Plotly HTML export keeps charts interactive and shareable (open them in any browser). 

### [`experiments/exp_01_baseline.py`](./experiments/exp_01_baseline.py)
A runnable “recipe” that wires everything:

- load splits from `data/`
- preprocess them
- train
- evaluate on gold
- generate plots

Make new experiments by copying this file and changing parameters.

---

## Notes for fine-tuning BGE-M3 (important)

You *can* set:

```python
model_name="BAAI/bge-m3"
```

But: fine-tuning is not “model-name only”.

When you fine-tune a specific model family (like BGE), you should align with:
- The **loss function** and training objective
- How batches are constructed (especially if using in-batch negatives)
- Any normalization / pooling conventions
- Any hard-negative mining strategy

For example, MultipleNegativesRankingLoss is commonly used for retrieval training and relies heavily on in-batch negatives, so batch composition and batch size become part of the method.  

This template defaults to a simple pairwise setup so it runs fast out-of-the-box. Treat it as a starting point, not the definitive “best” recipe for every model.

---

## Tips for experiments

- Keep experiments small and reproducible.
- Save run metadata (model name, epochs, batch size, loss type) into JSON next to results.
- Avoid editing old experiments; create new ones (`exp_02_*`) so your `experiments/` folder becomes your research log.

---


## Enjoy!

if you like the `emb` tool,then feel free to contribute or star our project to support us at https://github.com/mohamad-tohidi/embed_lab
"""
