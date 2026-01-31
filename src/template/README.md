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
