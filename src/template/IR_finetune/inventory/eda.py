from __future__ import annotations

import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
from collections import Counter
from pathlib import Path
from typing import List

from inventory.datasets import (
    PairExample,
)


def _prepare_df(
    examples: List[PairExample],
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "query": ex.query,
                "passage": ex.passage,
                "label": ex.label,
            }
            for ex in examples
        ]
    )

    # Lengths
    df["query_char_len"] = df["query"].str.len()
    df["passage_char_len"] = df["passage"].str.len()
    df["query_word_len"] = df["query"].apply(
        lambda x: len(x.split())
    )
    df["passage_word_len"] = df["passage"].apply(
        lambda x: len(x.split())
    )

    # Simple lexical overlap (Jaccard on word sets) - cheap proxy for hardness
    def jaccard(row):
        q_words = set(row["query"].lower().split())
        p_words = set(row["passage"].lower().split())
        if not q_words and not p_words:
            return 1.0
        union = q_words | p_words
        if not union:
            return 0.0
        return len(q_words & p_words) / len(union)

    df["lexical_jaccard"] = df.apply(jaccard, axis=1)
    return df


def analyze_basic_statistics(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
) -> None:
    df = _prepare_df(examples)

    stats = {
        "Total examples": len(df),
        "Positive examples (label=1)": len(
            df[df["label"] == 1]
        ),
        "Negative examples (label=0)": len(
            df[df["label"] == 0]
        ),
        "Unique queries": df["query"].nunique(),
        "Unique passages": df["passage"].nunique(),
        "Exact duplicate (query, passage) pairs": df.duplicated(
            subset=["query", "passage"]
        ).sum(),
        "Avg passages per query": round(
            len(df) / df["query"].nunique(), 2
        ),
        "Avg query word length": round(
            df["query_word_len"].mean(), 2
        ),
        "Avg passage word length": round(
            df["passage_word_len"].mean(), 2
        ),
    }

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightgrey",
                ),
                cells=dict(
                    values=[
                        list(stats.keys()),
                        list(stats.values()),
                    ]
                ),
            )
        ]
    )
    fig.update_layout(title="Basic Dataset Statistics")
    fig.write_html(
        save_dir / f"{prefix}basic_statistics.html"
    )


def analyze_label_distribution(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
) -> None:
    df = _prepare_df(examples)
    label_counts = df["label"].value_counts().sort_index()

    fig = px.pie(
        values=label_counts.values,
        names=["Negative (0)", "Positive (1)"],
        title="Label Distribution (Positive vs Negative)",
    )
    fig.write_html(
        save_dir / f"{prefix}label_distribution.html"
    )


def analyze_length_distributions(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
) -> None:
    df = _prepare_df(examples)

    # Overlay word length histograms (query vs passage)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=df["query_word_len"],
            name="Queries",
            opacity=0.7,
        )
    )
    fig.add_trace(
        go.Histogram(
            x=df["passage_word_len"],
            name="Passages",
            opacity=0.7,
        )
    )
    fig.update_layout(
        barmode="overlay",
        title="Query vs Passage Word Length Distribution",
    )
    fig.write_html(
        save_dir / f"{prefix}length_overlay_histogram.html"
    )

    # Box plot by type
    melted = df.melt(
        value_vars=["query_word_len", "passage_word_len"],
        var_name="Type",
        value_name="Word Length",
    )
    fig = px.box(
        melted,
        x="Type",
        y="Word Length",
        title="Query vs Passage Word Length (Box Plot)",
    )
    fig.write_html(
        save_dir / f"{prefix}length_boxplot.html"
    )

    # Lengths stratified by label (passage length is especially interesting)
    fig = px.histogram(
        df,
        x="passage_word_len",
        color="label",
        barmode="overlay",
        nbins=50,
        title="Passage Word Length Distribution by Label",
        color_discrete_map={0: "red", 1: "green"},
    )
    fig.write_html(
        save_dir / f"{prefix}passage_length_by_label.html"
    )


def analyze_lexical_overlap(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
) -> None:
    df = _prepare_df(examples)

    fig = px.histogram(
        df,
        x="lexical_jaccard",
        color="label",
        barmode="overlay",
        nbins=50,
        title="Lexical Jaccard Overlap (Query ↔ Passage) by Label",
        labels={"lexical_jaccard": "Jaccard Similarity"},
        color_discrete_map={0: "red", 1: "green"},
    )
    fig.update_layout(xaxis_range=[0, 1])
    fig.write_html(
        save_dir / f"{prefix}lexical_jaccard_by_label.html"
    )


def analyze_top_terms(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
    top_n: int = 30,
) -> None:
    df = _prepare_df(examples)

    # Top terms in queries (overall)
    query_words = Counter(
        word
        for q in df["query"].str.lower().str.split()
        for word in q
    )
    top_query = query_words.most_common(top_n)
    fig = px.bar(
        x=[word for word, _ in top_query],
        y=[count for _, count in top_query],
        title=f"Top {top_n} Terms in Queries (All Examples)",
    )
    fig.write_html(
        save_dir / f"{prefix}top_query_terms.html"
    )

    # Top terms in passages, stratified by label
    for label, label_name in [
        (1, "positive"),
        (0, "negative"),
    ]:
        subset = df[df["label"] == label]
        passage_words = Counter(
            word
            for p in subset["passage"]
            .str.lower()
            .str.split()
            for word in p
        )
        top_passage = passage_words.most_common(top_n)
        fig = px.bar(
            x=[word for word, _ in top_passage],
            y=[count for _, count in top_passage],
            title=f"Top {top_n} Terms in Passages ({label_name} examples)",
        )
        fig.write_html(
            save_dir
            / f"{prefix}top_passage_terms_{label_name}.html"
        )


def analyze_query_patterns(
    examples: List[PairExample],
    save_dir: Path,
    prefix: str = "",
    top_n: int = 20,
) -> None:
    df = _prepare_df(examples)

    # Starting bigrams (very useful for spotting "how to", "install", etc.)
    df["start_bigram"] = (
        df["query"]
        .str.lower()
        .apply(
            lambda x: " ".join(x.split()[:2])
            if len(x.split()) >= 2
            else x.split()[0]
            if x.split()
            else ""
        )
    )
    top_bigrams = (
        df["start_bigram"].value_counts().head(top_n)
    )

    fig = px.bar(
        x=top_bigrams.index,
        y=top_bigrams.values,
        title=f"Top {top_n} Query Starting Bigrams",
    )
    fig.write_html(
        save_dir / f"{prefix}query_starting_bigrams.html"
    )

    # Passages per query distribution
    passages_per_query = df.groupby("query").size()
    count_dist = (
        passages_per_query.value_counts().sort_index()
    )
    fig = px.bar(
        x=count_dist.index.astype(str),
        y=count_dist.values,
        title="Distribution of Number of Passages per Query",
    )
    fig.write_html(
        save_dir
        / f"{prefix}passages_per_query_distribution.html"
    )


def perform_all_eda(
    examples: List[PairExample],
    split_name: str,
    base_save_dir: Path = Path("eda_plots"),
) -> None:
    save_dir = base_save_dir / split_name
    save_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{split_name}_" if split_name else ""

    analyze_basic_statistics(examples, save_dir, prefix)
    analyze_label_distribution(examples, save_dir, prefix)
    analyze_length_distributions(examples, save_dir, prefix)
    analyze_lexical_overlap(examples, save_dir, prefix)
    analyze_top_terms(examples, save_dir, prefix)
    analyze_query_patterns(examples, save_dir, prefix)


def check_cross_split_leakage(
    train: List[PairExample],
    dev: List[PairExample],
    gold: List[PairExample],
    save_dir: Path = Path("eda_plots/leakage"),
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = {"train": train, "dev": dev, "gold": gold}
    queries = {
        name: {ex.query for ex in exs}
        for name, exs in splits.items()
    }
    passages = {
        name: {ex.passage for ex in exs}
        for name, exs in splits.items()
    }
    pairs = {
        name: {(ex.query, ex.passage) for ex in exs}
        for name, exs in splits.items()
    }

    leakage_stats = {
        "Query overlap train∩dev": len(
            queries["train"] & queries["dev"]
        ),
        "Query overlap train∩gold": len(
            queries["train"] & queries["gold"]
        ),
        "Query overlap dev∩gold": len(
            queries["dev"] & queries["gold"]
        ),
        "Passage overlap train∩dev": len(
            passages["train"] & passages["dev"]
        ),
        "Passage overlap train∩gold": len(
            passages["train"] & passages["gold"]
        ),
        "Exact (query,passage) pair overlap train∩dev": len(
            pairs["train"] & pairs["dev"]
        ),
        "Exact (query,passage) pair overlap train∩gold": len(
            pairs["train"] & pairs["gold"]
        ),
    }

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Leakage Type", "Count"],
                    fill_color="lightgrey",
                ),
                cells=dict(
                    values=[
                        list(leakage_stats.keys()),
                        list(leakage_stats.values()),
                    ]
                ),
            )
        ]
    )
    fig.update_layout(title="Cross-Split Leakage Check")
    fig.write_html(save_dir / "cross_split_leakage.html")
