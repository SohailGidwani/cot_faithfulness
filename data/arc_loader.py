"""Load and sample ARC-Challenge test split."""

from __future__ import annotations

from datasets import load_dataset

from config import SAMPLES_PER_DATASET, RANDOM_SEED


def _format_choices(choices: dict) -> tuple[str, list[str]]:
    """Format ARC choices into a readable string and list of labels.

    ARC choices come as {"text": [...], "label": [...]}.
    Returns (formatted_string, list_of_labels).
    """
    labels = choices["label"]
    texts = choices["text"]
    lines = []
    for label, text in zip(labels, texts):
        lines.append(f"({label}) {text}")
    return "\n".join(lines), labels


def load_arc_samples(n_samples: int = SAMPLES_PER_DATASET, seed: int = RANDOM_SEED) -> list[dict]:
    """Load n_samples from ARC-Challenge test split.

    Returns list of dicts with keys: id, question, choices_text, choice_labels, gold_answer
    """
    dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    samples = []
    for i, row in enumerate(dataset):
        choices_text, choice_labels = _format_choices(row["choices"])
        samples.append({
            "id": f"arc_{i}",
            "question": row["question"],
            "choices_text": choices_text,
            "choice_labels": choice_labels,
            "gold_answer": row["answerKey"],
        })
    return samples
