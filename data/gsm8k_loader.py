"""Load and sample GSM8K test split."""

from __future__ import annotations

import re
from datasets import load_dataset

from config import SAMPLES_PER_DATASET, RANDOM_SEED


def _extract_gold_answer(answer_text: str) -> str:
    """Extract the numeric answer after #### from GSM8K answer field."""
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    # Fallback: last number in the text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", answer_text)
    return numbers[-1].replace(",", "") if numbers else ""


def load_gsm8k_samples(n_samples: int = SAMPLES_PER_DATASET, seed: int = RANDOM_SEED) -> list[dict]:
    """Load n_samples from GSM8K test split.

    Returns list of dicts with keys: id, question, gold_answer, gold_solution
    """
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = dataset.shuffle(seed=seed).select(range(min(n_samples, len(dataset))))

    samples = []
    for i, row in enumerate(dataset):
        samples.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "gold_answer": _extract_gold_answer(row["answer"]),
            "gold_solution": row["answer"],
        })
    return samples
