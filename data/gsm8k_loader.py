from __future__ import annotations
import re
from datasets import load_dataset
from config import NUM_SAMPLES, SEED


def _pull_answer(ans_text):
    # the gold answer in gsm8k sits after ####
    m = re.search(r"####\s*([\d,]+(?:\.\d+)?)", ans_text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"[\d,]+(?:\.\d+)?", ans_text)
    return nums[-1].replace(",", "") if nums else ""


def load_gsm8k_samples(n=NUM_SAMPLES, seed=SEED):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out = []
    for i, row in enumerate(ds):
        out.append({
            "id": "gsm8k_%d" % i,
            "question": row["question"],
            "gold_answer": _pull_answer(row["answer"]),
            "gold_solution": row["answer"],
        })
    return out
