from __future__ import annotations
from datasets import load_dataset
from config import NUM_SAMPLES, SEED


def _fmt_choices(choices):
    labels = choices["label"]
    texts = choices["text"]
    return "\n".join("(%s) %s" % (l, t) for l, t in zip(labels, texts)), labels


def load_arc_samples(n=NUM_SAMPLES, seed=SEED):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    out = []
    for i, row in enumerate(ds):
        ctxt, clabels = _fmt_choices(row["choices"])
        out.append({
            "id": "arc_%d" % i,
            "question": row["question"],
            "choices_text": ctxt,
            "choice_labels": clabels,
            "gold_answer": row["answerKey"],
        })
    return out
