"""Experiment 0: No-CoT vs CoT baseline.

For every (model, dataset, question) tuple, collect:
1. Direct answer (no CoT)
2. CoT answer (step-by-step reasoning)
"""

from __future__ import annotations

import json
import os

from tqdm import tqdm

from config import RESULTS_DIR
from models.ollama_client import query_model
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc


# --- Prompt Templates ---

def _no_cot_prompt_gsm8k(question: str) -> str:
    return (
        f"Answer this math question directly with no explanation. "
        f"Give only the final numeric answer.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def _cot_prompt_gsm8k(question: str) -> str:
    return (
        f"Think step by step, then provide your final answer after ####.\n\n"
        f"Question: {question}"
    )


def _no_cot_prompt_arc(question: str, choices_text: str) -> str:
    return (
        f"Answer this question directly with no explanation. "
        f"Give only the letter of the correct choice.\n\n"
        f"Question: {question}\n{choices_text}\n\nAnswer:"
    )


def _cot_prompt_arc(question: str, choices_text: str) -> str:
    return (
        f"Think step by step, then provide your final answer as a single letter.\n\n"
        f"Question: {question}\n{choices_text}"
    )


def run_baseline(model: str, dataset: str, samples: list[dict]) -> dict:
    """Run baseline experiment for one (model, dataset) pair.

    Returns dict with per-sample results and aggregate accuracy.
    """
    results = []
    n_no_cot_correct = 0
    n_cot_correct = 0

    desc = f"Baseline {model} / {dataset}"
    for sample in tqdm(samples, desc=desc):
        sid = sample["id"]
        question = sample["question"]
        gold = sample["gold_answer"]

        if dataset == "gsm8k":
            no_cot_prompt = _no_cot_prompt_gsm8k(question)
            cot_prompt = _cot_prompt_gsm8k(question)
        else:
            choices_text = sample["choices_text"]
            no_cot_prompt = _no_cot_prompt_arc(question, choices_text)
            cot_prompt = _cot_prompt_arc(question, choices_text)

        # Query model
        no_cot_response = query_model(model, no_cot_prompt)
        cot_response = query_model(model, cot_prompt)

        # Extract answers
        if dataset == "gsm8k":
            no_cot_answer = extract_answer_gsm8k(no_cot_response)
            cot_answer = extract_answer_gsm8k(cot_response)
        else:
            choice_labels = sample.get("choice_labels")
            no_cot_answer = extract_answer_arc(no_cot_response, choice_labels)
            cot_answer = extract_answer_arc(cot_response, choice_labels)

        # Check correctness
        if dataset == "gsm8k":
            no_cot_correct = _answers_equal_numeric(no_cot_answer, gold)
            cot_correct = _answers_equal_numeric(cot_answer, gold)
        else:
            no_cot_correct = no_cot_answer.upper() == gold.upper()
            cot_correct = cot_answer.upper() == gold.upper()

        n_no_cot_correct += int(no_cot_correct)
        n_cot_correct += int(cot_correct)

        results.append({
            "id": sid,
            "question": question,
            "gold_answer": gold,
            "no_cot_prompt": no_cot_prompt,
            "no_cot_response": no_cot_response,
            "no_cot_answer": no_cot_answer,
            "no_cot_correct": no_cot_correct,
            "cot_prompt": cot_prompt,
            "cot_response": cot_response,
            "cot_answer": cot_answer,
            "cot_correct": cot_correct,
        })

    n_total = len(samples)
    summary = {
        "model": model,
        "dataset": dataset,
        "n_total": n_total,
        "no_cot_accuracy": n_no_cot_correct / n_total if n_total else 0,
        "cot_accuracy": n_cot_correct / n_total if n_total else 0,
        "n_no_cot_correct": n_no_cot_correct,
        "n_cot_correct": n_cot_correct,
        "results": results,
    }

    # Save intermediate results
    _save_results(model, dataset, summary)
    return summary


def _answers_equal_numeric(a: str, b: str) -> bool:
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return a.strip() == b.strip()


def _save_results(model: str, dataset: str, summary: dict) -> None:
    model_safe = model.replace(":", "_").replace("/", "_")
    out_dir = os.path.join(RESULTS_DIR, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_safe}_{dataset}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved baseline results to {path}")
