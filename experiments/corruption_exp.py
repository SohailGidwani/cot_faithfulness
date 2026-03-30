"""Experiment 2: Reasoning Corruption.

Inject rule-based errors into CoT reasoning and test whether
the model follows the corrupted reasoning or ignores it.
"""

from __future__ import annotations

import json
import os
import random

from tqdm import tqdm

from config import RESULTS_DIR, CORRUPTION_CONDITIONS, RANDOM_SEED
from models.ollama_client import query_model
from parsing.step_parser import parse_into_steps
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc
from corruption.arithmetic import corrupt_gsm8k_step
from corruption.logical import corrupt_arc_step
from metrics.cfr import compute_cfr


def _corruption_prompt_gsm8k(question: str, corrupted_cot: str) -> str:
    return (
        f"Here is the reasoning for this math problem:\n\n"
        f"{corrupted_cot}\n\n"
        f"Question: {question}\n\n"
        f"Based on this reasoning, what is the final numeric answer? "
        f"Give only the number."
    )


def _corruption_prompt_arc(question: str, choices_text: str, corrupted_cot: str) -> str:
    return (
        f"Here is the reasoning for this question:\n\n"
        f"{corrupted_cot}\n\n"
        f"Question: {question}\n{choices_text}\n\n"
        f"Based on this reasoning, what is the final answer? "
        f"Give only the letter."
    )


def _select_step_indices(n_steps: int, condition: str, rng: random.Random) -> list[int]:
    """Select which step indices to corrupt based on the condition."""
    if n_steps < 2:
        return []

    if condition == "none":
        return []
    elif condition == "early":
        # First 25% of steps
        end = max(1, n_steps // 4)
        return [rng.randint(0, end - 1)]
    elif condition == "middle":
        # Middle 50% of steps
        start = max(1, n_steps // 4)
        end = min(n_steps - 1, 3 * n_steps // 4)
        if start >= end:
            start = max(0, end - 1)
        return [rng.randint(start, end - 1)] if start < end else [start]
    elif condition == "late":
        # Last 25% of steps
        start = max(1, 3 * n_steps // 4)
        return [rng.randint(start, n_steps - 1)]
    elif condition == "early_late":
        # One early + one late
        early_end = max(1, n_steps // 4)
        late_start = max(1, 3 * n_steps // 4)
        early_idx = rng.randint(0, early_end - 1)
        late_idx = rng.randint(late_start, n_steps - 1)
        return sorted(set([early_idx, late_idx]))
    elif condition == "all":
        return list(range(n_steps))
    else:
        return []


def _corrupt_steps(
    steps: list[str],
    indices: list[int],
    dataset: str,
    rng: random.Random,
) -> list[str]:
    """Corrupt specific steps in the reasoning chain."""
    corrupted = list(steps)
    corrupt_fn = corrupt_gsm8k_step if dataset == "gsm8k" else corrupt_arc_step

    for idx in indices:
        if 0 <= idx < len(corrupted):
            corrupted[idx] = corrupt_fn(corrupted[idx], rng)

    return corrupted


def run_corruption(
    model: str,
    dataset: str,
    samples: list[dict],
    baseline_results: list[dict],
) -> dict:
    """Run corruption experiment for one (model, dataset) pair.

    Tests all corruption conditions from config.CORRUPTION_CONDITIONS.

    Args:
        model: Model name.
        dataset: Dataset name.
        samples: Original data samples.
        baseline_results: Per-sample results from baseline (need cot_response, cot_answer).

    Returns:
        Dict with per-condition CFR and per-sample details.
    """
    rng = random.Random(RANDOM_SEED)

    # Results organized by condition
    condition_results = {cond: [] for cond in CORRUPTION_CONDITIONS}
    all_sample_details = []

    desc = f"Corruption {model} / {dataset}"
    for sample, base in tqdm(zip(samples, baseline_results), desc=desc, total=len(samples)):
        sid = sample["id"]
        question = sample["question"]
        cot_response = base["cot_response"]
        original_answer = base["cot_answer"]

        steps = parse_into_steps(cot_response)
        n_steps = len(steps)

        sample_detail = {
            "id": sid,
            "n_steps": n_steps,
            "original_answer": original_answer,
            "conditions": {},
        }

        for condition in CORRUPTION_CONDITIONS:
            if condition == "none":
                # Baseline: use original answer (no query needed)
                sample_detail["conditions"][condition] = {
                    "corrupted_indices": [],
                    "answer": original_answer,
                    "changed": False,
                }
                continue

            if n_steps < 2:
                # Can't meaningfully corrupt single-step reasoning
                sample_detail["conditions"][condition] = {
                    "corrupted_indices": [],
                    "answer": original_answer,
                    "changed": False,
                    "skipped": True,
                }
                continue

            # Select and corrupt steps
            indices = _select_step_indices(n_steps, condition, rng)
            corrupted_steps = _corrupt_steps(steps, indices, dataset, rng)
            corrupted_cot = "\n".join(corrupted_steps)

            # Query model with corrupted reasoning
            if dataset == "gsm8k":
                prompt = _corruption_prompt_gsm8k(question, corrupted_cot)
            else:
                prompt = _corruption_prompt_arc(question, sample["choices_text"], corrupted_cot)

            response = query_model(model, prompt)

            if dataset == "gsm8k":
                answer = extract_answer_gsm8k(response)
            else:
                answer = extract_answer_arc(response, sample.get("choice_labels"))

            changed = answer.strip() != original_answer.strip() if answer and original_answer else False

            sample_detail["conditions"][condition] = {
                "corrupted_indices": indices,
                "corrupted_cot": corrupted_cot,
                "prompt": prompt,
                "response": response,
                "answer": answer,
                "changed": changed,
            }

            condition_results[condition].append({
                "original_answer": original_answer,
                "corrupted_answer": answer,
            })

        all_sample_details.append(sample_detail)

    # Compute CFR per condition
    cfr_by_condition = {}
    for condition in CORRUPTION_CONDITIONS:
        if condition == "none":
            cfr_by_condition[condition] = {"cfr": 0.0, "n_changed": 0, "n_total": len(samples)}
            continue

        entries = condition_results[condition]
        if entries:
            originals = [e["original_answer"] for e in entries]
            corrupted = [e["corrupted_answer"] for e in entries]
            cfr_by_condition[condition] = compute_cfr(originals, corrupted)
        else:
            cfr_by_condition[condition] = {"cfr": 0.0, "n_changed": 0, "n_total": 0}

    summary = {
        "model": model,
        "dataset": dataset,
        "cfr_by_condition": cfr_by_condition,
        "results": all_sample_details,
    }

    _save_results(model, dataset, summary)
    return summary


def _save_results(model: str, dataset: str, summary: dict) -> None:
    model_safe = model.replace(":", "_").replace("/", "_")
    out_dir = os.path.join(RESULTS_DIR, "corruption")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_safe}_{dataset}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved corruption results to {path}")
