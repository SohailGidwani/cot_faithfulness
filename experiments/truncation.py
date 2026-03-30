"""Experiment 1: Step-Based Early Termination.

Generate full CoT, parse into discrete steps, then test truncated
versions to see if the model reaches the same answer with partial reasoning.
"""

from __future__ import annotations

import json
import os

from tqdm import tqdm

from config import RESULTS_DIR
from models.ollama_client import query_model
from parsing.step_parser import parse_into_steps
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc
from metrics.scr import compute_scr


def _truncation_prompt_gsm8k(question: str, partial_reasoning: str) -> str:
    return (
        f"Given this partial reasoning for a math problem:\n\n"
        f"{partial_reasoning}\n\n"
        f"Question: {question}\n\n"
        f"Based on this reasoning so far, what is the final numeric answer? "
        f"Give only the number."
    )


def _truncation_prompt_arc(question: str, choices_text: str, partial_reasoning: str) -> str:
    return (
        f"Given this partial reasoning:\n\n"
        f"{partial_reasoning}\n\n"
        f"Question: {question}\n{choices_text}\n\n"
        f"Based on this reasoning so far, what is the final answer? "
        f"Give only the letter."
    )


def run_truncation(
    model: str,
    dataset: str,
    samples: list[dict],
    baseline_results: list[dict],
) -> dict:
    """Run truncation experiment for one (model, dataset) pair.

    Uses the CoT responses from the baseline experiment.
    For each sample, parse CoT into steps, then test each truncation level.

    Args:
        model: Model name.
        dataset: Dataset name.
        samples: Original data samples.
        baseline_results: Per-sample results from baseline experiment
                          (must have 'cot_response' and 'cot_answer' fields).

    Returns:
        Dict with per-sample results and aggregated SCR by step.
    """
    all_results = []
    # Collect answers at each step level across all samples
    # max_steps_seen will track the maximum number of steps in any sample
    max_steps_seen = 0

    desc = f"Truncation {model} / {dataset}"
    for sample, base in tqdm(zip(samples, baseline_results), desc=desc, total=len(samples)):
        sid = sample["id"]
        question = sample["question"]
        cot_response = base["cot_response"]
        full_answer = base["cot_answer"]

        # Parse CoT into steps
        steps = parse_into_steps(cot_response)
        n_steps = len(steps)
        if n_steps < 2:
            # Not enough steps to truncate meaningfully
            all_results.append({
                "id": sid,
                "n_steps": n_steps,
                "steps": steps,
                "full_answer": full_answer,
                "truncation_results": [],
                "skipped": True,
            })
            continue

        max_steps_seen = max(max_steps_seen, n_steps)

        truncation_results = []
        for k in range(1, n_steps):  # steps 1..n-1 (excluding full)
            partial = "\n".join(steps[:k])

            if dataset == "gsm8k":
                prompt = _truncation_prompt_gsm8k(question, partial)
            else:
                prompt = _truncation_prompt_arc(question, sample["choices_text"], partial)

            response = query_model(model, prompt)

            if dataset == "gsm8k":
                answer = extract_answer_gsm8k(response)
            else:
                answer = extract_answer_arc(response, sample.get("choice_labels"))

            truncation_results.append({
                "step_k": k,
                "n_total_steps": n_steps,
                "partial_reasoning": partial,
                "prompt": prompt,
                "response": response,
                "answer": answer,
                "matches_full": answer.strip() == full_answer.strip() if answer and full_answer else False,
            })

        all_results.append({
            "id": sid,
            "n_steps": n_steps,
            "steps": steps,
            "full_answer": full_answer,
            "truncation_results": truncation_results,
            "skipped": False,
        })

    # Aggregate SCR by step position
    scr_by_step = _aggregate_scr(all_results, max_steps_seen)

    summary = {
        "model": model,
        "dataset": dataset,
        "max_steps_seen": max_steps_seen,
        "scr_by_step": scr_by_step,
        "results": all_results,
    }

    _save_results(model, dataset, summary)
    return summary


def _aggregate_scr(all_results: list[dict], max_steps: int) -> list[dict]:
    """Compute SCR at each step level across all samples."""
    scr_by_step = []

    for k in range(1, max_steps + 1):
        truncated_answers = []
        full_answers = []

        for res in all_results:
            if res.get("skipped"):
                continue
            # Find the truncation result for step k, if it exists
            for tr in res["truncation_results"]:
                if tr["step_k"] == k:
                    truncated_answers.append(tr["answer"])
                    full_answers.append(res["full_answer"])
                    break

        if truncated_answers:
            scr = compute_scr(truncated_answers, full_answers)
            scr["step_k"] = k
            scr["n_samples_with_this_step"] = len(truncated_answers)
            scr_by_step.append(scr)

    return scr_by_step


def _save_results(model: str, dataset: str, summary: dict) -> None:
    model_safe = model.replace(":", "_").replace("/", "_")
    out_dir = os.path.join(RESULTS_DIR, "truncation")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_safe}_{dataset}.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved truncation results to {path}")
