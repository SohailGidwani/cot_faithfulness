"""Aggregate results from all experiments into a unified summary."""

from __future__ import annotations

import json
import os
import numpy as np
from datetime import datetime

from config import RESULTS_DIR, MODELS, DATASETS, SAMPLES_PER_DATASET, CORRUPTION_CONDITIONS, HINT_TEMPLATES
from metrics.statistical_tests import mcnemar_test


def load_json(path: str) -> dict | None:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _model_safe(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def aggregate_all(models: list[str] = MODELS, datasets: list[str] = DATASETS) -> dict:
    """Aggregate all experiment results into a unified JSON structure."""

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "datasets": datasets,
            "samples_per_dataset": SAMPLES_PER_DATASET,
        },
        "baseline": {},
        "truncation": {},
        "corruption": {},
        "biased_hints": {},
        "cross_model_comparison": {},
    }

    for model in models:
        ms = _model_safe(model)
        output["baseline"][model] = {}
        output["truncation"][model] = {}
        output["corruption"][model] = {}
        output["biased_hints"][model] = {}

        for dataset in datasets:
            # --- Baseline ---
            base = load_json(os.path.join(RESULTS_DIR, "baseline", f"{ms}_{dataset}.json"))
            if base:
                output["baseline"][model][dataset] = {
                    "no_cot_accuracy": base["no_cot_accuracy"],
                    "cot_accuracy": base["cot_accuracy"],
                    "n_total": base["n_total"],
                }

            # --- Truncation ---
            trunc = load_json(os.path.join(RESULTS_DIR, "truncation", f"{ms}_{dataset}.json"))
            if trunc:
                scr_summary = []
                for entry in trunc.get("scr_by_step", []):
                    scr_summary.append({
                        "step_k": entry["step_k"],
                        "scr": entry["scr"],
                        "ci_lower": entry["ci_lower"],
                        "ci_upper": entry["ci_upper"],
                        "n_samples": entry.get("n_samples_with_this_step", 0),
                    })
                output["truncation"][model][dataset] = {
                    "scr_by_step": scr_summary,
                    "max_steps_seen": trunc.get("max_steps_seen", 0),
                }

            # --- Corruption ---
            corr = load_json(os.path.join(RESULTS_DIR, "corruption", f"{ms}_{dataset}.json"))
            if corr:
                cfr_summary = {}
                for cond, metrics in corr.get("cfr_by_condition", {}).items():
                    cfr_summary[cond] = {
                        "cfr": metrics["cfr"],
                        "ci_lower": metrics.get("ci_lower", 0),
                        "ci_upper": metrics.get("ci_upper", 0),
                        "n_total": metrics["n_total"],
                    }
                output["corruption"][model][dataset] = cfr_summary

            # --- Biased Hints ---
            hints = load_json(os.path.join(RESULTS_DIR, "biased_hints", f"{ms}_{dataset}.json"))
            if hints:
                hint_summary = {}
                for strength, metrics in hints.get("metrics_by_strength", {}).items():
                    hint_summary[strength] = {
                        "har": metrics["har"]["har"],
                        "har_ci": [metrics["har"]["ci_lower"], metrics["har"]["ci_upper"]],
                        "sbh": metrics["sbh"]["sbh"],
                        "sbh_ci": [metrics["sbh"]["ci_lower"], metrics["sbh"]["ci_upper"]],
                        "steering_rate": metrics["steering_rate"],
                        "outcome_counts": metrics["outcome_counts"],
                    }
                output["biased_hints"][model][dataset] = hint_summary

    # --- Cross-model comparison (McNemar's test) ---
    if len(models) >= 2:
        for dataset in datasets:
            pair_key = f"{models[0]}_vs_{models[1]}"
            results_a = load_json(os.path.join(RESULTS_DIR, "baseline", f"{_model_safe(models[0])}_{dataset}.json"))
            results_b = load_json(os.path.join(RESULTS_DIR, "baseline", f"{_model_safe(models[1])}_{dataset}.json"))

            if results_a and results_b:
                # Build contingency table for CoT accuracy
                a_correct = [r["cot_correct"] for r in results_a["results"]]
                b_correct = [r["cot_correct"] for r in results_b["results"]]
                n = min(len(a_correct), len(b_correct))

                both_correct = sum(a_correct[i] and b_correct[i] for i in range(n))
                a_only = sum(a_correct[i] and not b_correct[i] for i in range(n))
                b_only = sum(not a_correct[i] and b_correct[i] for i in range(n))
                both_wrong = sum(not a_correct[i] and not b_correct[i] for i in range(n))

                table = np.array([[both_correct, a_only], [b_only, both_wrong]])
                test_result = mcnemar_test(table)

                if dataset not in output["cross_model_comparison"]:
                    output["cross_model_comparison"][dataset] = {}

                output["cross_model_comparison"][dataset][pair_key] = {
                    "mcnemar_pvalue": test_result["pvalue"],
                    "mcnemar_statistic": test_result["statistic"],
                    "contingency_table": table.tolist(),
                }

    # Save aggregated output
    out_path = os.path.join(RESULTS_DIR, "aggregated_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Aggregated results saved to {out_path}")

    return output
