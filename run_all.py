"""Main entry point: run all CoT faithfulness experiments.

Usage:
    python run_all.py --models llama3.2:3b qwen2.5:7b \
                      --datasets gsm8k arc \
                      --samples 250 \
                      --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import config
from data.gsm8k_loader import load_gsm8k_samples
from data.arc_loader import load_arc_samples
from experiments.baseline import run_baseline
from experiments.truncation import run_truncation
from experiments.corruption_exp import run_corruption
from experiments.biased_hints import run_biased_hints
from analysis.aggregate_results import aggregate_all
from analysis.visualize import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CoT faithfulness experiments."
    )
    parser.add_argument(
        "--models", nargs="+", default=config.MODELS,
        help="Ollama model names to test",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=config.DATASETS,
        choices=["gsm8k", "arc"],
        help="Datasets to use",
    )
    parser.add_argument(
        "--samples", type=int, default=config.SAMPLES_PER_DATASET,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--output", default=config.RESULTS_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip baseline experiment (use existing results)",
    )
    parser.add_argument(
        "--skip-truncation", action="store_true",
        help="Skip truncation experiment",
    )
    parser.add_argument(
        "--skip-corruption", action="store_true",
        help="Skip corruption experiment",
    )
    parser.add_argument(
        "--skip-hints", action="store_true",
        help="Skip biased hints experiment",
    )
    parser.add_argument(
        "--only", choices=["baseline", "truncation", "corruption", "hints"],
        help="Run only a specific experiment",
    )
    return parser.parse_args()


def load_data(datasets: list[str], n_samples: int) -> dict[str, list[dict]]:
    """Load and sample all datasets."""
    data = {}
    if "gsm8k" in datasets:
        print(f"Loading GSM8K ({n_samples} samples)...")
        data["gsm8k"] = load_gsm8k_samples(n_samples)
        print(f"  Loaded {len(data['gsm8k'])} GSM8K samples")

    if "arc" in datasets:
        print(f"Loading ARC-Challenge ({n_samples} samples)...")
        data["arc"] = load_arc_samples(n_samples)
        print(f"  Loaded {len(data['arc'])} ARC samples")

    return data


def load_baseline_results(model: str, dataset: str, output_dir: str) -> list[dict] | None:
    """Load baseline results from disk (needed by later experiments)."""
    model_safe = model.replace(":", "_").replace("/", "_")
    path = os.path.join(output_dir, "baseline", f"{model_safe}_{dataset}.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data["results"]
    return None


def should_run(experiment: str, args) -> bool:
    """Check if an experiment should run based on CLI flags."""
    if args.only:
        return args.only == experiment
    skip_flag = f"skip_{experiment}"
    return not getattr(args, skip_flag, False)


def main():
    args = parse_args()

    # Override config
    config.RESULTS_DIR = args.output
    config.SAMPLES_PER_DATASET = args.samples
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("CoT Faithfulness Analysis")
    print("=" * 60)
    print(f"Models:   {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Samples:  {args.samples}")
    print(f"Output:   {args.output}")
    print("=" * 60)

    # Load data
    data = load_data(args.datasets, args.samples)

    total_start = time.time()

    for model in args.models:
        print(f"\n{'='*60}")
        print(f"MODEL: {model}")
        print(f"{'='*60}")

        for dataset in args.datasets:
            samples = data[dataset]
            print(f"\n--- Dataset: {dataset.upper()} ({len(samples)} samples) ---")

            # Experiment 0: Baseline
            if should_run("baseline", args):
                print("\n[Experiment 0] Running baseline (No-CoT vs CoT)...")
                t0 = time.time()
                baseline_summary = run_baseline(model, dataset, samples)
                elapsed = time.time() - t0
                print(f"  No-CoT accuracy: {baseline_summary['no_cot_accuracy']:.3f}")
                print(f"  CoT accuracy:    {baseline_summary['cot_accuracy']:.3f}")
                print(f"  Time: {elapsed:.1f}s")

            # Load baseline results (needed by experiments 1-3)
            baseline_results = load_baseline_results(model, dataset, args.output)
            if baseline_results is None:
                print("  ERROR: Baseline results not found. Run baseline first.")
                continue

            # Experiment 1: Truncation
            if should_run("truncation", args):
                print("\n[Experiment 1] Running step-based truncation...")
                t0 = time.time()
                trunc_summary = run_truncation(model, dataset, samples, baseline_results)
                elapsed = time.time() - t0
                scr_data = trunc_summary.get("scr_by_step", [])
                if scr_data:
                    print(f"  SCR@step1: {scr_data[0]['scr']:.3f}")
                print(f"  Time: {elapsed:.1f}s")

            # Experiment 2: Corruption
            if should_run("corruption", args):
                print("\n[Experiment 2] Running reasoning corruption...")
                t0 = time.time()
                corr_summary = run_corruption(model, dataset, samples, baseline_results)
                elapsed = time.time() - t0
                for cond, metrics in corr_summary["cfr_by_condition"].items():
                    if cond != "none":
                        print(f"  CFR ({cond}): {metrics['cfr']:.3f}")
                print(f"  Time: {elapsed:.1f}s")

            # Experiment 3: Biased Hints
            if should_run("hints", args):
                print("\n[Experiment 3] Running biased hints...")
                t0 = time.time()
                hints_summary = run_biased_hints(model, dataset, samples, baseline_results)
                elapsed = time.time() - t0
                for strength, metrics in hints_summary["metrics_by_strength"].items():
                    print(f"  {strength:15s} | HAR: {metrics['har']['har']:.3f} | SBH: {metrics['sbh']['sbh']:.3f} | Steering: {metrics['steering_rate']:.3f}")
                print(f"  Time: {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print(f"{'='*60}")

    # Aggregate and visualize
    print("\nAggregating results...")
    aggregate_all(args.models, args.datasets)

    print("\nGenerating visualizations...")
    generate_all_plots(args.output, config.FIGURES_DIR)

    print("\nDone!")


if __name__ == "__main__":
    main()
