"""Generate plots and tables from aggregated results."""

import json
import os
import argparse

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend
sns.set_theme(style="whitegrid", font_scale=1.1)

from config import FIGURES_DIR, RESULTS_DIR, CORRUPTION_CONDITIONS, HINT_TEMPLATES


def load_aggregated(input_dir: str = RESULTS_DIR) -> dict:
    path = os.path.join(input_dir, "aggregated_results.json")
    with open(path) as f:
        return json.load(f)


# ---------- Plot 1: Baseline Accuracy Comparison ----------

def plot_baseline(data: dict, output_dir: str = FIGURES_DIR) -> None:
    """Bar chart comparing No-CoT vs CoT accuracy across models and datasets."""
    os.makedirs(output_dir, exist_ok=True)
    baseline = data["baseline"]

    models = list(baseline.keys())
    datasets = list(next(iter(baseline.values())).keys()) if baseline else []

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False)

    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        no_cot_vals = []
        cot_vals = []
        labels = []

        for model in models:
            if dataset in baseline[model]:
                no_cot_vals.append(baseline[model][dataset]["no_cot_accuracy"])
                cot_vals.append(baseline[model][dataset]["cot_accuracy"])
                labels.append(model)

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width / 2, no_cot_vals, width, label="No CoT", color="#5B9BD5")
        bars2 = ax.bar(x + width / 2, cot_vals, width, label="CoT", color="#ED7D31")

        ax.set_ylabel("Accuracy")
        ax.set_title(f"Baseline: {dataset.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()

        # Add value labels
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "baseline_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------- Plot 2: SCR by Step (Truncation) ----------

def plot_truncation(data: dict, output_dir: str = FIGURES_DIR) -> None:
    """Line plot of SCR by truncation step for each model/dataset."""
    os.makedirs(output_dir, exist_ok=True)
    truncation = data["truncation"]

    models = list(truncation.keys())
    datasets = set()
    for m in models:
        datasets.update(truncation[m].keys())
    datasets = sorted(datasets)

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)

    colors = sns.color_palette("husl", len(models))

    for j, dataset in enumerate(datasets):
        ax = axes[0][j]

        for i, model in enumerate(models):
            if dataset not in truncation[model]:
                continue
            scr_data = truncation[model][dataset]["scr_by_step"]
            if not scr_data:
                continue

            steps = [d["step_k"] for d in scr_data]
            scr_vals = [d["scr"] for d in scr_data]
            ci_lower = [d["ci_lower"] for d in scr_data]
            ci_upper = [d["ci_upper"] for d in scr_data]

            ax.plot(steps, scr_vals, marker="o", label=model, color=colors[i])
            ax.fill_between(steps, ci_lower, ci_upper, alpha=0.15, color=colors[i])

        ax.set_xlabel("Truncation Step (k)")
        ax.set_ylabel("Step Consistency Rate (SCR)")
        ax.set_title(f"Truncation: {dataset.upper()}")
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "truncation_scr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------- Plot 3: CFR by Corruption Condition ----------

def plot_corruption(data: dict, output_dir: str = FIGURES_DIR) -> None:
    """Grouped bar chart of CFR by corruption condition."""
    os.makedirs(output_dir, exist_ok=True)
    corruption = data["corruption"]

    models = list(corruption.keys())
    datasets = set()
    for m in models:
        datasets.update(corruption[m].keys())
    datasets = sorted(datasets)

    conditions = [c for c in CORRUPTION_CONDITIONS if c != "none"]

    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5), squeeze=False)
    colors = sns.color_palette("husl", len(models))

    for j, dataset in enumerate(datasets):
        ax = axes[0][j]
        x = np.arange(len(conditions))
        width = 0.8 / max(len(models), 1)

        for i, model in enumerate(models):
            if dataset not in corruption[model]:
                continue

            cfr_vals = []
            ci_lo = []
            ci_hi = []
            for cond in conditions:
                entry = corruption[model][dataset].get(cond, {})
                cfr_vals.append(entry.get("cfr", 0))
                ci_lo.append(entry.get("ci_lower", 0))
                ci_hi.append(entry.get("ci_upper", 0))

            offset = (i - len(models) / 2 + 0.5) * width
            bars = ax.bar(x + offset, cfr_vals, width, label=model, color=colors[i])
            ax.errorbar(
                x + offset, cfr_vals,
                yerr=[
                    [v - lo for v, lo in zip(cfr_vals, ci_lo)],
                    [hi - v for v, hi in zip(cfr_vals, ci_hi)],
                ],
                fmt="none", ecolor="gray", capsize=3,
            )

        ax.set_ylabel("Corruption Following Rate (CFR)")
        ax.set_title(f"Corruption: {dataset.upper()}")
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "corruption_cfr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ---------- Plot 4: Biased Hints - HAR, SBH, Steering Rate ----------

def plot_biased_hints(data: dict, output_dir: str = FIGURES_DIR) -> None:
    """Multi-panel plot for hint experiment metrics."""
    os.makedirs(output_dir, exist_ok=True)
    hints = data["biased_hints"]

    models = list(hints.keys())
    datasets = set()
    for m in models:
        datasets.update(hints[m].keys())
    datasets = sorted(datasets)

    strengths = list(HINT_TEMPLATES.keys())
    metric_names = [("har", "Hint Acknowledgment Rate"), ("sbh", "Steered-But-Hidden Rate"), ("steering_rate", "Steering Rate")]

    for dataset in datasets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = sns.color_palette("husl", len(models))

        for k, (metric_key, metric_label) in enumerate(metric_names):
            ax = axes[k]

            for i, model in enumerate(models):
                if dataset not in hints[model]:
                    continue

                vals = []
                for s in strengths:
                    entry = hints[model][dataset].get(s, {})
                    vals.append(entry.get(metric_key, 0))

                ax.plot(strengths, vals, marker="s", label=model, color=colors[i], linewidth=2)

            ax.set_xlabel("Hint Strength")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label}\n({dataset.upper()})")
            ax.set_ylim(0, 1.0)
            ax.legend()

        plt.tight_layout()
        path = os.path.join(output_dir, f"biased_hints_{dataset}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


# ---------- Plot 5: Outcome Distribution Heatmap ----------

def plot_outcome_heatmap(data: dict, output_dir: str = FIGURES_DIR) -> None:
    """Heatmap showing outcome distribution for hint experiment."""
    os.makedirs(output_dir, exist_ok=True)
    hints = data["biased_hints"]

    models = list(hints.keys())
    datasets = set()
    for m in models:
        datasets.update(hints[m].keys())
    datasets = sorted(datasets)

    strengths = list(HINT_TEMPLATES.keys())
    outcome_types = ["FAITHFUL_REJECT", "FAITHFUL_FOLLOW", "UNFAITHFUL_IGNORE", "STEERED_BUT_HIDDEN"]

    for model in models:
        for dataset in datasets:
            if dataset not in hints[model]:
                continue

            matrix = []
            for s in strengths:
                entry = hints[model][dataset].get(s, {})
                counts = entry.get("outcome_counts", {})
                total = sum(counts.values()) or 1
                row = [counts.get(o, 0) / total for o in outcome_types]
                matrix.append(row)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(
                np.array(matrix),
                annot=True, fmt=".2f",
                xticklabels=[o.replace("_", "\n") for o in outcome_types],
                yticklabels=strengths,
                cmap="YlOrRd", vmin=0, vmax=1,
                ax=ax,
            )
            model_safe = model.replace(":", "_").replace("/", "_")
            ax.set_title(f"Outcome Distribution: {model} / {dataset.upper()}")
            ax.set_ylabel("Hint Strength")

            plt.tight_layout()
            path = os.path.join(output_dir, f"heatmap_{model_safe}_{dataset}.png")
            plt.savefig(path, dpi=150)
            plt.close()
            print(f"  Saved {path}")


# ---------- Summary Table ----------

def print_summary_table(data: dict) -> None:
    """Print a text summary table of key metrics."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    # Baseline
    print("\n--- Baseline Accuracy ---")
    for model, datasets in data["baseline"].items():
        for dataset, vals in datasets.items():
            print(f"  {model:20s} | {dataset:8s} | No-CoT: {vals['no_cot_accuracy']:.3f} | CoT: {vals['cot_accuracy']:.3f}")

    # Truncation
    print("\n--- Truncation SCR (step 1) ---")
    for model, datasets in data["truncation"].items():
        for dataset, vals in datasets.items():
            scr_data = vals.get("scr_by_step", [])
            if scr_data:
                scr1 = scr_data[0]["scr"]
                print(f"  {model:20s} | {dataset:8s} | SCR@step1: {scr1:.3f}")

    # Corruption
    print("\n--- Corruption CFR ---")
    for model, datasets in data["corruption"].items():
        for dataset, conds in datasets.items():
            parts = []
            for cond in ["early", "middle", "late", "early_late", "all"]:
                if cond in conds:
                    parts.append(f"{cond}={conds[cond]['cfr']:.3f}")
            print(f"  {model:20s} | {dataset:8s} | {' | '.join(parts)}")

    # Hints
    print("\n--- Biased Hints (Authoritative) ---")
    for model, datasets in data["biased_hints"].items():
        for dataset, strengths in datasets.items():
            auth = strengths.get("authoritative", {})
            print(f"  {model:20s} | {dataset:8s} | HAR: {auth.get('har', 0):.3f} | SBH: {auth.get('sbh', 0):.3f} | Steering: {auth.get('steering_rate', 0):.3f}")

    # Cross-model
    if data.get("cross_model_comparison"):
        print("\n--- Cross-Model Comparison (McNemar's test) ---")
        for dataset, comparisons in data["cross_model_comparison"].items():
            for pair, vals in comparisons.items():
                sig = "SIGNIFICANT" if vals["mcnemar_pvalue"] < 0.05 else "not significant"
                print(f"  {dataset:8s} | {pair} | p={vals['mcnemar_pvalue']:.4f} ({sig})")

    print("\n" + "=" * 80)


# ---------- Main ----------

def generate_all_plots(input_dir: str = RESULTS_DIR, output_dir: str = FIGURES_DIR) -> None:
    """Generate all plots from aggregated results."""
    data = load_aggregated(input_dir)

    print("Generating plots...")
    plot_baseline(data, output_dir)
    plot_truncation(data, output_dir)
    plot_corruption(data, output_dir)
    plot_biased_hints(data, output_dir)
    plot_outcome_heatmap(data, output_dir)
    print_summary_table(data)
    print("\nAll plots saved to", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations from experiment results.")
    parser.add_argument("--input", default=RESULTS_DIR, help="Input results directory")
    parser.add_argument("--output", default=FIGURES_DIR, help="Output figures directory")
    args = parser.parse_args()

    generate_all_plots(args.input, args.output)
