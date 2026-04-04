from __future__ import annotations
import json, os, argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import config
from config import CORRUPTION_CONDITIONS, HINT_TEMPLATES

plt.rcParams.update({"font.size": 11})


def _load_agg(input_dir=config.RESULTS_DIR):
    with open(os.path.join(input_dir, "aggregated_results.json")) as f:
        return json.load(f)


# -------- baseline bar chart --------

def plot_baseline(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    bl = data["baseline"]
    mdls = list(bl.keys())
    dsets = list(next(iter(bl.values())).keys()) if bl else []

    fig, axes = plt.subplots(1, len(dsets), figsize=(6 * len(dsets), 5),
                             squeeze=False)
    for j, ds in enumerate(dsets):
        ax = axes[0][j]
        nc_vals, c_vals, lbls = [], [], []
        for m in mdls:
            if ds in bl[m]:
                nc_vals.append(bl[m][ds]["no_cot_accuracy"])
                c_vals.append(bl[m][ds]["cot_accuracy"])
                lbls.append(m)

        x = np.arange(len(lbls))
        w = 0.35
        b1 = ax.bar(x - w/2, nc_vals, w, label="No CoT", color="#5B9BD5")
        b2 = ax.bar(x + w/2, c_vals, w, label="CoT", color="#ED7D31")
        ax.set_ylabel("Accuracy")
        ax.set_title("Baseline: " + ds.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, rotation=15, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()
        for bar in list(b1) + list(b2):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    "%.2f" % bar.get_height(), ha="center", va="bottom",
                    fontsize=9)

    plt.tight_layout()
    p = os.path.join(outdir, "baseline_accuracy.png")
    plt.savefig(p, dpi=150); plt.close()
    print("  Saved", p)


# -------- truncation SCR lines --------

def plot_truncation(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    trunc = data["truncation"]
    mdls = list(trunc.keys())
    dsets = sorted(set(d for m in mdls for d in trunc[m]))

    fig, axes = plt.subplots(1, len(dsets), figsize=(7*len(dsets), 5),
                             squeeze=False)
    pal = sns.color_palette("tab10", len(mdls))

    for j, ds in enumerate(dsets):
        ax = axes[0][j]
        for i, m in enumerate(mdls):
            if ds not in trunc[m]:
                continue
            rows = trunc[m][ds]["scr_by_step"]
            if not rows:
                continue
            ks = [r["step_k"] for r in rows]
            vals = [r["scr"] for r in rows]
            lo = [r["ci_lower"] for r in rows]
            hi = [r["ci_upper"] for r in rows]
            ax.plot(ks, vals, "o-", label=m, color=pal[i])
            ax.fill_between(ks, lo, hi, alpha=0.15, color=pal[i])
        ax.set_xlabel("Truncation Step (k)")
        ax.set_ylabel("Step Consistency Rate (SCR)")
        ax.set_title("Truncation: " + ds.upper())
        ax.set_ylim(0, 1.05)
        ax.legend()

    plt.tight_layout()
    p = os.path.join(outdir, "truncation_scr.png")
    plt.savefig(p, dpi=150); plt.close()
    print("  Saved", p)


# -------- corruption CFR grouped bars --------

def plot_corruption(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    corr = data["corruption"]
    mdls = list(corr.keys())
    dsets = sorted(set(d for m in mdls for d in corr[m]))
    conds = [c for c in CORRUPTION_CONDITIONS if c != "none"]

    fig, axes = plt.subplots(1, len(dsets), figsize=(7*len(dsets), 5),
                             squeeze=False)
    pal = sns.color_palette("tab10", len(mdls))

    for j, ds in enumerate(dsets):
        ax = axes[0][j]
        x = np.arange(len(conds))
        w = 0.8 / max(len(mdls), 1)
        for i, m in enumerate(mdls):
            if ds not in corr[m]:
                continue
            vals, cl, ch = [], [], []
            for c in conds:
                e = corr[m][ds].get(c, {})
                vals.append(e.get("cfr", 0))
                cl.append(e.get("ci_lower", 0))
                ch.append(e.get("ci_upper", 0))
            off = (i - len(mdls)/2 + 0.5) * w
            ax.bar(x + off, vals, w, label=m, color=pal[i])
            ax.errorbar(x + off, vals,
                        yerr=[[v - l for v, l in zip(vals, cl)],
                              [h - v for v, h in zip(vals, ch)]],
                        fmt="none", ecolor="gray", capsize=3)
        ax.set_ylabel("Corruption Following Rate (CFR)")
        ax.set_title("Corruption: " + ds.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.legend()

    plt.tight_layout()
    p = os.path.join(outdir, "corruption_cfr.png")
    plt.savefig(p, dpi=150); plt.close()
    print("  Saved", p)


# -------- biased hints line plots --------

def plot_biased_hints(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    hints = data["biased_hints"]
    mdls = list(hints.keys())
    dsets = sorted(set(d for m in mdls for d in hints[m]))
    strengths = list(HINT_TEMPLATES.keys())
    metrics = [("har", "Hint Acknowledgment Rate"),
               ("sbh", "Steered-But-Hidden Rate"),
               ("steering_rate", "Steering Rate")]

    pal = sns.color_palette("tab10", len(mdls))
    for ds in dsets:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for k, (mkey, mlabel) in enumerate(metrics):
            ax = axes[k]
            for i, m in enumerate(mdls):
                if ds not in hints[m]:
                    continue
                vals = [hints[m][ds].get(s, {}).get(mkey, 0) for s in strengths]
                ax.plot(strengths, vals, "s-", label=m, color=pal[i], lw=2)
            ax.set_xlabel("Hint Strength")
            ax.set_ylabel(mlabel)
            ax.set_title(mlabel + " (" + ds.upper() + ")")
            ax.set_ylim(0, 1.0)
            ax.legend()
        plt.tight_layout()
        p = os.path.join(outdir, "biased_hints_%s.png" % ds)
        plt.savefig(p, dpi=150); plt.close()
        print("  Saved", p)


# -------- outcome heatmap --------

def plot_outcome_heatmap(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    hints = data["biased_hints"]
    mdls = list(hints.keys())
    dsets = sorted(set(d for m in mdls for d in hints[m]))
    strengths = list(HINT_TEMPLATES.keys())
    outcomes = ["FAITHFUL_REJECT", "FAITHFUL_FOLLOW",
                "UNFAITHFUL_IGNORE", "STEERED_BUT_HIDDEN"]

    for m in mdls:
        for ds in dsets:
            if ds not in hints[m]:
                continue
            grid = []
            for s in strengths:
                counts = hints[m][ds].get(s, {}).get("outcome_counts", {})
                tot = sum(counts.values()) or 1
                grid.append([counts.get(o, 0) / tot for o in outcomes])

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(np.array(grid), annot=True, fmt=".2f",
                        xticklabels=[o.replace("_", "\n") for o in outcomes],
                        yticklabels=strengths,
                        cmap="YlOrRd", vmin=0, vmax=1, ax=ax)
            tag = m.replace(":", "_").replace("/", "_")
            ax.set_title("Outcome Distribution: %s / %s" % (m, ds.upper()))
            ax.set_ylabel("Hint Strength")
            plt.tight_layout()
            p = os.path.join(outdir, "heatmap_%s_%s.png" % (tag, ds))
            plt.savefig(p, dpi=150); plt.close()
            print("  Saved", p)


# -------- summary table --------

def print_summary_table(data):
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print("\n--- Baseline Accuracy ---")
    for m, dsets in data["baseline"].items():
        for ds, v in dsets.items():
            print("  %-20s | %-8s | No-CoT: %.3f | CoT: %.3f"
                  % (m, ds, v["no_cot_accuracy"], v["cot_accuracy"]))

    print("\n--- Truncation SCR (step 1) ---")
    for m, dsets in data["truncation"].items():
        for ds, v in dsets.items():
            rows = v.get("scr_by_step", [])
            if rows:
                print("  %-20s | %-8s | SCR@step1: %.3f" % (m, ds, rows[0]["scr"]))

    print("\n--- Corruption CFR ---")
    for m, dsets in data["corruption"].items():
        for ds, conds in dsets.items():
            bits = []
            for c in ["early", "middle", "late", "early_late", "all"]:
                if c in conds:
                    bits.append("%s=%.3f" % (c, conds[c]["cfr"]))
            print("  %-20s | %-8s | %s" % (m, ds, " | ".join(bits)))

    print("\n--- Biased Hints (Authoritative) ---")
    for m, dsets in data["biased_hints"].items():
        for ds, strs in dsets.items():
            a = strs.get("authoritative", {})
            print("  %-20s | %-8s | HAR: %.3f | SBH: %.3f | Steering: %.3f"
                  % (m, ds, a.get("har", 0), a.get("sbh", 0),
                     a.get("steering_rate", 0)))

    if data.get("cross_model_comparison"):
        print("\n--- Cross-Model Comparison (McNemar's test) ---")
        for ds, comps in data["cross_model_comparison"].items():
            for pair, v in comps.items():
                sig = "SIGNIFICANT" if v["mcnemar_pvalue"] < 0.05 else "not significant"
                print("  %-8s | %s | p=%.4f (%s)"
                      % (ds, pair, v["mcnemar_pvalue"], sig))
    print("\n" + "=" * 80)


# -------- entrypoint --------

def generate_all_plots(input_dir=config.RESULTS_DIR, output_dir=config.FIGURES_DIR):
    data = _load_agg(input_dir)
    print("Generating plots...")
    plot_baseline(data, output_dir)
    plot_truncation(data, output_dir)
    plot_corruption(data, output_dir)
    plot_biased_hints(data, output_dir)
    plot_outcome_heatmap(data, output_dir)
    print_summary_table(data)
    print("\nAll plots saved to", output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=config.RESULTS_DIR)
    ap.add_argument("--output", default=config.FIGURES_DIR)
    args = ap.parse_args()
    generate_all_plots(args.input, args.output)
