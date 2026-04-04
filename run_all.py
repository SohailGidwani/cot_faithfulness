"""
Usage:
    python run_all.py --models llama3.2:3b qwen2.5:7b \
                      --datasets gsm8k arc --samples 250
"""
from __future__ import annotations
import argparse, json, os, time

import config
from data.gsm8k_loader import load_gsm8k_samples
from data.arc_loader import load_arc_samples
from experiments.baseline import run_baseline
from experiments.truncation import run_truncation
from experiments.corruption_exp import run_corruption
from experiments.biased_hints import run_biased_hints
from analysis.aggregate_results import aggregate_all
from analysis.visualize import generate_all_plots


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=config.MODELS)
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS,
                    choices=["gsm8k", "arc"])
    ap.add_argument("--samples", type=int, default=config.NUM_SAMPLES)
    ap.add_argument("--output", default=config.RESULTS_DIR)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-truncation", action="store_true")
    ap.add_argument("--skip-corruption", action="store_true")
    ap.add_argument("--skip-hints", action="store_true")
    ap.add_argument("--only", choices=["baseline","truncation","corruption","hints"])
    return ap.parse_args()


def _load_data(datasets, n):
    data = {}
    if "gsm8k" in datasets:
        print("Loading GSM8K (%d samples)..." % n)
        data["gsm8k"] = load_gsm8k_samples(n)
        print("  got %d" % len(data["gsm8k"]))
    if "arc" in datasets:
        print("Loading ARC-Challenge (%d samples)..." % n)
        data["arc"] = load_arc_samples(n)
        print("  got %d" % len(data["arc"]))
    return data


def _load_baseline(model, dataset, outdir):
    tag = model.replace(":", "_").replace("/", "_")
    p = os.path.join(outdir, "baseline", "%s_%s.json" % (tag, dataset))
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)["results"]
    return None


def _should_run(exp, args):
    if args.only:
        return args.only == exp
    return not getattr(args, "skip_" + exp, False)


def main():
    args = _parse()
    config.RESULTS_DIR = args.output
    config.NUM_SAMPLES = args.samples
    os.makedirs(args.output, exist_ok=True)

    print("=" * 60)
    print("CoT Faithfulness Analysis")
    print("=" * 60)
    print("Models:  ", args.models)
    print("Datasets:", args.datasets)
    print("Samples: ", args.samples)
    print("Output:  ", args.output)
    print("=" * 60)

    data = _load_data(args.datasets, args.samples)
    t_start = time.time()

    for model in args.models:
        print("\n" + "=" * 60)
        print("MODEL:", model)
        print("=" * 60)

        for ds in args.datasets:
            samps = data[ds]
            print("\n--- %s (%d samples) ---" % (ds.upper(), len(samps)))

            if _should_run("baseline", args):
                print("\n[Exp 0] Baseline...")
                t0 = time.time()
                bsum = run_baseline(model, ds, samps)
                print("  No-CoT: %.3f  CoT: %.3f  (%.1fs)"
                      % (bsum["no_cot_accuracy"], bsum["cot_accuracy"],
                         time.time() - t0))

            bl = _load_baseline(model, ds, args.output)
            if bl is None:
                print("  ERROR: baseline results missing, run baseline first")
                continue

            if _should_run("truncation", args):
                print("\n[Exp 1] Truncation...")
                t0 = time.time()
                tsum = run_truncation(model, ds, samps, bl)
                scr_data = tsum.get("scr_by_step", [])
                if scr_data:
                    print("  SCR@step1: %.3f" % scr_data[0]["scr"], end="")
                print("  (%.1fs)" % (time.time() - t0))

            if _should_run("corruption", args):
                print("\n[Exp 2] Corruption...")
                t0 = time.time()
                csum = run_corruption(model, ds, samps, bl)
                for c, met in csum["cfr_by_condition"].items():
                    if c != "none":
                        print("  CFR(%s): %.3f" % (c, met["cfr"]))
                print("  (%.1fs)" % (time.time() - t0))

            if _should_run("hints", args):
                print("\n[Exp 3] Biased hints...")
                t0 = time.time()
                hsum = run_biased_hints(model, ds, samps, bl)
                for s, met in hsum["metrics_by_strength"].items():
                    print("  %-15s | HAR: %.3f | SBH: %.3f | Steer: %.3f"
                          % (s, met["har"]["har"], met["sbh"]["sbh"],
                             met["steering_rate"]))
                print("  (%.1fs)" % (time.time() - t0))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("All experiments done in %.1fs" % elapsed)
    print("=" * 60)

    print("\nAggregating...")
    aggregate_all(args.models, args.datasets)

    print("\nPlotting...")
    generate_all_plots(args.output, config.FIGURES_DIR)
    print("\nDone!")


if __name__ == "__main__":
    main()
