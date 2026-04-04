from __future__ import annotations
import json, os
import numpy as np
from datetime import datetime

import config
from config import MODELS, DATASETS, NUM_SAMPLES, CORRUPTION_CONDITIONS, HINT_TEMPLATES
from metrics.statistical_tests import mcnemar_test


def _load(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None

def _safe(model):
    return model.replace(":", "_").replace("/", "_")


def aggregate_all(models=MODELS, datasets=DATASETS):
    out = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "models": models, "datasets": datasets,
            "samples_per_dataset": NUM_SAMPLES,
        },
        "baseline": {}, "truncation": {}, "corruption": {},
        "biased_hints": {}, "cross_model_comparison": {},
    }

    for m in models:
        ms = _safe(m)
        out["baseline"][m] = {}
        out["truncation"][m] = {}
        out["corruption"][m] = {}
        out["biased_hints"][m] = {}

        for ds in datasets:
            # baseline
            bl = _load(os.path.join(config.RESULTS_DIR, "baseline", "%s_%s.json" % (ms, ds)))
            if bl:
                out["baseline"][m][ds] = {
                    "no_cot_accuracy": bl["no_cot_accuracy"],
                    "cot_accuracy": bl["cot_accuracy"],
                    "n_total": bl["n_total"],
                }

            # truncation
            tr = _load(os.path.join(config.RESULTS_DIR, "truncation", "%s_%s.json" % (ms, ds)))
            if tr:
                rows = []
                for e in tr.get("scr_by_step", []):
                    rows.append({
                        "step_k": e["step_k"], "scr": e["scr"],
                        "ci_lower": e["ci_lower"], "ci_upper": e["ci_upper"],
                        "n_samples": e.get("n_samples_with_this_step", 0),
                    })
                out["truncation"][m][ds] = {
                    "scr_by_step": rows,
                    "max_steps_seen": tr.get("max_steps_seen", 0),
                }

            # corruption
            cr = _load(os.path.join(config.RESULTS_DIR, "corruption", "%s_%s.json" % (ms, ds)))
            if cr:
                cfr_map = {}
                for cond, met in cr.get("cfr_by_condition", {}).items():
                    cfr_map[cond] = {
                        "cfr": met["cfr"],
                        "ci_lower": met.get("ci_lower", 0),
                        "ci_upper": met.get("ci_upper", 0),
                        "n_total": met["n_total"],
                    }
                out["corruption"][m][ds] = cfr_map

            # biased hints
            bh = _load(os.path.join(config.RESULTS_DIR, "biased_hints", "%s_%s.json" % (ms, ds)))
            if bh:
                hmap = {}
                for stren, met in bh.get("metrics_by_strength", {}).items():
                    hmap[stren] = {
                        "har": met["har"]["har"],
                        "har_ci": [met["har"]["ci_lower"], met["har"]["ci_upper"]],
                        "sbh": met["sbh"]["sbh"],
                        "sbh_ci": [met["sbh"]["ci_lower"], met["sbh"]["ci_upper"]],
                        "steering_rate": met["steering_rate"],
                        "outcome_counts": met["outcome_counts"],
                    }
                out["biased_hints"][m][ds] = hmap

    # cross-model mcnemar
    if len(models) >= 2:
        for ds in datasets:
            rA = _load(os.path.join(config.RESULTS_DIR, "baseline",
                                    "%s_%s.json" % (_safe(models[0]), ds)))
            rB = _load(os.path.join(config.RESULTS_DIR, "baseline",
                                    "%s_%s.json" % (_safe(models[1]), ds)))
            if rA and rB:
                a_ok = [r["cot_correct"] for r in rA["results"]]
                b_ok = [r["cot_correct"] for r in rB["results"]]
                n = min(len(a_ok), len(b_ok))
                both_r = sum(a_ok[i] and b_ok[i] for i in range(n))
                a_only = sum(a_ok[i] and not b_ok[i] for i in range(n))
                b_only = sum(not a_ok[i] and b_ok[i] for i in range(n))
                both_w = sum(not a_ok[i] and not b_ok[i] for i in range(n))

                tbl = np.array([[both_r, a_only], [b_only, both_w]])
                tst = mcnemar_test(tbl)

                key = "%s_vs_%s" % (models[0], models[1])
                if ds not in out["cross_model_comparison"]:
                    out["cross_model_comparison"][ds] = {}
                out["cross_model_comparison"][ds][key] = {
                    "mcnemar_pvalue": tst["pvalue"],
                    "mcnemar_statistic": tst["statistic"],
                    "contingency_table": tbl.tolist(),
                }

    dest = os.path.join(config.RESULTS_DIR, "aggregated_results.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print("Aggregated results ->", dest)
    return out
