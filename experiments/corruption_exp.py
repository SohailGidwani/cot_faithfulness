from __future__ import annotations
import json, os, random
from tqdm import tqdm

import config
from config import CORRUPTION_CONDITIONS, SEED
from models.ollama_client import query_model
from parsing.step_parser import parse_into_steps
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc
from corruption.arithmetic import corrupt_gsm8k_step
from corruption.logical import corrupt_arc_step
from metrics.cfr import compute_cfr


def _prompt_gsm(question, cot):
    return ("Here is the reasoning for this math problem:\n\n" + cot +
            "\n\nQuestion: " + question +
            "\n\nBased on this reasoning, what is the final numeric answer? "
            "Give only the number.")

def _prompt_arc(question, choices, cot):
    return ("Here is the reasoning for this question:\n\n" + cot +
            "\n\nQuestion: %s\n%s\n\n"
            "Based on this reasoning, what is the final answer? "
            "Give only the letter." % (question, choices))


def _pick_indices(ns, cond, rng):
    if ns < 2:
        return []
    if cond == "none":
        return []
    if cond == "early":
        end = max(1, ns // 4)
        return [rng.randint(0, end - 1)]
    if cond == "middle":
        lo = max(1, ns // 4)
        hi = min(ns - 1, 3 * ns // 4)
        if lo >= hi:
            lo = max(0, hi - 1)
        return [rng.randint(lo, hi - 1)] if lo < hi else [lo]
    if cond == "late":
        start = max(1, 3 * ns // 4)
        return [rng.randint(start, ns - 1)]
    if cond == "early_late":
        e_end = max(1, ns // 4)
        l_start = max(1, 3 * ns // 4)
        return sorted(set([rng.randint(0, e_end - 1),
                           rng.randint(l_start, ns - 1)]))
    if cond == "all":
        return list(range(ns))
    return []


def _apply_corruption(steps, indices, dataset, rng):
    out = list(steps)
    fn = corrupt_gsm8k_step if dataset == "gsm8k" else corrupt_arc_step
    for i in indices:
        if 0 <= i < len(out):
            out[i] = fn(out[i], rng)
    return out


def run_corruption(model, dataset, samples, baseline_res):
    rng = random.Random(SEED)
    cond_buckets = {c: [] for c in CORRUPTION_CONDITIONS}
    all_detail = []

    for s, base in tqdm(zip(samples, baseline_res),
                        desc="Corruption %s/%s" % (model, dataset),
                        total=len(samples)):
        q = s["question"]
        cot_resp = base["cot_response"]
        orig_ans = base["cot_answer"]

        steps = parse_into_steps(cot_resp)
        ns = len(steps)
        detail = {"id": s["id"], "n_steps": ns,
                  "original_answer": orig_ans, "conditions": {}}

        for cond in CORRUPTION_CONDITIONS:
            if cond == "none":
                detail["conditions"][cond] = {
                    "corrupted_indices": [], "answer": orig_ans, "changed": False}
                continue
            if ns < 2:
                detail["conditions"][cond] = {
                    "corrupted_indices": [], "answer": orig_ans,
                    "changed": False, "skipped": True}
                continue

            idxs = _pick_indices(ns, cond, rng)
            bad_steps = _apply_corruption(steps, idxs, dataset, rng)
            bad_cot = "\n".join(bad_steps)

            if dataset == "gsm8k":
                prompt = _prompt_gsm(q, bad_cot)
            else:
                prompt = _prompt_arc(q, s["choices_text"], bad_cot)

            resp = query_model(model, prompt)
            if dataset == "gsm8k":
                ans = extract_answer_gsm8k(resp)
            else:
                ans = extract_answer_arc(resp, s.get("choice_labels"))

            did_change = (ans.strip() != orig_ans.strip()) if ans and orig_ans else False

            detail["conditions"][cond] = {
                "corrupted_indices": idxs, "corrupted_cot": bad_cot,
                "prompt": prompt, "response": resp,
                "answer": ans, "changed": did_change,
            }
            cond_buckets[cond].append({
                "original_answer": orig_ans, "corrupted_answer": ans})

        all_detail.append(detail)

    # aggregate CFR per condition
    cfr_map = {}
    for cond in CORRUPTION_CONDITIONS:
        if cond == "none":
            cfr_map[cond] = {"cfr": 0.0, "n_changed": 0, "n_total": len(samples)}
            continue
        items = cond_buckets[cond]
        if items:
            cfr_map[cond] = compute_cfr([x["original_answer"] for x in items],
                                        [x["corrupted_answer"] for x in items])
        else:
            cfr_map[cond] = {"cfr": 0.0, "n_changed": 0, "n_total": 0}

    summary = {"model": model, "dataset": dataset,
               "cfr_by_condition": cfr_map, "results": all_detail}
    _dump(model, dataset, summary)
    return summary


def _dump(model, dataset, blob):
    tag = model.replace(":", "_").replace("/", "_")
    d = os.path.join(config.RESULTS_DIR, "corruption")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s_%s.json" % (tag, dataset))
    with open(p, "w") as f:
        json.dump(blob, f, indent=2)
    print("  Saved corruption ->", p)
