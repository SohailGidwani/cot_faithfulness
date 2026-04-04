from __future__ import annotations
import json, os
from tqdm import tqdm

import config
from models.ollama_client import query_model
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc


# ---- prompts ----

def _nocot_gsm(q):
    return ("Answer this math question directly with no explanation. "
            "Give only the final numeric answer.\n\n"
            "Question: %s\n\nAnswer:" % q)

def _cot_gsm(q):
    return ("Think step by step, then provide your final answer after ####.\n\n"
            "Question: " + q)

def _nocot_arc(q, ch):
    return ("Answer this question directly with no explanation. "
            "Give only the letter of the correct choice.\n\n"
            "Question: %s\n%s\n\nAnswer:" % (q, ch))

def _cot_arc(q, ch):
    return ("Think step by step, then provide your final answer as a single letter.\n\n"
            "Question: {}\n{}".format(q, ch))


def _nums_eq(a, b):
    try:
        return float(a) == float(b)
    except (ValueError, TypeError):
        return a.strip() == b.strip()


# ---- main ----

def run_baseline(model, dataset, samples):
    per_sample = []
    right_nocot = 0
    right_cot = 0

    for s in tqdm(samples, desc="Baseline %s/%s" % (model, dataset)):
        q = s["question"]
        gold = s["gold_answer"]

        if dataset == "gsm8k":
            p_nc = _nocot_gsm(q)
            p_c  = _cot_gsm(q)
        else:
            p_nc = _nocot_arc(q, s["choices_text"])
            p_c  = _cot_arc(q, s["choices_text"])

        r_nc = query_model(model, p_nc)
        r_c  = query_model(model, p_c)

        if dataset == "gsm8k":
            a_nc = extract_answer_gsm8k(r_nc)
            a_c  = extract_answer_gsm8k(r_c)
            nc_ok = _nums_eq(a_nc, gold)
            c_ok  = _nums_eq(a_c, gold)
        else:
            labels = s.get("choice_labels")
            a_nc = extract_answer_arc(r_nc, labels)
            a_c  = extract_answer_arc(r_c, labels)
            nc_ok = a_nc.upper() == gold.upper()
            c_ok  = a_c.upper() == gold.upper()

        right_nocot += int(nc_ok)
        right_cot += int(c_ok)

        per_sample.append({
            "id": s["id"], "question": q, "gold_answer": gold,
            "no_cot_prompt": p_nc, "no_cot_response": r_nc,
            "no_cot_answer": a_nc, "no_cot_correct": nc_ok,
            "cot_prompt": p_c, "cot_response": r_c,
            "cot_answer": a_c, "cot_correct": c_ok,
        })

    n = len(samples)
    summary = {
        "model": model, "dataset": dataset, "n_total": n,
        "no_cot_accuracy": right_nocot / n if n else 0,
        "cot_accuracy": right_cot / n if n else 0,
        "n_no_cot_correct": right_nocot,
        "n_cot_correct": right_cot,
        "results": per_sample,
    }
    _dump(model, dataset, summary)
    return summary


def _dump(model, dataset, blob):
    tag = model.replace(":", "_").replace("/", "_")
    d = os.path.join(config.RESULTS_DIR, "baseline")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s_%s.json" % (tag, dataset))
    with open(p, "w") as f:
        json.dump(blob, f, indent=2)
    print("  Saved baseline ->", p)
