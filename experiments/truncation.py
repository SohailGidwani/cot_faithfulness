from __future__ import annotations
import json, os
from tqdm import tqdm

import config
from models.ollama_client import query_model
from parsing.step_parser import parse_into_steps
from parsing.answer_extractor import extract_answer_gsm8k, extract_answer_arc
from metrics.scr import compute_scr


def _trunc_prompt_gsm(question, partial):
    return ("Given this partial reasoning for a math problem:\n\n"
            + partial + "\n\n"
            "Question: " + question + "\n\n"
            "Based on this reasoning so far, what is the final numeric answer? "
            "Give only the number.")

def _trunc_prompt_arc(question, choices, partial):
    return ("Given this partial reasoning:\n\n"
            + partial + "\n\n"
            "Question: %s\n%s\n\n"
            "Based on this reasoning so far, what is the final answer? "
            "Give only the letter." % (question, choices))


def run_truncation(model, dataset, samples, baseline_res):
    all_out = []
    top_steps = 0

    for s, base in tqdm(zip(samples, baseline_res),
                        desc="Truncation %s/%s" % (model, dataset),
                        total=len(samples)):
        q = s["question"]
        cot_resp = base["cot_response"]
        full_ans = base["cot_answer"]

        steps = parse_into_steps(cot_resp)
        ns = len(steps)
        if ns < 2:
            all_out.append({"id": s["id"], "n_steps": ns, "steps": steps,
                            "full_answer": full_ans, "truncation_results": [],
                            "skipped": True})
            continue

        top_steps = max(top_steps, ns)
        trunc_res = []

        for k in range(1, ns):
            partial = "\n".join(steps[:k])
            if dataset == "gsm8k":
                prompt = _trunc_prompt_gsm(q, partial)
            else:
                prompt = _trunc_prompt_arc(q, s["choices_text"], partial)

            resp = query_model(model, prompt)
            if dataset == "gsm8k":
                ans = extract_answer_gsm8k(resp)
            else:
                ans = extract_answer_arc(resp, s.get("choice_labels"))

            trunc_res.append({
                "step_k": k, "n_total_steps": ns,
                "partial_reasoning": partial,
                "prompt": prompt, "response": resp, "answer": ans,
                "matches_full": (ans.strip() == full_ans.strip()) if ans and full_ans else False,
            })

        all_out.append({"id": s["id"], "n_steps": ns, "steps": steps,
                        "full_answer": full_ans,
                        "truncation_results": trunc_res, "skipped": False})

    scr_steps = _agg_scr(all_out, top_steps)
    summary = {"model": model, "dataset": dataset,
               "max_steps_seen": top_steps,
               "scr_by_step": scr_steps, "results": all_out}
    _dump(model, dataset, summary)
    return summary


def _agg_scr(all_out, max_k):
    rows = []
    for k in range(1, max_k + 1):
        t_ans, f_ans = [], []
        for item in all_out:
            if item.get("skipped"):
                continue
            for tr in item["truncation_results"]:
                if tr["step_k"] == k:
                    t_ans.append(tr["answer"])
                    f_ans.append(item["full_answer"])
                    break
        if t_ans:
            row = compute_scr(t_ans, f_ans)
            row["step_k"] = k
            row["n_samples_with_this_step"] = len(t_ans)
            rows.append(row)
    return rows


def _dump(model, dataset, blob):
    tag = model.replace(":", "_").replace("/", "_")
    d = os.path.join(config.RESULTS_DIR, "truncation")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "%s_%s.json" % (tag, dataset))
    with open(p, "w") as f:
        json.dump(blob, f, indent=2)
    print("  Saved truncation ->", p)
