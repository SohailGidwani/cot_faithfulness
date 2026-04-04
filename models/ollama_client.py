from __future__ import annotations
import time
import ollama
from config import TEMP, MAX_TOKENS, RETRIES


def query_model(model, prompt, max_retries=RETRIES, temperature=TEMP,
                num_predict=MAX_TOKENS):
    for attempt in range(max_retries):
        try:
            resp = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": num_predict},
            )
            return resp["message"]["content"]
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    "query to %s failed after %d tries: %s" % (model, max_retries, exc)
                ) from exc
