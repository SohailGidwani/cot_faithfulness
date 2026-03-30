"""Unified Ollama interface with retry logic."""

from __future__ import annotations

import time
import ollama

from config import OLLAMA_TEMPERATURE, OLLAMA_NUM_PREDICT, OLLAMA_MAX_RETRIES


def query_model(
    model: str,
    prompt: str,
    max_retries: int = OLLAMA_MAX_RETRIES,
    temperature: float = OLLAMA_TEMPERATURE,
    num_predict: int = OLLAMA_NUM_PREDICT,
) -> str:
    """Query an Ollama model with retry logic.

    Returns the model's response text.
    """
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_predict": num_predict},
            )
            return response["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"Failed to query {model} after {max_retries} attempts: {e}"
                ) from e
