"""Streaming client for the OpenAI-compatible /v1/completions endpoint.

Each server-sent event with text is recorded as one "chunk". vLLM and
SGLang usually send one token per chunk, but this is not guaranteed,
so analysis works on text, and chunk counts are approximate token counts.
"""
import json
import time

import requests


def stream_completion(base_url, model, prompt, max_tokens, temperature, seed,
                      on_chunk=None, timeout_s=600):
    """Stream a completion. Returns (chunks, error).

    chunks: list of dicts {i, text, t_wall, t_mono}
    error:  None if the stream finished cleanly, else a string describing
            how it ended (e.g. connection dropped because the server died).
    on_chunk(i, text) may return True to request that the caller's fault
    action run; the stream keeps being read afterwards so that any tokens
    already in flight are still recorded.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    if seed is not None:
        body["seed"] = seed
    chunks = []
    try:
        with requests.post(f"{base_url.rstrip('/')}/v1/completions", json=body,
                           stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[len("data:"):].strip()
                if data == "[DONE]":
                    return chunks, None
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                if text == "":
                    continue
                chunk = {"i": len(chunks), "text": text,
                         "t_wall": time.time(), "t_mono": time.monotonic()}
                chunks.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk["i"], text)
        return chunks, None
    except Exception as e:  # stream dies when the server is killed
        return chunks, f"{type(e).__name__}: {e}"


def join(chunks):
    return "".join(c["text"] for c in chunks)
