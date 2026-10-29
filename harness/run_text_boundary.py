"""Text a client sees when a token-id prompt ends inside a character (Phase 5B follow-up).

In the Phase 5B rerun, 3 of 390 requests showed the whole decoded prompt as the
text of their first token: the client sent the prompt as token ids, the prompt
ended inside a multi-byte character, and vLLM repeated the prompt. Commit
852962e makes the gateway check that text with the prompt as context. This
script provokes the case on purpose, on one machine, without failures:

  * prompts: natural text with multi-byte characters (CJK, emoji, accented),
    tokenized by the engine (/tokenize), then cut at every token position where
    the prefix decodes to an unfinished character (/detokenize ends in U+FFFD);
  * each prompt goes through two gateways on the same engine: one deriving the
    text (the fix) and, as a negative control, one releasing the engine's own
    text (--engine-text-after-resume);
  * every request is recorded like a failover trial (trial_<k>.json, gateway
    log), so analysis/recheck_text.py checks the client text with the tokenizer.

  python -m harness.run_text_boundary --run-type measured \\
    --server-cmd "vllm serve MODEL --port {port} --max-model-len 4096" --model MODEL
Writes experiments/<ts>_textboundary_derive and <ts>_textboundary_engine.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import requests

from gateway.check import delivery, raw_text
from gateway.client import stream_via_gateway
from harness.run_gateway_trial import wait_http
from harness.run_trial import OUT_DIRS, environment
from harness.server import ServerProcess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECIAL = re.compile(r"<\|[A-Za-z0-9_]+\|>")

TEXTS = [
    "请用中文解释数据库的预写日志是如何工作的",
    "東京の天気は今日とても良いです。散歩に行きましょう",
    "데이터베이스 복구는 로그를 사용합니다",
    "Ünïcödé façade naïve café crème brûlée smörgåsbord",
    "Emoji test: 🚀🔥🎉 rockets, fire and parties 🧪🧬",
    "Математика и физика — основа инженерии",
    "मशीन लर्निंग मॉडल का प्रशिक्षण",
    "العربية لغة جميلة وغنية",
]


def post(url, path, body):
    r = requests.post(url + path, json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def boundary_prompts(url, model, limit):
    """Token-id prefixes of TEXTS that end inside a character."""
    out = []
    for text in TEXTS:
        ids = post(url, "/tokenize", {"model": model, "prompt": text})["tokens"]
        for k in range(2, len(ids)):
            dec = post(url, "/detokenize", {"model": model, "tokens": ids[:k]})["prompt"]
            if dec.endswith("�"):
                out.append(ids[:k])
                if len(out) >= limit:
                    return out
    return out


def in_context(url, model, prompt, ids):
    base = post(url, "/detokenize", {"model": model, "tokens": prompt})["prompt"].rstrip("�")
    full = post(url, "/detokenize", {"model": model, "tokens": prompt + ids})["prompt"]
    # /detokenize shows special tokens that the stream omits; removed by their text
    # here, a quick check only (analysis/recheck_text.py identifies them by id)
    return SPECIAL.sub("", full[len(base):] if full.startswith(base) else full)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--server-cmd", required=True, help="engine command, with {port}")
    p.add_argument("--model", required=True)
    p.add_argument("--max-prompts", type=int, default=60)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ready-timeout", type=float, default=900.0)
    a = p.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")
    url = f"http://127.0.0.1:{a.port}"
    first = os.path.join(REPO, OUT_DIRS[a.run_type], f"{ts}_textboundary_engine")
    os.makedirs(first, exist_ok=True)
    server = ServerProcess(a.server_cmd.replace("{port}", str(a.port)), url,
                           os.path.join(first, "server_0.log"))
    gws, logs = [], []
    try:
        server.start()
        server.wait_ready(a.ready_timeout)
        prompts = boundary_prompts(url, a.model, a.max_prompts)
        print(f"{len(prompts)} prompts ending inside a character", flush=True)
        for label, gport, extra in (("derive", 9000, []), ("engine", 9001, ["--engine-text-after-resume"])):
            out = os.path.join(REPO, OUT_DIRS[a.run_type], f"{ts}_textboundary_{label}")
            os.makedirs(out, exist_ok=True)
            cmd = [sys.executable, "-m", "gateway.server", "--backends", url, "--model", a.model,
                   "--port", str(gport), "--log-dir", os.path.join(out, "gateway_log"),
                   "--mode", "ids"] + extra
            log = open(os.path.join(out, "gateway.log"), "a", encoding="utf-8")
            logs.append(log)
            gws.append(subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT))
            wait_http(f"http://127.0.0.1:{gport}/health", 60)
            meta = {"run_id": os.path.basename(out), "run_type": a.run_type, "label": label,
                    "config": {**vars(a), "gateway_cmd": cmd, "concurrency": 1, "prompt_len": None},
                    "environment": env, "texts": TEXTS, "n_prompts": len(prompts)}
            with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            ok = 0
            for k, prompt in enumerate(prompts):
                t0 = time.monotonic()
                res = stream_via_gateway(f"http://127.0.0.1:{gport}",
                                         {"model": a.model, "prompt": prompt, "max_tokens": a.max_tokens,
                                          "temperature": 0.0, "seed": 1234, "ignore_eos": True})
                res["t_start_mono"] = t0
                ev = res["events"]
                ids = [t for e in ev for t in (e.get("ids") or [])]
                ref = in_context(url, a.model, prompt, ids)
                shown = raw_text(ev)
                equal = shown == ref or (ref.endswith("�") and shown.startswith(ref.rstrip("�")))
                ok += equal
                rec = {"trial": k, "prompt_ids": prompt, "client": res, "delivery": delivery(ev),
                       "text_check": {"equal": equal, "shown_chars": len(shown), "decoded_chars": len(ref)}}
                with open(os.path.join(out, f"trial_{k:03d}.json"), "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=1)
            print(f"{label}: client text equals the in-context decoding in {ok} of {len(prompts)}",
                  flush=True)
    finally:
        for g in gws:
            g.kill()
        for log in logs:
            log.close()
        server.stop()


if __name__ == "__main__":
    main()
