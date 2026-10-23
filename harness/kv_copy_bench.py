"""Phase 4 (RQ3), checkpoint side: what does it cost to move a request's KV cache
out of the GPU and back?

This is a raw-bandwidth microbenchmark, not a checkpoint implementation. For each
context length L it allocates a GPU tensor of the size the model's KV cache has
for L tokens (2 x layers x KV heads x head dim x dtype bytes per token, from the
model's config.json) and times:

  d2h         GPU -> pinned host memory (torch copy, synchronized)
  h2d         pinned host memory -> GPU
  disk_write  host memory -> a file on local disk, with fsync
  disk_read   that file -> host memory (after dropping it from the page cache
              when possible, i.e. when run with sudo; otherwise flagged)

These are lower bounds on checkpoint and restore cost: a real checkpoint also
gathers non-contiguous KV blocks and coordinates with the engine. Run it with the
GPU otherwise idle (no inference server running).

  python -m harness.kv_copy_bench --run-type measured \
    --model meta-llama/Llama-3.1-8B-Instruct --lengths 512,1024,2048,4096,8192,16384,32768
"""
import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from harness.run_trial import OUT_DIRS, environment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kv_bytes_per_token(model):
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(model, "config.json"), encoding="utf-8") as f:
        c = json.load(f)
    heads = c["num_attention_heads"]
    kv_heads = c.get("num_key_value_heads", heads)
    head_dim = c.get("head_dim") or c["hidden_size"] // heads
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}[c.get("torch_dtype", "bfloat16")]
    return 2 * c["num_hidden_layers"] * kv_heads * head_dim * dtype_bytes, c


def drop_page_cache():
    try:
        subprocess.run(["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                       check=True, capture_output=True, timeout=60)
        return True
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--scratch", default=os.path.expanduser("~/kvbench.bin"),
                   help="file used for the disk measurements (deleted afterwards)")
    a = p.parse_args()
    import torch

    per_tok, cfg = kv_bytes_per_token(a.model)
    lengths = [int(x) for x in a.lengths.split(",")]
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_kvcopy"
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    meta = {"run_id": run_id, "run_type": a.run_type, "config": vars(a),
            "kv_bytes_per_token": per_tok,
            "model_config": {k: cfg.get(k) for k in ("num_hidden_layers", "num_attention_heads",
                                                     "num_key_value_heads", "hidden_size",
                                                     "head_dim", "torch_dtype")},
            "environment": environment(), "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0)}
    records = []
    try:
        for L in lengths:
            n = per_tok * L
            gpu = torch.empty(n, dtype=torch.uint8, device="cuda")
            gpu.random_(0, 255)
            host = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            host.copy_(gpu); torch.cuda.synchronize()          # warm-up
            for rep in range(a.reps):
                r = {"length": L, "bytes": n, "rep": rep}
                torch.cuda.synchronize(); t = time.perf_counter()
                host.copy_(gpu, non_blocking=True); torch.cuda.synchronize()
                r["d2h_s"] = time.perf_counter() - t
                t = time.perf_counter()
                gpu.copy_(host, non_blocking=True); torch.cuda.synchronize()
                r["h2d_s"] = time.perf_counter() - t
                buf = host.numpy()
                t = time.perf_counter()
                with open(a.scratch, "wb") as f:
                    f.write(memoryview(buf))
                    f.flush(); os.fsync(f.fileno())
                r["disk_write_s"] = time.perf_counter() - t
                r["page_cache_dropped"] = drop_page_cache()
                t = time.perf_counter()
                with open(a.scratch, "rb") as f:
                    f.readinto(memoryview(buf))
                r["disk_read_s"] = time.perf_counter() - t
                records.append(r)
            mb = n / 1e6
            med = lambda k: sorted(x[k] for x in records if x["length"] == L)[a.reps // 2]
            print(f"  L={L} ({mb:.0f} MB): d2h {med('d2h_s'):.3f} s, h2d {med('h2d_s'):.3f} s, "
                  f"disk write {med('disk_write_s'):.2f} s, read {med('disk_read_s'):.2f} s")
            del gpu, host
            torch.cuda.empty_cache()
    finally:
        if os.path.exists(a.scratch):
            os.remove(a.scratch)
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(out, "records.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
