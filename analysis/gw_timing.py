"""Timing of gateway requests from the gateway's own log, not from the client.

Why: until commit 64c39ec the measuring client (gateway/client.py) read the
gateway's event stream with requests' iter_lines(), which waits for 512 bytes on
a stream without chunked transfer encoding. The client therefore saw each event
when the next one arrived, so times measured at the client through the gateway
(time to first token, recovery latency) were too long by up to about one
inter-token interval, and more after a failover, when the first resumed events
came in a burst. Direct requests to vLLM and SGLang (chunked streams) were not
affected, nor was anything that does not go through the gateway.

The gateway log stamps every entry with the wall-clock time at which it was
appended, just before it is written (and fsynced, with --fsync) and released to
the client. On the same machine as the client, that is when the client could
have read it, give or take the write, the optional fsync (a few ms), and a
local socket send. These functions use those stamps.
"""
import json
import os


def entries(run_dir, req_id):
    if not req_id:
        return None
    p = os.path.join(run_dir, "gateway_log", f"{req_id}.jsonl")
    if not os.path.exists(p):
        return None
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tokens(log):
    return [e for e in (log or []) if e.get("kind") == "tokens"]


def recovery_s(log, t_kill):
    """Kill to the gateway's commit of the first entry after the failure."""
    if t_kill is None:
        return None
    first = next((e for e in tokens(log) if (e.get("attempt") or 0) >= 1), None)
    return first["t"] - t_kill if first else None


def client_start_wall(res):
    """Wall-clock time at which the client sent the request (from its own clock)."""
    ev = res.get("events") or []
    if not ev or res.get("t_start_mono") is None:
        return None
    return ev[0]["t_wall"] - (ev[0]["t_mono"] - res["t_start_mono"])


def ttft_s(log, res):
    """Client's request start to the gateway's commit of the first entry."""
    tk, t0 = tokens(log), client_start_wall(res)
    return tk[0]["t"] - t0 if tk and t0 is not None else None


def tpot_s(log):
    """Time per output token between the first and last committed entries."""
    tk = tokens(log)
    n = sum(e["n"] for e in tk)
    return (tk[-1]["t"] - tk[0]["t"]) / (n - tk[0]["n"]) if len(tk) > 1 and n > tk[0]["n"] else None
