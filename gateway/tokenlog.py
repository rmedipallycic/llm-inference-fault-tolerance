"""Durable per-request token log (the gateway's output-commit record).

One JSON-lines file per request:

    {"kind": "header", "req_id": ..., "request": {...}, "prompt_ids": [...] | null, "t": ...}
    {"kind": "tokens", "seq": 0, "ids": [...] | null, "text": "...", "attempt": 0, "backend": 0, "t": ...}
    {"kind": "attempt", "attempt": 1, "reason": "...", "t": ...}
    {"kind": "done", "finish_reason": "length", "t": ...}

A "tokens" entry is appended (and, with fsync=True, forced to disk) before the
gateway releases those tokens to the client. Everything the client has ever
seen is therefore in the log, so a restarted engine can be resumed from it and
a reconnecting client can be served the rest without duplicates.

seq is the position of the entry's first token in the generated output. When
the engine returns token ids, seq counts tokens; otherwise it counts chunks.
"""
import json
import os
import threading
import time


class TokenLog:
    def __init__(self, path, fsync=False):
        self.path = path
        self.fsync = fsync
        self._f = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def append(self, entry):
        entry = {**entry, "t": time.time()}
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock:
            self._f.write(line)
            self._f.flush()
            if self.fsync:
                os.fsync(self._f.fileno())
        return entry

    def close(self):
        with self._lock:
            self._f.close()


def read_log(path):
    """All entries of a log; a torn last line (crash mid-write) is ignored."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break  # only the final line can be torn
    return out


def committed(entries):
    """(token entries in order, done entry or None) from a parsed log."""
    toks = [e for e in entries if e.get("kind") == "tokens"]
    done = next((e for e in entries if e.get("kind") == "done"), None)
    return toks, done
