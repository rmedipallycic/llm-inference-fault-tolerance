"""Exactly-once token gateway for OpenAI-compatible streaming completions.

The gateway sits between a streaming client and one or more unmodified engine
replicas (vLLM, SGLang). For every request it:

  1. numbers the generated output: each streamed entry carries `seq`, the
     position of its first token, and `n`, how many tokens it holds;
  2. appends each entry to a durable per-request log *before* releasing it to
     the client (output commit, see gateway/tokenlog.py);
  3. if the engine dies or stalls mid-stream, resumes generation on a healthy
     replica (a restarted one, or another one) by resubmitting the prompt plus
     every token already committed, asking only for the remaining tokens;
  4. lets a client that lost its connection reconnect and receive the entries
     from a position it names: GET /v1/streams/<req_id>?from_seq=N. The client
     keeps N (one past the last position it consumed); the gateway does not
     know what a client received, only what it committed.

What this guarantees, and what it does not (failure model: fail-stop crashes
of engines and the gateway; the log directory survives; a reconnecting client
remembers its last consumed position):
  * Every output position reaches the client at most once and, unless the
    request fails for good, at least once; positions arrive in order. A
    position is a token in ids mode and a chunk in text mode. This is what the
    gateway adds to an unmodified engine.
  * In ids mode the text of every entry after a failure, and from the start
    when the client itself sent token ids, is derived from the token ids
    (engine /detokenize of the output ids), not taken from the engine's stream,
    until the engine's own text agrees again: an engine given a token-id prompt
    that ends inside a UTF-8 character can return the whole decoded prompt (and
    the delivered text) as the text of its first token (vLLM 0.31, Phase 5
    Option B). A text prompt tokenized by the gateway ends on a whole character. So the concatenated text a client shows equals the decoding of the
    delivered token ids, not only the positions.
  * The continuation after a failure is generated with the delivered tokens
    as context, so it never contradicts what the client has already shown.
  * It is the same text as a failure-free run only if the engine itself is
    deterministic (Phase 1: vLLM batch-invariant mode, SGLang deterministic
    inference). At temperature > 0 the engines do not expose their RNG
    position, so the resumed part is a fresh sample, not the one the failed
    engine would have drawn.

Resume modes:
  ids   (default) prompt token ids from the engine's /tokenize, and the ids of
        every generated token (vLLM: logprobs + return_tokens_as_token_ids).
        Nothing is re-tokenized. seq counts tokens.
  text  prompt text + delivered text; for engines that cannot return token
        ids. seq counts chunks.

Run:
  python -m gateway.server --backends http://127.0.0.1:8000 \
      --model meta-llama/Llama-3.1-8B-Instruct --port 9000 --log-dir gw_logs
"""
import argparse
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests

from gateway.client import sse_lines
from gateway.tokenlog import TokenLog, committed, read_log

VLLM_IDS_FIELDS = {"logprobs": 1, "return_tokens_as_token_ids": True}
REQ_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def ids_from_logprobs(lp):
    """vLLM's return_tokens_as_token_ids format: tokens like "token_id:123"."""
    if not lp or not isinstance(lp, dict) or not lp.get("tokens"):
        return None
    ids = []
    for t in lp["tokens"]:
        if not (isinstance(t, str) and t.startswith("token_id:")):
            return None
        ids.append(int(t.split(":", 1)[1]))
    return ids


class Generation:
    """One client request: its committed entries and the thread producing them."""

    def __init__(self, gw, req_id, request, log_path, prompt_ids=None, entries=None):
        self.gw = gw
        self.req_id = req_id
        self.request = request
        self.prompt_ids = prompt_ids
        self.ids_from_client = isinstance(request.get("prompt"), list)
        self.entries = list(entries or [])          # committed "tokens" entries
        self.attempt = 0
        self.finish_reason = None
        self.error = None
        self.cond = threading.Condition()
        self.log = TokenLog(log_path, fsync=gw.fsync)
        self._pending_ids, self._pending_text, self._engine_text_ok = [], "", False

    # ---- producer -------------------------------------------------------
    def produced(self):
        """Tokens (ids mode) or chunks (text mode) committed so far."""
        return sum(e["n"] for e in self.entries)

    def delivered_ids(self):
        return [t for e in self.entries for t in (e.get("ids") or [])]

    def delivered_text(self):
        return "".join(e["text"] for e in self.entries)

    def run(self):
        try:
            self._run()
        except Exception as e:  # never leave a client waiting forever
            self._finish("error", f"{type(e).__name__}: {e}")

    def _run(self):
        req, gw = self.request, self.gw
        max_tokens = int(req.get("max_tokens", 16))
        last_backend = None
        while True:
            remaining = max_tokens - self.produced()
            if remaining <= 0:
                return self._finish("length")
            if self.attempt > gw.max_attempts:
                return self._finish("error", f"gave up after {gw.max_attempts} recoveries")
            backend = gw.pick_backend(avoid=last_backend if self.attempt else None)
            if backend is None:
                return self._finish("error", "no healthy backend within recovery timeout")
            if gw.mode == "ids":
                prompt = self.prompt_ids + self.delivered_ids()
            else:
                prompt = req["prompt"] + self.delivered_text()
            body = {k: v for k, v in req.items() if k not in ("prompt", "max_tokens", "stream")}
            body.update({"model": req.get("model", gw.model), "prompt": prompt,
                         "max_tokens": remaining, "stream": True})
            if gw.mode == "ids":
                body.update(gw.ids_fields)
            last_backend = backend
            outcome = self._stream_from(backend, body)
            if outcome is not None:              # engine finished the request
                return self._finish(outcome)
            self.attempt += 1
            self.log.append({"kind": "attempt", "attempt": self.attempt,
                             "reason": self._last_reason, "failed_backend": backend,
                             "produced": self.produced()})

    def _stream_from(self, backend, body):
        """Stream one engine attempt. Returns finish_reason, or None if it broke."""
        self._last_reason = None
        self._pending_ids, self._pending_text, self._engine_text_ok = [], "", False
        url = self.gw.backends[backend] + "/v1/completions"
        try:
            with requests.post(url, json=body, stream=True,
                               timeout=(5, self.gw.stall_timeout)) as r:
                if r.status_code != 200:
                    self._last_reason = f"HTTP {r.status_code}: {r.text[:200]}"
                    return None
                for raw in sse_lines(r):
                    if not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        # a clean end without a finish_reason: the engine stopped early
                        self._accept("", None, backend, final=True)
                        return "stop"
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    ch = (payload.get("choices") or [{}])[0]
                    text = ch.get("text") or ""
                    ids = ids_from_logprobs(ch.get("logprobs"))
                    if self.gw.mode == "ids" and text and ids is None:
                        raise RuntimeError("ids mode, but the engine returned no token ids; "
                                           "use --mode text or pass the right --ids-fields")
                    self._accept(text, ids, backend, final=bool(ch.get("finish_reason")))
                    if ch.get("finish_reason"):
                        return ch["finish_reason"]
            self._last_reason = "stream ended without finish_reason"
        except requests.RequestException as e:
            self._last_reason = f"{type(e).__name__}: {str(e)[:200]}"
        return None

    def _accept(self, text, ids, backend, final):
        """Commit one engine event, deriving its text from token ids after a failure.

        On a resumed attempt (ids mode), and on the first attempt of a request whose
        prompt the client sent as token ids, the text released for new tokens is
        detokenize(delivered ids + new ids) minus the text already released. Tokens
        whose decoding ends in an incomplete character (U+FFFD) are held back, not
        committed, until it completes or the request ends; a held-back token that is
        lost in a further failure was never released and is simply regenerated. Once
        an event's engine text equals the derived text, the engine is trusted again
        for the rest of the attempt. If /detokenize fails or the derived text does
        not extend what was released, the engine's text is used and the entry says so.
        """
        derive = (self.gw.mode == "ids" and self.gw.derive_text and not self._engine_text_ok
                  and (self.attempt >= 1 or self.ids_from_client))
        if not derive:
            if text or ids:
                self._commit(text, ids, backend)
            return
        if ids:
            self._pending_ids += ids
            self._pending_text += text
        if not self._pending_ids:
            return
        released = self.delivered_text()
        try:
            full = self.gw.detokenize(backend, self.delivered_ids() + self._pending_ids)
        except (requests.RequestException, KeyError, ValueError):
            full = None
        if full is None or not full.startswith(released):
            self._commit(self._pending_text, self._pending_ids, backend,
                         text_source="engine (derivation failed)")
            self._pending_ids, self._pending_text, self._engine_text_ok = [], "", True
            return
        delta = full[len(released):]
        if delta.endswith("\ufffd") and not final:
            return                                   # incomplete character: hold back
        same = delta == self._pending_text
        self._commit(delta, self._pending_ids, backend,
                     text_source="engine" if same else "detokenized",
                     engine_text_chars=None if same else len(self._pending_text))
        self._pending_ids, self._pending_text = [], ""
        self._engine_text_ok = same

    def _commit(self, text, ids, backend, **extra):
        n = len(ids) if ids is not None else 1
        entry = {"kind": "tokens", "seq": self.produced(), "n": n, "ids": ids, "text": text,
                 "attempt": self.attempt, "backend": backend}
        entry.update({k: v for k, v in extra.items() if v is not None})
        entry = self.log.append(entry)           # durable before the client sees it
        with self.cond:
            self.entries.append(entry)
            self.cond.notify_all()

    def _finish(self, reason, error=None):
        self.error = error
        self.log.append({"kind": "done", "finish_reason": reason, "error": error,
                         "produced": self.produced(), "attempts": self.attempt})
        self.log.close()
        with self.cond:
            self.finish_reason = reason
            self.cond.notify_all()

    # ---- consumers ------------------------------------------------------
    def follow(self, from_seq=0, poll_s=1.0):
        """Yield committed entries with seq >= from_seq, then wait for more until done."""
        i = 0
        while True:
            with self.cond:
                while i >= len(self.entries) and self.finish_reason is None:
                    self.cond.wait(poll_s)
                batch = self.entries[i:]
                i = len(self.entries)
                done = self.finish_reason is not None and i >= len(self.entries)
            for e in batch:
                if e["seq"] >= from_seq:
                    yield e
            if done:
                return


class Gateway:
    def __init__(self, backends, model, log_dir, mode="ids", ids_fields=None, fsync=False,
                 stall_timeout=60.0, recovery_timeout=900.0, max_attempts=5, health_poll=0.5,
                 derive_text=True):
        self.backends = [b.rstrip("/") for b in backends]
        self.model = model
        self.log_dir = log_dir
        self.mode = mode
        self.ids_fields = VLLM_IDS_FIELDS if ids_fields is None else ids_fields
        self.fsync = fsync
        self.stall_timeout = stall_timeout
        self.recovery_timeout = recovery_timeout
        self.max_attempts = max_attempts
        self.health_poll = health_poll
        self.derive_text = derive_text
        self.gens = {}
        self.lock = threading.Lock()
        os.makedirs(log_dir, exist_ok=True)

    # ---- backends -------------------------------------------------------
    def healthy(self, i):
        try:
            return requests.get(self.backends[i] + "/v1/models", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def pick_backend(self, avoid=None):
        """First healthy backend, trying others before the one that just failed."""
        order = [i for i in range(len(self.backends)) if i != avoid]
        if avoid is not None:
            order.append(avoid)
        deadline = time.monotonic() + self.recovery_timeout
        while time.monotonic() < deadline:
            for i in order:
                if self.healthy(i):
                    return i
            time.sleep(self.health_poll)
        return None

    def tokenize(self, prompt, model):
        b = self.pick_backend()
        if b is None:
            raise RuntimeError("no healthy backend to tokenize the prompt")
        r = requests.post(self.backends[b] + "/tokenize", json={"model": model, "prompt": prompt},
                          timeout=30)
        r.raise_for_status()
        return r.json()["tokens"]

    def detokenize(self, backend, ids):
        r = requests.post(self.backends[backend] + "/detokenize",
                          json={"model": self.model, "tokens": ids}, timeout=30)
        r.raise_for_status()
        return r.json()["prompt"]

    # ---- requests -------------------------------------------------------
    def start(self, request, req_id=None):
        req_id = req_id or uuid.uuid4().hex
        if not REQ_ID_RE.match(req_id):
            raise ValueError("bad request id")
        with self.lock:
            if req_id in self.gens or os.path.exists(self._path(req_id)):
                raise ValueError(f"request id {req_id} already used")
        prompt = request.get("prompt")
        if isinstance(prompt, list):
            prompt_ids = [int(t) for t in prompt]
            if self.mode == "text":
                raise ValueError("text mode needs a text prompt")
        elif self.mode == "ids":
            prompt_ids = self.tokenize(prompt, request.get("model", self.model))
        else:
            prompt_ids = None
        gen = Generation(self, req_id, request, self._path(req_id), prompt_ids)
        gen.log.append({"kind": "header", "req_id": req_id, "request": request,
                        "prompt_ids": prompt_ids, "mode": self.mode,
                        "backends": self.backends})
        with self.lock:
            self.gens[req_id] = gen
        threading.Thread(target=gen.run, daemon=True).start()
        return gen

    def get(self, req_id):
        """A live generation, or one rebuilt from its log (completed or not)."""
        with self.lock:
            if req_id in self.gens:
                return self.gens[req_id]
        if not REQ_ID_RE.match(req_id) or not os.path.exists(self._path(req_id)):
            return None
        return self._recover(req_id)

    def recover_unfinished(self):
        """After a gateway restart: resume every logged request that has no "done" entry."""
        resumed = []
        for name in sorted(os.listdir(self.log_dir)):
            if name.endswith(".jsonl"):
                req_id = name[:-6]
                entries = read_log(self._path(req_id))
                if not any(e.get("kind") == "done" for e in entries):
                    self._recover(req_id)
                    resumed.append(req_id)
        return resumed

    def _recover(self, req_id):
        entries = read_log(self._path(req_id))
        header = next((e for e in entries if e.get("kind") == "header"), None)
        if header is None:
            return None
        toks, done = committed(entries)
        attempts = sum(1 for e in entries if e.get("kind") == "attempt")
        gen = Generation(self, req_id, header["request"], self._path(req_id),
                         header.get("prompt_ids"), toks)
        gen.attempt = attempts
        with self.lock:
            if req_id in self.gens:
                gen.log.close()
                return self.gens[req_id]
            self.gens[req_id] = gen
        if done is not None:
            gen.log.close()
            gen.finish_reason, gen.error = done["finish_reason"], done.get("error")
        else:
            gen.attempt += 1
            gen.log.append({"kind": "attempt", "attempt": gen.attempt,
                            "reason": "gateway restarted", "failed_backend": None,
                            "produced": gen.produced()})
            threading.Thread(target=gen.run, daemon=True).start()
        return gen

    def _path(self, req_id):
        return os.path.join(self.log_dir, f"{req_id}.jsonl")


def sse_chunk(gen, e=None, finish_reason=None):
    choice = {"index": 0, "text": e["text"] if e else "", "logprobs": None,
              "finish_reason": finish_reason}
    out = {"id": gen.req_id, "object": "text_completion",
           "model": gen.request.get("model", gen.gw.model), "choices": [choice]}
    if e is not None:
        out["gateway"] = {"seq": e["seq"], "n": e["n"], "attempt": e["attempt"],
                          "backend": e["backend"]}
        if e.get("ids") is not None:
            out["gateway"]["ids"] = e["ids"]
    else:
        out["gateway"] = {"attempts": gen.attempt, "error": gen.error}
    # ensure_ascii: every non-ASCII character is escaped, so no client can split an
    # event in two however it decodes or splits lines (Phase 5 Option B finding)
    return "data: " + json.dumps(out, ensure_ascii=True) + "\n\n"


def make_handler(gw):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"     # stream until close; no chunked encoding needed

        def log_message(self, fmt, *args):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, gen, from_seq):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Gateway-Request-Id", gen.req_id)
            self.end_headers()
            try:
                for e in gen.follow(from_seq):
                    self.wfile.write(sse_chunk(gen, e).encode())
                    self.wfile.flush()
                self.wfile.write(sse_chunk(gen, finish_reason=gen.finish_reason).encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass   # client left; generation continues and stays resumable

        def do_GET(self):
            u = urlparse(self.path)
            if u.path in ("/health", "/healthz"):
                return self._json(200, {"ok": True})
            if u.path == "/v1/models":
                return self._json(200, {"object": "list", "data": [{"id": gw.model, "object": "model"}]})
            if u.path.startswith("/v1/streams/"):
                gen = gw.get(u.path[len("/v1/streams/"):])
                if gen is None:
                    return self._json(404, {"error": "unknown request id"})
                q = parse_qs(u.query)
                return self._stream(gen, int(q.get("from_seq", ["0"])[0]))
            return self._json(404, {"error": "not found"})

        def do_POST(self):
            if urlparse(self.path).path != "/v1/completions":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(n) or b"{}")
                gen = gw.start(request, self.headers.get("X-Request-Id"))
            except (ValueError, RuntimeError, requests.RequestException) as e:
                return self._json(400, {"error": str(e)})
            if request.get("stream", False):
                return self._stream(gen, 0)
            for _ in gen.follow(0):
                pass
            return self._json(200, {
                "id": gen.req_id, "object": "text_completion", "model": gw.model,
                "choices": [{"index": 0, "text": gen.delivered_text(),
                             "finish_reason": gen.finish_reason}],
                "gateway": {"attempts": gen.attempt, "error": gen.error,
                            "tokens": gen.produced()}})

    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", required=True, help="comma-separated engine base URLs")
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--mode", choices=["ids", "text"], default="ids")
    p.add_argument("--ids-fields", default=None,
                   help="JSON request fields that make the engine return token ids "
                        f"(default, vLLM: {json.dumps(VLLM_IDS_FIELDS)})")
    p.add_argument("--fsync", action="store_true", help="fsync the log before every release")
    p.add_argument("--stall-timeout", type=float, default=60.0,
                   help="seconds without a chunk before an attempt counts as failed")
    p.add_argument("--recovery-timeout", type=float, default=900.0,
                   help="seconds to wait for a healthy backend after a failure")
    p.add_argument("--max-attempts", type=int, default=5)
    p.add_argument("--engine-text-after-resume", action="store_true",
                   help="ids mode: release the engine's own text after a failure instead of "
                        "deriving it from token ids (the behaviour before this option existed)")
    p.add_argument("--health-poll", type=float, default=0.5,
                   help="seconds between health checks while waiting for a backend")
    a = p.parse_args()
    gw = Gateway(a.backends.split(","), a.model, a.log_dir, mode=a.mode,
                 ids_fields=json.loads(a.ids_fields) if a.ids_fields else None,
                 fsync=a.fsync, stall_timeout=a.stall_timeout,
                 recovery_timeout=a.recovery_timeout, max_attempts=a.max_attempts,
                 health_poll=a.health_poll, derive_text=not a.engine_text_after_resume)
    resumed = gw.recover_unfinished()
    if resumed:
        print(f"resuming {len(resumed)} unfinished request(s) from {a.log_dir}", flush=True)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(gw))
    srv.daemon_threads = True
    print(f"gateway on http://{a.host}:{a.port} -> {', '.join(gw.backends)} "
          f"(mode {a.mode}, fsync {a.fsync})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
