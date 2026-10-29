"""A tiny OpenAI-compatible mock engine for testing the harness and gateway.

It is NOT a model and is never used for measured results. It streams one token
per chunk, like vLLM and SGLang, so that tests can kill it mid-stream:

  * tokens are words "w0".."w499"; the token after a context is a hash of the
    last three token ids, so output is a pure function of the context
    (like an engine in deterministic mode);
  * --history-dependent adds the number of requests served so far to the hash,
    so a cold restart produces different text (like vLLM's default mode with
    prefix caching, Phase 1);
  * temperature > 0 samples from a seeded RNG that restarts with every request
    (like an engine that does not expose its RNG position);
  * /tokenize and the vLLM token-id logprobs format are supported;
  * --tricky-text makes every 5th token's text non-ASCII, including U+2028 and a
    character whose UTF-8 contains byte 0x85 (tests SSE framing; Phase 5B);
  * --resume-echo imitates two things vLLM 0.31 did in Phase 5B: a "pending" token
    (id % 7 == 3, like a byte that is only part of a UTF-8 character) streams
    empty text and its text arrives with the next token, and /detokenize of ids
    ending in one decodes it as U+FFFD; and a request whose token-id prompt ends
    in a pending token returns the whole decoded prompt as its first token's text;
  * --special-tokens makes ids with id % 50 == 7 special: streamed text omits them
    (like skip_special_tokens) while /detokenize shows them as "<|s7|>", as vLLM's
    /detokenize shows Llama 3's <|eot_id|>;
  * --chunked streams with HTTP/1.1 chunked transfer encoding, as vLLM and SGLang
    do (a kill then cuts the stream mid-chunk);
  * /detokenize is supported.

  python -m harness.mock_engine --port 8100 --delay 0.01
"""
import argparse
import json
import random
import re
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VOCAB = 500
BOS = VOCAB          # id of the beginning-of-sequence token
WORD = re.compile(r"^w(\d+)$")


def tokenize(text):
    ids = [BOS]
    for w in text.split():
        m = WORD.match(w)
        ids.append(int(m.group(1)) % VOCAB if m else zlib.crc32(w.encode()) % VOCAB)
    return ids


SPECIAL = {"on": False}


def special(tok):
    return SPECIAL["on"] and tok % 50 == 7


def surface(tok):
    return "" if tok == BOS or special(tok) else f" w{tok}"


def shown(tok):
    """What /detokenize shows for a token: special tokens appear."""
    return f"<|s{tok}|>" if special(tok) else surface(tok)


def pending(tok):
    return tok % 7 == 3 and not special(tok)


def detokenize(ids, echo):
    # like bytes of an unfinished UTF-8 character: every trailing pending token
    # decodes to U+FFFD until a non-pending token completes it
    k = len(ids)
    while echo and k > 0 and pending(ids[k - 1]):
        k -= 1
    return "".join(shown(t) for t in ids[:k]) + "\ufffd" * (len(ids) - k)


def next_token(ctx, salt, rng=None):
    a, b, c = ([BOS, BOS, BOS] + ctx)[-3:]
    base = zlib.crc32(f"{a},{b},{c},{salt}".encode()) % VOCAB
    if rng is None:
        return base
    return (base + rng.randrange(5)) % VOCAB      # "sampling" among nearby ids


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                return self._json(200, {"object": "list", "data": [{"id": state["model"]}]})
            self._json(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
            if self.path == "/tokenize":
                return self._json(200, {"tokens": tokenize(body.get("prompt", ""))})
            if self.path == "/detokenize":
                return self._json(200, {"prompt": detokenize([int(t) for t in body.get("tokens", [])],
                                                             state["echo"])})
            if self.path != "/v1/completions":
                return self._json(404, {"error": "not found"})
            p = body.get("prompt", "")
            ctx = [int(t) for t in p] if isinstance(p, list) else tokenize(p)
            temp = float(body.get("temperature", 0) or 0)
            rng = random.Random(body.get("seed")) if temp > 0 else None
            salt = state["served"] if state["history"] else 0
            state["served"] += 1
            as_ids = bool(body.get("return_tokens_as_token_ids"))
            n = int(body.get("max_tokens", 16))
            chunked = state["chunked"]
            if chunked:
                self.protocol_version = "HTTP/1.1"
                self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def out(b):
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" if chunked else b)
                self.wfile.flush()
            echo = state["echo"] and isinstance(p, list) and bool(ctx) and pending(ctx[-1])
            held = ""
            for i in range(n):
                tok = next_token(ctx, salt, rng)
                ctx.append(tok)
                text = surface(tok)
                if state["tricky"] and tok % 5 == 0:
                    text = f" \u0445\u2028w{tok}\u2026"     # Cyrillic kha (D1 85), LS, ellipsis
                if state["echo"]:
                    if echo:                               # vLLM: decoded prompt + new token
                        text, echo = "".join(surface(t) for t in ctx[:-1]) + text, False
                    elif pending(tok) and i < n - 1:
                        held, text = held + text, ""
                    else:
                        text, held = held + text, ""
                ch = {"index": 0, "text": text,
                      "finish_reason": "length" if i == n - 1 else None,
                      "logprobs": {"tokens": [f"token_id:{tok}"]} if as_ids else None}
                out(("data: " + json.dumps({"choices": [ch]}, ensure_ascii=False) + "\n\n").encode())
                time.sleep(state["delay"])
            out(b"data: [DONE]\n\n")
            if chunked:
                self.wfile.write(b"0\r\n\r\n")

    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--delay", type=float, default=0.01, help="seconds per token")
    p.add_argument("--history-dependent", action="store_true")
    p.add_argument("--model", default="mock")
    p.add_argument("--tricky-text", action="store_true")
    p.add_argument("--resume-echo", action="store_true")
    p.add_argument("--special-tokens", action="store_true")
    p.add_argument("--chunked", action="store_true")
    a = p.parse_args()
    SPECIAL["on"] = a.special_tokens
    state = {"model": a.model, "delay": a.delay, "history": a.history_dependent, "served": 0,
             "tricky": a.tricky_text, "echo": a.resume_echo,
             "chunked": a.chunked}
    ThreadingHTTPServer.request_queue_size = 1024     # many concurrent clients in tests
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(state))
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
