"""Tests for the exactly-once gateway, against the mock engine (no GPU needed).

    python -m unittest discover -s tests -v

Engines and the gateway run as real processes and are SIGKILLed mid-stream,
as in the measured experiments. The mock engine is not a model; these tests
check the gateway's delivery logic, not engine behaviour.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gateway.check import delivery, first_divergence, raw_text, text_of  # noqa: E402
from gateway.client import stream_via_gateway  # noqa: E402
from gateway.tokenlog import TokenLog, committed, read_log  # noqa: E402

PROMPT = "Explain how a write-ahead log lets a database recover after a crash."
MAX_TOKENS = 60


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_http(url, timeout=15):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.05)
    raise TimeoutError(url)


class Proc:
    """A child process that can be SIGKILLed and started again on the same port."""

    def __init__(self, args, health_url):
        self.args, self.health_url, self.p = args, health_url, None

    def start(self):
        self.p = subprocess.Popen([sys.executable, "-m"] + self.args, cwd=ROOT,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True)
        wait_http(self.health_url)
        return self

    def kill(self):
        if self.p and self.p.poll() is None:
            os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
            self.p.wait()


def engine(port, *extra):
    return Proc(["harness.mock_engine", "--port", str(port), "--delay", "0.01", *extra],
                f"http://127.0.0.1:{port}/v1/models")


def gateway(port, backends, log_dir, *extra):
    return Proc(["gateway.server", "--backends", ",".join(backends), "--model", "mock",
                 "--port", str(port), "--log-dir", log_dir, "--health-poll", "0.05",
                 "--stall-timeout", "5", "--recovery-timeout", "20", *extra],
                f"http://127.0.0.1:{port}/health")


def direct(base, temperature=0.0, seed=None, max_tokens=MAX_TOKENS):
    body = {"model": "mock", "prompt": PROMPT, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False}
    if seed is not None:
        body["seed"] = seed
    text = []
    body["stream"] = True
    with requests.post(base + "/v1/completions", json=body, stream=True, timeout=30) as r:
        for raw in r.iter_lines(decode_unicode=True):
            if raw and raw.startswith("data:") and raw[5:].strip() != "[DONE]":
                text.append(json.loads(raw[5:])["choices"][0]["text"])
    return "".join(text)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.procs = []
        self.tmp = tempfile.mkdtemp(prefix="gwtest_")

    def tearDown(self):
        for p in self.procs:
            p.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def track(self, proc):
        self.procs.append(proc)
        return proc

    def setup(self, n_engines=1, engine_args=(), gw_args=()):
        ports = [free_port() for _ in range(n_engines)]
        engines = [self.track(engine(p, *engine_args).start()) for p in ports]
        gport = free_port()
        urls = [f"http://127.0.0.1:{p}" for p in ports]
        gw = self.track(gateway(gport, urls, os.path.join(self.tmp, "log"), *gw_args).start())
        return engines, gw, f"http://127.0.0.1:{gport}", urls

    def body(self, **kw):
        return {"model": "mock", "prompt": PROMPT, "max_tokens": MAX_TOKENS, **kw}

    def kill_at(self, k, victim, restart_after=None):
        """on_event hook: SIGKILL `victim(event)` once position k is delivered."""
        state = {}

        def hook(e):
            if not state and e["seq"] + e["n"] >= k:
                proc = victim(e)
                state["t_kill"] = time.time()
                proc.kill()
                if restart_after is not None:
                    threading.Timer(restart_after, proc.start).start()
        return hook, state

    def assert_exactly_once(self, res, positions=MAX_TOKENS):
        d = delivery(res["events"])
        self.assertTrue(d["exactly_once"], d)
        self.assertEqual(d["positions"], positions)
        self.assertEqual(res["finish_reason"], "length")
        return d

    # ------------------------------------------------------------------
    def test_failure_free_matches_engine(self):
        _, _, gw, urls = self.setup()
        ref = direct(urls[0])
        res = stream_via_gateway(gw, self.body())
        self.assert_exactly_once(res)
        self.assertEqual(text_of(res["events"]), ref)
        self.assertEqual(res["final"]["attempts"], 0)

    def test_engine_killed_and_restarted(self):
        engines, _, gw, urls = self.setup()
        ref = direct(urls[0])
        hook, st = self.kill_at(20, lambda e: engines[0], restart_after=0.5)
        res = stream_via_gateway(gw, self.body(), on_event=hook)
        self.assert_exactly_once(res)
        self.assertEqual(res["final"]["attempts"], 1)
        # deterministic engine: the recovered output is the failure-free output
        self.assertEqual(text_of(res["events"]), ref)
        after = [e for e in res["events"] if e["attempt"] == 1]
        self.assertGreater(after[0]["t_wall"] - st["t_kill"], 0.4)   # waited for restart

    def test_failover_to_second_replica(self):
        engines, _, gw, urls = self.setup(n_engines=2)
        ref = direct(urls[0])
        hook, st = self.kill_at(20, lambda e: engines[e["backend"]])   # no restart
        res = stream_via_gateway(gw, self.body(), on_event=hook)
        self.assert_exactly_once(res)
        self.assertEqual(text_of(res["events"]), ref)
        backends = {e["attempt"]: e["backend"] for e in res["events"]}
        self.assertNotEqual(backends[0], backends[1])
        after = [e for e in res["events"] if e["attempt"] == 1]
        self.assertLess(after[0]["t_wall"] - st["t_kill"], 2.0)

    def test_history_dependent_engine_never_contradicts(self):
        engines, _, gw, urls = self.setup(engine_args=("--history-dependent",))
        hook, _ = self.kill_at(20, lambda e: engines[0], restart_after=0.3)
        res = stream_via_gateway(gw, self.body(), on_event=hook)
        self.assert_exactly_once(res)
        pre = text_of([e for e in res["events"] if e["attempt"] == 0])
        self.assertTrue(text_of(res["events"]).startswith(pre))

    def test_sampling_temperature(self):
        engines, _, gw, _ = self.setup()
        hook, _ = self.kill_at(30, lambda e: engines[0], restart_after=0.3)
        res = stream_via_gateway(gw, self.body(temperature=0.8, seed=1234), on_event=hook)
        self.assert_exactly_once(res)

    def test_text_mode(self):
        engines, _, gw, urls = self.setup(gw_args=("--mode", "text"))
        ref = direct(urls[0])
        hook, _ = self.kill_at(25, lambda e: engines[0], restart_after=0.3)
        res = stream_via_gateway(gw, self.body(), on_event=hook)
        self.assert_exactly_once(res)
        self.assertEqual(text_of(res["events"]), ref)

    def test_two_failures_in_one_request(self):
        engines, _, gw, urls = self.setup()
        ref = direct(urls[0])
        kills = []

        def hook(e):
            end = e["seq"] + e["n"]
            if (end >= 15 and not kills) or (end >= 40 and len(kills) == 1):
                kills.append(end)
                engines[0].kill()
                threading.Timer(0.3, engines[0].start).start()
        res = stream_via_gateway(gw, self.body(), on_event=hook)
        self.assert_exactly_once(res)
        self.assertEqual(res["final"]["attempts"], 2)
        self.assertEqual(text_of(res["events"]), ref)

    def test_client_reconnects(self):
        _, _, gw, urls = self.setup()
        ref = direct(urls[0])
        res = stream_via_gateway(gw, self.body(), drop_after=10)
        self.assertEqual(res["reconnects"], 1)
        self.assert_exactly_once(res)
        self.assertEqual(text_of(res["events"]), ref)

    def test_gateway_killed_and_restarted(self):
        _, gwp, gw, urls = self.setup()
        ref = direct(urls[0])
        hook, _ = self.kill_at(15, lambda e: gwp, restart_after=0.3)
        res = stream_via_gateway(gw, self.body(), on_event=hook, max_reconnects=20)
        self.assertGreaterEqual(res["reconnects"], 1)
        self.assert_exactly_once(res)
        self.assertEqual(text_of(res["events"]), ref)

    def test_negative_control_naive_retry_without_gateway(self):
        """The checker must flag what Phase 1 measured: a retried stream repeats positions."""
        eng = self.track(engine(free_port(), "--history-dependent").start())
        base = eng.health_url.rsplit("/v1/", 1)[0]
        events, attempt = [], 0
        for attempt in range(2):
            body = {"model": "mock", "prompt": PROMPT, "max_tokens": MAX_TOKENS, "stream": True}
            try:
                with requests.post(base + "/v1/completions", json=body, stream=True,
                                   timeout=10) as r:
                    for i, raw in enumerate(x for x in r.iter_lines(decode_unicode=True)
                                            if x and x.startswith("data:")):
                        if raw[5:].strip() == "[DONE]":
                            break
                        events.append({"seq": i, "n": 1, "attempt": attempt,
                                       "text": json.loads(raw[5:])["choices"][0]["text"]})
                        if attempt == 0 and i + 1 == 20:
                            eng.kill()
            except requests.RequestException:
                pass
            if attempt == 0:
                eng.kill()      # no-op if already dead
                eng.start()     # naive retry: same prompt on the restarted engine
        self.assertEqual({e["attempt"] for e in events}, {0, 1})
        d = delivery(events)
        self.assertFalse(d["exactly_once"])
        self.assertGreaterEqual(d["duplicated_positions"], 19)
        shown = raw_text(events)
        self.assertIsNotNone(first_divergence(shown, text_of(events)))

    # Phase 5 Option B found a client losing an event whose text held non-ASCII
    # characters: requests decoded the stream as Latin-1 and split a JSON line on
    # U+0085. The gateway now sends ASCII-only JSON with charset=utf-8, and
    # gateway.client splits raw bytes.

    def test_non_ascii_text_survives_failover(self):
        engines, gw, gw_url, _ = self.setup(2, engine_args=("--tricky-text",))
        hook, st = self.kill_at(20, lambda e: engines[e["backend"]])
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assert_exactly_once(res)
        text = text_of(res["events"])
        self.assertIn("\u0445", text)
        self.assertIn("\u2028", text)

    def test_naive_latin1_client_sees_every_event(self):
        engines, gw, gw_url, _ = self.setup(1, engine_args=("--tricky-text",))
        seqs = []
        with requests.post(gw_url + "/v1/completions", json={**self.body(), "stream": True},
                           stream=True, timeout=30) as r:
            r.encoding = "latin-1"                    # what requests assumed before the fix
            for raw in r.iter_lines(decode_unicode=True):
                if raw and raw.startswith("data:") and raw[5:].strip() != "[DONE]":
                    g = json.loads(raw[5:]).get("gateway") or {}
                    if "seq" in g:
                        seqs.append(g["seq"])
        self.assertEqual(seqs, list(range(MAX_TOKENS)))


    # ------------------------------------------------------------------
    # Phase 5 Option B also found vLLM, resumed from a token-id prompt that ended
    # in part of a UTF-8 character, returning the whole decoded prompt (and the
    # delivered text) as its first token's text. Positions stayed exactly-once,
    # but the text a client concatenates did not. The mock's --resume-echo
    # imitates that; the gateway now derives resumed text from the token ids.

    @staticmethod
    def _resumed_after(res):
        """Id of the last token delivered before the first resumed entry."""
        ev = res["events"]
        i = next(k for k, e in enumerate(ev) if e["attempt"])
        return ev[i - 1]["ids"][-1]

    def _kill_on_pending(self, engines, min_seq=10):
        state = {}

        def hook(e):
            ids = e.get("ids") or []
            if not state and e["seq"] >= min_seq and ids and ids[-1] % 7 == 3:
                state["t_kill"] = time.time()
                engines[e["backend"]].kill()
        return hook, state

    def test_client_visible_text_after_resume_equals_failure_free(self):
        engines, gw, gw_url, urls = self.setup(2, engine_args=("--resume-echo", "--delay", "0.05"))
        expected = direct(urls[1])
        hook, st = self._kill_on_pending(engines)
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assertEqual(self._resumed_after(res) % 7, 3)      # the kill hit a pending token
        self.assert_exactly_once(res)
        self.assertEqual(raw_text(res["events"]), expected)      # what a naive client shows
        self.assertEqual(text_of(res["events"]), expected)

    def test_client_token_id_prompt_ending_mid_character(self):
        # no failure: the echo can also happen on the first attempt when the client
        # sends a token-id prompt that ends in a pending token
        from harness.mock_engine import detokenize
        engines, gw, gw_url, urls = self.setup(1, engine_args=("--resume-echo", "--delay", "0.05"))
        prompt = [500, 11, 12, 13, 17]               # 17 % 7 == 3: pending
        res = stream_via_gateway(gw_url, {**self.body(), "prompt": prompt})
        self.assert_exactly_once(res)
        ids = [t for e in res["events"] for t in e["ids"]]
        self.assertEqual(raw_text(res["events"]).rstrip("\ufffd"), detokenize(ids, True).rstrip("\ufffd")[:len(raw_text(res["events"]).rstrip("\ufffd"))])
        self.assertLess(len(raw_text(res["events"])), len(detokenize(ids, True)) + 16)

    def test_special_tokens_are_not_shown_after_resume(self):
        # /detokenize shows special tokens (<|eot_id|>), the engine's stream does not;
        # the gateway must not release them (Phase 5B rerun, 32 concurrent requests)
        engines, gw, gw_url, urls = self.setup(2, engine_args=("--resume-echo", "--special-tokens", "--delay", "0.05"))
        expected = direct(urls[1])
        hook, st = self._kill_on_pending(engines, min_seq=28)   # after a special token
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assertEqual(self._resumed_after(res) % 7, 3)
        self.assert_exactly_once(res)
        self.assertNotIn("<|", raw_text(res["events"]))
        self.assertEqual(raw_text(res["events"]), expected)

    def test_failover_from_chunked_engine(self):
        # vLLM streams with chunked transfer encoding; a kill cuts the stream
        # mid-chunk, which urllib3 reports with its own exception (Phase 5B rerun:
        # requests stopped at the kill until the reader wrapped it)
        engines, gw, gw_url, urls = self.setup(2, engine_args=("--chunked", "--delay", "0.03"))
        expected = direct(urls[1])
        hook, st = self.kill_at(20, lambda e: engines[e["backend"]])
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assert_exactly_once(res)
        self.assertEqual(raw_text(res["events"]), expected)
        self.assertTrue(any(e["attempt"] for e in res["events"]))

    def test_client_visible_text_chunked_engine(self):
        engines, gw, gw_url, urls = self.setup(2, engine_args=("--chunked", "--resume-echo",
                                                               "--special-tokens", "--delay", "0.05"))
        expected = direct(urls[1])
        hook, st = self._kill_on_pending(engines, min_seq=28)
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assert_exactly_once(res)
        self.assertEqual(raw_text(res["events"]), expected)

    def test_burst_of_clients_all_delivered(self):
        # 64 clients at once: none may be lost (Phase 5B: the default listen backlog
        # reset connections, and those requests never started)
        engines, gw, gw_url, _ = self.setup(1)
        results = [None] * 64

        def one(j):
            results[j] = stream_via_gateway(gw_url, self.body(max_tokens=20))
        ts = [threading.Thread(target=one, args=(j,)) for j in range(64)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for res in results:
            self.assert_exactly_once(res, positions=20)

    def test_negative_control_engine_text_after_resume(self):
        engines, gw, gw_url, urls = self.setup(2, engine_args=("--resume-echo", "--delay", "0.05"),
                                               gw_args=("--engine-text-after-resume",))
        expected = direct(urls[1])
        hook, st = self._kill_on_pending(engines)
        res = stream_via_gateway(gw_url, self.body(), on_event=hook)
        self.assertIn("t_kill", st)
        self.assertEqual(self._resumed_after(res) % 7, 3)      # the kill hit a pending token
        self.assert_exactly_once(res)                             # positions still fine
        self.assertNotEqual(raw_text(res["events"]), expected)    # text is not
        self.assertGreater(len(raw_text(res["events"])), len(expected))


class UnitTests(unittest.TestCase):
    def test_delivery_counts(self):
        ev = [{"seq": 0, "n": 2}, {"seq": 2, "n": 1}, {"seq": 2, "n": 1}, {"seq": 5, "n": 1}]
        d = delivery(ev)
        self.assertEqual(d["duplicated_positions"], 1)
        self.assertEqual(d["missing_positions"], 2)
        self.assertFalse(d["exactly_once"])
        self.assertTrue(delivery([{"seq": 0, "n": 3}, {"seq": 3, "n": 1}])["exactly_once"])

    def test_torn_log_line_is_ignored(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, "r.jsonl")
            log = TokenLog(path)
            log.append({"kind": "tokens", "seq": 0, "n": 1, "text": " a"})
            log.close()
            with open(path, "a") as f:
                f.write('{"kind": "tokens", "seq": 1, "n"')
            toks, done = committed(read_log(path))
            self.assertEqual(len(toks), 1)
            self.assertIsNone(done)
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
