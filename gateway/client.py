"""Client side of the gateway: stream a completion, reconnect if the connection drops.

stream_via_gateway() records every event the client receives, with the
gateway's position fields, so that delivery can be checked afterwards
(gateway/check.py). If the connection to the gateway breaks before [DONE],
it reconnects to GET /v1/streams/<req_id>?from_seq=<next position> and
continues; this exercises the client-resume path. The client picks the request
id itself, so a submission that failed before the gateway answered is retried
under the same id (resumed if the gateway had started it, submitted otherwise).
"""
import http.client
import json
import time
import uuid

import requests
import urllib3


def sse_lines(resp):
    """Lines of a server-sent-event stream, split on \\n / \\r only, as they arrive.

    Splitting the raw bytes matters: decoded text would be split by
    str.splitlines() on U+0085, U+2028 and other characters too, and requests
    decodes a text/event-stream without a charset as Latin-1, which turns some
    UTF-8 bytes into U+0085. Either cuts a JSON event in two and loses it
    (found in Phase 5 Option B with random-token prompts).

    Reading with read1 matters too: requests' iter_lines() waits for 512 bytes
    on a stream without chunked transfer encoding (the gateway's), so an event
    was seen only when the next one arrived, one token late (found in the Phase 5B
    rerun; vLLM's chunked stream was not affected).
    """
    raw = resp.raw
    if not hasattr(raw, "read1"):                    # very old urllib3
        for line in resp.iter_lines(chunk_size=1):
            if line:
                yield line.decode("utf-8", errors="replace")
        return
    buf = b""
    while True:
        try:
            data = raw.read1(65536)
        except (urllib3.exceptions.HTTPError, http.client.HTTPException, OSError) as e:
            # read1 raises urllib3's errors directly; iter_lines() wrapped them. A
            # stream cut mid-chunk (a killed vLLM) must look like any broken stream.
            raise requests.exceptions.ChunkedEncodingError(f"stream broken: {e}") from e
        if not data:
            break
        buf += data
        while True:
            i = buf.find(b"\n")
            if i < 0:
                break
            line, buf = buf[:i].rstrip(b"\r"), buf[i + 1:]
            if line:
                yield line.decode("utf-8", errors="replace")
    if buf.strip():
        yield buf.rstrip(b"\r").decode("utf-8", errors="replace")


def _events(resp):
    for raw in sse_lines(resp):
        if not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            yield None
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def stream_via_gateway(base_url, body, req_id=None, on_event=None, max_reconnects=5,
                       timeout_s=1800, drop_after=None):
    """Returns dict(events, finish_reason, final, reconnects, error, req_id).

    events: one dict per text-bearing event: seq, n, text, attempt, backend,
            ids (if any), connection (0 = first, 1.. = reconnects), t_wall, t_mono.
    on_event(event) may return True to have the caller's fault action run.
    drop_after: test hook; close the connection after this many events once,
            to exercise reconnect.
    """
    base = base_url.rstrip("/")
    events, out = [], {"reconnects": 0, "finish_reason": None, "final": None, "error": None}
    # The client chooses the request id, so a submission whose connection failed can
    # be retried safely: if the gateway started it, the stream is resumed; if not,
    # it is submitted again (Phase 5B: connections reset under a burst of clients).
    req_id = req_id or uuid.uuid4().hex
    headers = {"X-Request-Id": req_id}
    conn, next_seq, dropped, accepted = 0, 0, False, False
    while True:
        try:
            if not accepted:
                g = requests.get(f"{base}/v1/streams/{req_id}", params={"from_seq": next_seq},
                                 stream=True, timeout=timeout_s) if conn > 0 else None
                if g is not None and g.status_code == 200:
                    r = g
                else:
                    if g is not None:
                        g.close()
                    r = requests.post(base + "/v1/completions", json={**body, "stream": True},
                                      headers=headers, stream=True, timeout=timeout_s)
            else:
                r = requests.get(f"{base}/v1/streams/{req_id}", params={"from_seq": next_seq},
                                 stream=True, timeout=timeout_s)
            with r:
                if r.status_code != 200:
                    out["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    break
                accepted = True
                for ev in _events(r):
                    if ev is None:
                        out["req_id"] = req_id
                        out["events"] = events
                        return out
                    g = ev.get("gateway") or {}
                    ch = (ev.get("choices") or [{}])[0]
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
                        out["final"] = g
                    if "seq" not in g:
                        continue
                    e = {"seq": g["seq"], "n": g["n"], "text": ch.get("text", ""),
                         "attempt": g.get("attempt"), "backend": g.get("backend"),
                         "ids": g.get("ids"), "connection": conn,
                         "t_wall": time.time(), "t_mono": time.monotonic()}
                    events.append(e)
                    next_seq = e["seq"] + e["n"]
                    if on_event is not None:
                        on_event(e)
                    if drop_after is not None and not dropped and len(events) >= drop_after:
                        dropped = True
                        raise ConnectionResetError("test hook: dropped connection")
            raise ConnectionResetError("stream ended without [DONE]")
        except (requests.RequestException, ConnectionResetError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
            if conn >= max_reconnects:
                break
            conn += 1
            out["reconnects"] = conn
            time.sleep(0.2)
    out["req_id"] = req_id
    out["events"] = events
    return out
