"""Client side of the gateway: stream a completion, reconnect if the connection drops.

stream_via_gateway() records every event the client receives, with the
gateway's position fields, so that delivery can be checked afterwards
(gateway/check.py). If the connection to the gateway breaks before [DONE],
it reconnects to GET /v1/streams/<req_id>?from_seq=<next position> and
continues; this exercises the client-resume path.
"""
import json
import time

import requests


def _events(resp):
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
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
    headers = {"X-Request-Id": req_id} if req_id else {}
    conn, next_seq, dropped = 0, 0, False
    while True:
        try:
            if conn == 0:
                r = requests.post(base + "/v1/completions", json={**body, "stream": True},
                                  headers=headers, stream=True, timeout=timeout_s)
            else:
                r = requests.get(f"{base}/v1/streams/{req_id}", params={"from_seq": next_seq},
                                 stream=True, timeout=timeout_s)
            with r:
                if r.status_code != 200:
                    out["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
                    break
                req_id = req_id or r.headers.get("X-Gateway-Request-Id")
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
            if req_id is None or conn >= max_reconnects:
                break
            conn += 1
            out["reconnects"] = conn
            time.sleep(0.2)
    out["req_id"] = req_id
    out["events"] = events
    return out
