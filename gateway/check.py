"""Delivery checks on what a client received. Pure functions, no I/O.

A client-received stream is a list of events, each covering output positions
[seq, seq + n). Exactly-once delivery means every position from 0 to the end
is covered exactly once, in order.
"""


def delivery(events):
    """Counts of duplicated, missing and out-of-order positions."""
    seen = {}
    out_of_order = 0
    expect = 0
    for e in events:
        if e["seq"] != expect:
            out_of_order += 1
        for p in range(e["seq"], e["seq"] + e["n"]):
            seen[p] = seen.get(p, 0) + 1
        expect = max(expect, e["seq"] + e["n"])
    total = max(seen) + 1 if seen else 0
    return {
        "positions": total,
        "duplicated_positions": sum(1 for c in seen.values() if c > 1),
        "missing_positions": sum(1 for p in range(total) if p not in seen),
        "out_of_order_events": out_of_order,
        "exactly_once": bool(seen) and out_of_order == 0
                        and all(c == 1 for c in seen.values()) and len(seen) == total,
    }


def text_of(events):
    """The text a client shows: each position once, in order (first copy wins)."""
    covered, parts = set(), []
    for e in sorted(events, key=lambda e: e["seq"]):
        span = range(e["seq"], e["seq"] + e["n"])
        if any(p in covered for p in span):
            continue
        covered.update(span)
        parts.append(e["text"])
    return "".join(parts)


def raw_text(events):
    """Concatenation in arrival order: what a naive client would display."""
    return "".join(e["text"] for e in events)


def first_divergence(a, b):
    """Index of the first character where a and b differ, or None if equal."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n
