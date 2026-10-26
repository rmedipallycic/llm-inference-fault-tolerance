"""Phase 4 (RQ3), step 2: what does it cost to move a request's KV cache to a
standby machine over the network?

Failover (Phase 3) resumes a request on a replica on another machine. If the KV
cache had been copied there, the standby could skip the prefill; this measures
the raw cost of that copy. Like harness/kv_copy_bench.py it is a bandwidth
microbenchmark, not a checkpoint implementation: for each context length L it
moves a buffer of the model's KV size for L tokens

  sender GPU -> sender pinned host memory -> TCP -> receiver pinned host memory
  -> receiver GPU

and times every stage. Run the receiver on the standby first, then the sender:

  machine 2:  python -m harness.kv_net_bench receive --port 8000
  machine 1:  python -m harness.kv_net_bench send --run-type measured \
                --peer 172.31.6.222 --port 8000 --model meta-llama/Llama-3.1-8B-Instruct

The sender runs three parts in order and records a wall-clock time for every
transfer, because EC2 instances of this size have a burst network rate and a
lower sustained one:
  sizes-1    each L in --lengths, --reps times (probably within the burst)
  sustained  the largest L, --sustained times back to back
  sizes-2    the --lengths sweep again (after the sustained part)
--streams N splits each transfer over N parallel TCP connections (EC2 limits a
single flow to less than the instance rate outside a placement group).

--device cpu skips the GPU copies (for testing without a GPU).
Raw records only; analysis/phase4.py summarizes.
"""
import argparse
import json
import os
import socket
import struct
import threading
import time
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- buffers ----------

class Buffers:
    """A pinned host buffer and (optionally) a GPU buffer of the same size."""

    def __init__(self, nbytes, device):
        self.device = device
        if device == "cuda":
            import torch
            self.torch = torch
            self.gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            self.gpu.random_(0, 255)
            self.host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self.view = memoryview(self.host.numpy())
        else:
            self.gpu = None
            self.raw = bytearray(os.urandom(min(nbytes, 1 << 20)) * (nbytes // (1 << 20) + 1))[:nbytes]
            self.view = memoryview(self.raw)

    def d2h(self, n):
        if self.gpu is None:
            return 0.0
        t = time.perf_counter()
        self.host[:n].copy_(self.gpu[:n], non_blocking=True)
        self.torch.cuda.synchronize()
        return time.perf_counter() - t

    def h2d(self, n):
        if self.gpu is None:
            return 0.0
        t = time.perf_counter()
        self.gpu[:n].copy_(self.host[:n], non_blocking=True)
        self.torch.cuda.synchronize()
        return time.perf_counter() - t


def slices(n, k):
    step = -(-n // k)
    return [(i * step, min(n, (i + 1) * step)) for i in range(k) if i * step < n]


def recv_exact(sock, view):
    got = 0
    while got < len(view):
        r = sock.recv_into(view[got:])
        if r == 0:
            raise ConnectionError("peer closed")
        got += r


def send_msg(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(b)) + b)


def recv_msg(sock):
    hdr = bytearray(4)
    recv_exact(sock, memoryview(hdr))
    body = bytearray(struct.unpack("!I", hdr)[0])
    recv_exact(sock, memoryview(body))
    return json.loads(body)


# ---------- receiver ----------

def receive(a):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", a.port))
    srv.listen(64)
    print(f"receiver listening on port {a.port}", flush=True)
    while True:
        ctrl, addr = srv.accept()
        hello = recv_msg(ctrl)
        if hello.get("cmd") == "quit":
            print("quit received", flush=True)
            return
        conns = [ctrl] + [srv.accept()[0] for _ in range(hello["streams"] - 1)]
        for c in conns:
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        bufs = Buffers(hello["max_bytes"], a.device)
        send_msg(ctrl, {"ready": True, "device": a.device, "hostname": socket.gethostname()})
        print(f"session from {addr[0]}: {hello['streams']} streams, up to {hello['max_bytes'] / 1e6:.0f} MB",
              flush=True)
        try:
            while True:
                msg = recv_msg(ctrl)
                if msg.get("cmd") == "end":
                    break
                n = msg["bytes"]
                t0 = time.perf_counter()
                ths = [threading.Thread(target=recv_exact, args=(c, bufs.view[s:e]))
                       for c, (s, e) in zip(conns, slices(n, len(conns)))]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
                recv_s = time.perf_counter() - t0
                h2d_s = bufs.h2d(n)
                send_msg(ctrl, {"recv_s": recv_s, "h2d_s": h2d_s})
        finally:
            for c in conns:
                c.close()
            del bufs


# ---------- sender ----------

def send(a):
    from harness.kv_copy_bench import kv_bytes_per_token
    from harness.run_trial import OUT_DIRS, environment
    per_tok = a.kv_bytes_per_token or kv_bytes_per_token(a.model)[0]
    lengths = [int(x) for x in a.lengths.split(",")]
    max_bytes = per_tok * max(lengths)
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_kvnet_s{a.streams}"
    out = os.path.join(REPO, OUT_DIRS[a.run_type], run_id)
    os.makedirs(out, exist_ok=True)
    env = environment()
    if a.run_type == "measured" and env["git_dirty"]:
        print("WARNING: uncommitted changes; the logged commit will not match the code that ran.")

    conns = []
    for _ in range(a.streams):
        c = socket.create_connection((a.peer, a.port), timeout=600)
        c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conns.append(c)
        if len(conns) == 1:
            send_msg(c, {"cmd": "hello", "streams": a.streams, "max_bytes": max_bytes})
    ctrl = conns[0]
    peer_info = recv_msg(ctrl)
    bufs = Buffers(max_bytes, a.device)
    meta = {"run_id": run_id, "run_type": a.run_type, "config": vars(a), "peer": a.peer,
            "peer_info": peer_info, "kv_bytes_per_token": per_tok, "lengths": lengths,
            "environment": env, "device": a.device}
    if a.device == "cuda":
        import torch
        meta["torch"], meta["gpu"] = torch.__version__, torch.cuda.get_device_name(0)
    records = []
    t_start = time.time()

    def one(part, L, rep):
        n = per_tok * L
        d2h_s = bufs.d2h(n)
        t0 = time.perf_counter()
        send_msg(ctrl, {"bytes": n})
        ths = [threading.Thread(target=c.sendall, args=(bufs.view[s:e],))
               for c, (s, e) in zip(conns, slices(n, len(conns)))]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        ack = recv_msg(ctrl)
        net_s = time.perf_counter() - t0
        r = {"part": part, "length": L, "rep": rep, "bytes": n, "t_wall": time.time(),
             "t_since_start_s": time.time() - t_start, "d2h_s": d2h_s,
             "net_and_remote_h2d_s": net_s, "remote_recv_s": ack["recv_s"],
             "remote_h2d_s": ack["h2d_s"]}
        records.append(r)
        return r

    try:
        bufs.d2h(per_tok * lengths[0])                      # warm-up, not recorded
        for part in ("sizes-1", "sustained", "sizes-2"):
            plan = ([(lengths[-1], i) for i in range(a.sustained)] if part == "sustained"
                    else [(L, i) for L in lengths for i in range(a.reps)])
            for L, rep in plan:
                r = one(part, L, rep)
                if part == "sustained" or rep == a.reps - 1:
                    gbps = r["bytes"] * 8 / r["remote_recv_s"] / 1e9
                    print(f"  {part:9s} L={L:6d} ({r['bytes'] / 1e6:5.0f} MB) rep {rep}: "
                          f"d2h {r['d2h_s']:.3f} s, network {r['remote_recv_s']:.2f} s "
                          f"({gbps:.1f} Gbit/s), remote h2d {r['remote_h2d_s']:.3f} s", flush=True)
        send_msg(ctrl, {"cmd": "end"})
    finally:
        for c in conns:
            c.close()
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
        with open(os.path.join(out, "records.jsonl"), "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="role", required=True)
    r = sub.add_parser("receive")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    s = sub.add_parser("send")
    s.add_argument("--run-type", choices=["smoke", "measured"], required=True)
    s.add_argument("--peer", required=True)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model", required=True)
    s.add_argument("--lengths", default="512,1024,2048,4096,8192,16384,32768")
    s.add_argument("--reps", type=int, default=5)
    s.add_argument("--sustained", type=int, default=15)
    s.add_argument("--streams", type=int, default=1)
    s.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    s.add_argument("--kv-bytes-per-token", type=int, default=None,
                   help="override the size taken from the model's config.json (testing only)")
    a = p.parse_args()
    receive(a) if a.role == "receive" else send(a)


if __name__ == "__main__":
    main()
