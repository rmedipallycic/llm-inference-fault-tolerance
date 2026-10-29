"""Recheck the client-visible text of Phase 5B failover trials with the model's tokenizer.

The harness compared the text a client concatenated with vLLM's /detokenize of the
delivered token ids, after deleting anything shaped like <|...|>: /detokenize
shows special tokens that the stream omits. That rule is wrong when the model
writes such a string as ordinary text (seen: "<|reserved_special_token_173|>"
spelled out from ordinary tokens). Here special tokens are identified by id:
the reference text is tokenizer.decode(ids, skip_special_tokens=True), which is
what the engine's stream shows, decoded with the prompt as context (the text the
output adds to the prompt's decoding).

Run on a machine with the model's tokenizer (Phase 5B: the first machine):
  HF_HUB_OFFLINE=1 python analysis/recheck_text.py --model meta-llama/Llama-3.1-8B-Instruct
Writes experiments/<run>/text_recheck.json for every *_ckptfo_* run; analysis/phase5b.py
uses it for the "text ok" column when present.
"""
import argparse
import glob
import json
import os


def compare(shown, ref):
    if shown == ref:
        return True
    r = ref.rstrip("�")            # unfinished last character
    return r != ref and shown.startswith(r) and len(shown) - len(r) <= 16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--root", default="experiments")
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    for d in sorted(glob.glob(os.path.join(a.root, "*_ckptfo_*"))
                    + glob.glob(os.path.join(a.root, "*_textboundary_*"))):
        out, n_ok, n = {}, 0, 0
        for f in sorted(glob.glob(os.path.join(d, "trial_*.json"))):
            with open(f, encoding="utf-8") as fh:
                t = json.load(fh)
            ev = t["client"]["events"]
            ids = [i for e in ev for i in (e.get("ids") or [])]
            shown = "".join(e["text"] for e in ev)
            prompt = t["client"].get("prompt_ids") or []
            hdr = os.path.join(d, "gateway_log", f"{t['client'].get('req_id')}.jsonl")
            if not prompt and os.path.exists(hdr):
                with open(hdr, encoding="utf-8") as fh:
                    prompt = json.loads(fh.readline()).get("prompt_ids") or []

            def dec(c):
                # the text the output adds to the prompt's decoding
                kw = {"skip_special_tokens": True, "clean_up_tokenization_spaces": c}
                base = tok.decode(prompt, **kw).rstrip("\ufffd") if prompt else ""
                full = tok.decode(prompt + ids, **kw)
                return full[len(base):] if full.startswith(base) else full
            variants = {"plain": dec(False), "cleanup": dec(True)}
            match = [k for k, v in variants.items() if compare(shown, v)]
            ref = variants["plain"]
            first = next((i for i in range(min(len(shown), len(ref))) if shown[i] != ref[i]),
                         None if len(shown) == len(ref) else min(len(shown), len(ref)))
            out[os.path.basename(f)] = {"equal": bool(match), "variants": match,
                                        "shown_chars": len(shown), "ref_chars": len(ref),
                                        "first_difference": None if match else first}
            n += 1
            n_ok += bool(match)
        with open(os.path.join(d, "text_recheck.json"), "w", encoding="utf-8") as fh:
            json.dump({"model": a.model, "records": out}, fh, indent=1)
        print(f"{os.path.basename(d)}: text ok {n_ok}/{n}")


if __name__ == "__main__":
    main()
