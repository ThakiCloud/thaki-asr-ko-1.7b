"""Build the Qwen3-ASR SFT format + run the eval-contamination leak gate.

Source lines are plain jsonl: `{"audio": <path>, "text": <transcript>, ...}`
(see `data/audio.py` for how each audio should be normalised first). This writes
the official SFT format and, against the frozen evaluation set(s), enforces two
contamination checks so a held-out evaluation is not memorised:

  (a) audio sha256 exact match  -> HARD leak: report it and exit non-zero.
  (b) normalised transcript overlap -> that utterance is dropped from training
      (reported, not fatal) — same sentence can be read by a train speaker and
      an eval speaker in a read corpus, and training on it inflates eval CER.

Run from the repo root in module mode so `eval` is importable:

    python -m data.sft --train <src.jsonl ...> --eval <frozen.jsonl ...> --out sft.jsonl

Deterministic, no model in the loop.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

from eval.score import normalize  # shared normalisation (single source of truth)

SFT_TEMPLATE = "language {lang}<asr_text>{text}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: str):
    """Yield source lines with `audio` resolved to an absolute path (`_abs`)."""
    base = Path(path).parent
    for line in open(path):
        r = json.loads(line)
        r["_abs"] = str((base / r["audio"]).resolve())
        yield r


def sft_line(audio_abs: str, text: str, language: str = "Korean") -> dict:
    """The official Qwen3-ASR SFT record."""
    return {"audio": audio_abs, "text": SFT_TEMPLATE.format(lang=language, text=text.strip())}


def build_sft(train_paths: list[str], eval_paths: list[str] | None = None,
              out: str = "sft.jsonl", language: str = "Korean", max_dur: float = 30.0,
              report: str = "") -> dict:
    """Write `out` (SFT jsonl) and return a report dict.

    Exits non-zero (SystemExit) if an exact audio-sha256 leak against an eval set
    is found. Text-overlap utterances are dropped and counted instead.
    """
    eval_paths = eval_paths or []
    eval_sha: set[str] = set()
    eval_txt: set[str] = set()
    for p in eval_paths:
        for r in _load(p):
            eval_sha.add(sha256(Path(r["_abs"])))
            eval_txt.add(normalize(r["text"]))

    n = dropped = text_overlap = 0
    leaks: list = []
    per_src: dict = {}
    with open(out, "w") as f:
        for p in train_paths:
            for r in _load(p):
                if r.get("duration", 0) > max_dur or not r["text"].strip():
                    dropped += 1
                    continue
                if eval_paths:
                    if sha256(Path(r["_abs"])) in eval_sha:
                        leaks.append(r.get("id", r["_abs"]))
                        continue
                    if normalize(r["text"]) in eval_txt:
                        text_overlap += 1
                        continue
                lang = r.get("language_name", language)
                f.write(json.dumps(sft_line(r["_abs"], r["text"], lang), ensure_ascii=False) + "\n")
                n += 1
                per_src[r.get("source", "?")] = per_src.get(r.get("source", "?"), 0) + 1

    report_d = {"written": n, "dropped_len_or_empty": dropped,
                "dropped_text_overlap_with_eval": text_overlap, "leaks": len(leaks),
                "leak_ids": leaks[:20], "per_source": per_src}
    if report:
        json.dump(report_d, open(report, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report_d, ensure_ascii=False))
    if leaks:
        print("LEAK (audio sha256):", leaks[:10])
        raise SystemExit(2)
    print(f"leak check: 0 audio leaks · {text_overlap} text-overlap utterances removed from training")
    return report_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="source jsonl(s)")
    ap.add_argument("--eval", nargs="*", default=[], help="frozen eval jsonl(s) (leak check)")
    ap.add_argument("--out", default="sft.jsonl")
    ap.add_argument("--language", default="Korean")
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--report", default="", help="leak report JSON path")
    a = ap.parse_args()
    build_sft(a.train, a.eval, a.out, a.language, a.max_dur, a.report)


if __name__ == "__main__":
    main()
