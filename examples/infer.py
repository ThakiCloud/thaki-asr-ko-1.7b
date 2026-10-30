"""Reproducible inference example for ThakiCloud/Thaki-ASR-KO-1.7B.

Mirrors the quick-start on the Hugging Face model card. Load the released model
and transcribe one or more (Korean) audio files. The model was released with the
16 kHz mono PCM_16 audio contract (see `data/audio.py`); feed it audio in that
form for results consistent with training.

Requires:  pip install torch transformers qwen-asr  # (qwen3-asr runtime)

Usage:
    python examples/infer.py audio1.wav audio2.wav
"""
from __future__ import annotations
import sys
from pathlib import Path

MODEL = "ThakiCloud/Thaki-ASR-KO-1.7B"


def main(paths: list[str]) -> int:
    import torch
    from qwen_asr import Qwen3ASRModel  # official Qwen3-ASR runtime

    model = Qwen3ASRModel.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=16,
        max_new_tokens=256,
    )

    for p in paths:
        out = model.transcribe(p, language="Korean")
        text = out.get("text", out) if isinstance(out, dict) else str(out)
        print(f"[{Path(p).name}] {text}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    raise SystemExit(main(sys.argv[1:]))
