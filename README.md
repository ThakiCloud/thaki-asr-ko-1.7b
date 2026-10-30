# Thaki-ASR-KO-1.7B — reproducible training & evaluation code

This is the **code** repository for
**[ThakiCloud/Thaki-ASR-KO-1.7B](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B)**,
a Korean speech-recognition (ASR) model fine-tuned from
[Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (Apache-2.0) with an
emphasis on Korean telephony / customer-service (상담) speech and numeric content.

**The model weights live on Hugging Face, not here.** This repository does not
duplicate the weights and does not contain any training audio, transcripts, or
derived data. The Hugging Face model card is the source of truth for the
published evaluation numbers, the full sealed-evaluation history (Step 5 NO-GO
→ Step 5b GO), and the ≤ 20 s scope limit — this README cross-links it and does
not restate those numbers.

> **Model card (numbers + history + scope): https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B**

## What is in this repository

| Path | What it is |
|---|---|
| `eval/` | **Evaluator v2** — the deterministic, CPU-only number-robustness metric (`match_utterance_v2`), the v1 side-by-side metric, shared digit extraction, and the shared CER/normalisation used by the data builder. No model in the loop. |
| `data/` | **Preprocessing** — audio normalisation to the training contract (16 kHz mono PCM_16) and the Qwen3-ASR SFT-format builder with the eval-contamination **leak gate**. |
| `training/` | **Training recipe** — the pinned official Qwen3-ASR SFT script (`qwen3_asr_sft.py`, provenance SHA in `UPSTREAM_PIN.txt`), the thin wrapper that fixes one upstream collator bug and adds training guards (`sft_train.py`), and the optional KD preservation loss (`kd_patch.py`). |
| `examples/infer.py` | **Reproducible inference** — load the released model and transcribe (Korean) audio, mirroring the card's quick-start. |
| `docs/training-recipe.md` | The full recipe: data composition, audio/SFT contract, hyperparameters, and the evaluation protocol. |
| `LICENSE` | Apache-2.0 (the weights and config are Apache-2.0; this fine-tune follows the base model's license). |

## Reproducing

### Inference

```bash
pip install torch transformers qwen-asr
python examples/infer.py path/to/korean_audio.wav
```

The released model follows the 16 kHz mono PCM_16 audio contract
(`data/audio.py`); feed it audio in that form for results consistent with
training.

### Data (preprocessing)

```bash
# normalise your own audio to the training contract
python -m data.audio <in.wav> <out_16k.wav>
# build the SFT format + run the leak gate against a frozen eval set
python -m data.sft --train <src.jsonl ...> --eval <frozen.jsonl ...> --out sft.jsonl
```

### Training

```bash
pip install -U qwen-asr datasets
# from the repo root, on one GPU, audio encoder frozen, fp32
FREEZE_AUDIO=1 FULL_FP32=1 \
python -m training.sft_train \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./sft.jsonl \
  --output_dir ./out \
  --batch_size 4 --grad_acc 32 --lr 2e-6 --epochs 1 \
  --save_steps 100 --save_total_limit 5
```

See `docs/training-recipe.md` for the exact hyperparameters, the
knowledge-distillation settings used for the release, and the checkpoint
selection rule.

### Evaluation

`eval/num_match_v2.py` is the frozen number-robustness gate
(`match_utterance_v2`); each file's `__main__` block runs positive/negative
controls. The full gate set, thresholds, sets, and the "what 981 counts"
explanation are on the model card (section 3 and 6) — the code is intentionally
the frozen metric only; the pre-registered gate structure is described on the
card.

## What is NOT in this repository (by policy)

- **Model weights** — fetch them from the Hugging Face repo.
- **No training audio, transcripts, metadata, or derived data** — the telephony
  source (KtelSpeech, AI Hub 012) is internal-only under AI Hub terms and is not
  redistributed; Zeroth-Korean is CC BY 4.0. The training corpora are described
  (composition, not content) in `docs/training-recipe.md`.
- **No PII and no internal credentials.**
- **No internal cluster configuration or storage paths.**

## License

Code and weights: **Apache-2.0** (see `LICENSE`). The underlying training
corpora are not provided and remain subject to their own, more restrictive
terms; this repository does not grant a license to those corpora.
