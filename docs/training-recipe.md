# Training recipe — Thaki-ASR-KO-1.7B

This document describes how the released checkpoint was produced: data
composition, the audio/SFT contract, hyperparameters, and the evaluation
protocol. The published evaluation **numbers** and the full sealed-evaluation
history (Step 5 NO-GO → Step 5b GO) are on the
[model card](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B); this document
describes the recipe and the code, not the numbers.

## 1. Base model and provenance

- **Base:** [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) — Apache-2.0, 30-language ASR.
- **Relation:** single supervised fine-tune (SFT) of the base. The released
  checkpoint was initialised from the original base weights, fine-tuned once
  (the "R1" run), and validated by an independent sealed evaluation before
  release. No re-training was performed for the release.
- **Pinned training script:** `training/qwen3_asr_sft.py` is the official
  Qwen3-ASR SFT script, byte-pinned (`training/UPSTREAM_PIN.txt` holds the git
  SHA and fetch time). `training/sft_train.py` imports it and patches it
  in-memory without modifying the pinned file.

## 2. Data composition

The R1 run used a Korean mixture (by sample count): **telephony 65% / Kspon 20%
/ general 15%.**

| Component | Share | Source | Scale | Notes |
|---|---|---|---|---|
| Telephony / 상담 | 65% | AI-Hub **상담 음성** (dataSetSn 100, distributed as KtelSpeech) | 133,651 utts / 200.0 h / 224 speakers | 8 kHz mobile-call audio, upsampled to 16 kHz; D60/D61/D62 partitions |
| Kspon | 20% | AI-Hub **한국어 음성** (dataSetSn 123, distributed as KsponSpeech) | 54,993 utts / 80.0 h | — |
| General | 15% | Zeroth-Korean (CC BY 4.0) + a small preserved amount of Emilia-YODAS Korean (CC BY 4.0) | (preserved) | held-out read/conversational Korean |

**This is a composition description, not a data release.** Both AI-Hub
sources (상담 음성 and 한국어 음성) are governed by the AI-Hub terms of use,
which allow the data to be used for training AI models and prohibit providing
the data, or simple edits of it, to unapproved third parties. They are not
redistributed, uploaded, or exported. **No audio, transcripts, or metadata from
any source are in this repository** — only the trained weights (on Hugging
Face) and this recipe. Zeroth-Korean and Emilia-YODAS (Korean subset of
`amphion/Emilia-Dataset`, the CC BY 4.0 YODAS portion only) are CC BY 4.0.

**Source attribution (required by the AI-Hub terms, which extend the
requirement to derivative works):** 이 모델은 한국지능정보사회진흥원(NIA)의 사업결과인 AI-Hub "상담 음성", "한국어 음성" 데이터를 활용해 학습했습니다. *(This model was trained using the AI-Hub datasets "상담 음성" and "한국어 음성", which are results of a project of the National Information Society Agency (NIA), Korea.)*

## 3. Preprocessing contract

Every utterance (training and evaluation) follows one audio contract,
implemented in `data/audio.py`:

1. read as float32
2. downmix to mono (mean over channels) if multi-channel
3. resample to **16 000 Hz** (polyphase, exact-ratio via gcd)
4. write as **16-bit PCM**

The telephony source is 8 kHz mobile-call audio; it is upsampled to 16 kHz
before training.

**SFT text format** (`data/sft.py`): each line is
`{"audio": <abs path>, "text": "language Korean<asr_text><transcript>"}`.

**Eval-contamination leak gate** (`data/sft.py`, run at data-build time):

- **(a) audio sha256 exact match** against a frozen eval set → a hard leak; the
  builder reports it and exits non-zero.
- **(b) normalised transcript overlap** against a frozen eval set → that
  utterance is dropped from training (reported, not fatal). In a read corpus the
  same sentence can be read by a train speaker and an eval speaker; training on
  it would inflate the eval CER, so it is removed instead.

## 4. Hyperparameters

| Knob | Value |
|---|---|
| Initialization | original Qwen3-ASR-1.7B weights |
| Method | SFT |
| Audio encoder | **frozen** (`FREEZE_AUDIO=1` — only the decoder/ASR head trained) |
| Precision | **fp32** (`FULL_FP32=1` — bf16 autocast off, eager attention) |
| Learning rate | 2e-6 |
| Effective batch size | 128 (micro-batch 4 × gradient accumulation 32) |
| Loss | cross-entropy **+ knowledge distillation** (λ = 0.05) on **general samples only**, distilled from the original base model |
| Duration | 1 epoch, single H100 (1 GPU) |
| Optimizer / schedule | the official script's defaults (see `training/README.upstream.md`) |

### Collator fix (the one upstream bug)

`training/sft_train.py` forces right padding (`padding_side="right"`) because the
official collator masks the prompt from the left while the processor pads from
the left; for shorter samples in a batch the mask then shifts by the pad length
and the prompt / `<|AUDIO|>` tokens leak into the labels (train loss above
chance, eval NaN). The fix masks the right side instead. The pinned vendor file
is left untouched.

### Knowledge distillation

`training/kd_patch.py` adds, at label positions only, a KL term between a frozen
teacher and the student. For the release the teacher is the original
Qwen3-ASR-1.7B base for **both** the Korean and non-Korean paths
(`KD_TEACHER_KO` / `KD_TEACHER_ML`), and KD is applied to the **general
(preserved-Korean) samples only** (selected by `KD_FLAG_MARKER` on the audio
path), with weight λ = 0.05. Its purpose is to preserve general-Korean
conversational quality while the telephony/numeric adaptation happens on the
rest of the data.

### Checkpoint selection

checkpoint-1200 was selected by **minimum telephony (상담) DEV CER** among the
run's checkpoints (5.24% candidate vs 6.05% base on the held-out telephony DEV).
No re-selection was performed for the release.

## 5. Evaluation protocol

The model was released from an independent 2-arm sealed evaluation (Step 5b).
The protocol, in brief — the numbers and the full history are on the card:

- **Arms:** Base (original Qwen3-ASR-1.7B) vs Candidate (R1 checkpoint-1200).
- **Decoding:** greedy (temperature 0), 3 reps; CER/RTF = median of 3 reps,
  counts = sum over reps (deterministic greedy, so reps agree).
- **Gates (pre-registered, frozen):** A1 (telephony CER), B1 (general CER),
  C1 (number robustness), C2 (wrong-but-plausible), C3 (silence/runaway/RTF).
  GO = all five pass simultaneously.
- **Sets:** TEST-C (general, 456), TEST-D (telephony, 900), TEST-E (numeric, 250;
  E ⊂ D), all length-windowed to **≤ 20 s** (window frozen before the run).
- **The C1 "981" denominator** = numeric-containing utterances in (TEST-E ∪
  TEST-C) × 3 greedy reps — see the card's section 3 for the exact definition.
- **Integrity:** set hashes and the frozen verdict-code hashes were verified at
  evaluation time.

## 6. Scope limitation

All published numbers are measured on audio **≤ 20 s** per utterance. Performance
on longer audio is uncharacterized; do not extrapolate the GO to long-form audio.
See the card's Limitations.
