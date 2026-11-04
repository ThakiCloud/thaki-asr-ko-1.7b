# Thaki-ASR-KO-1.7B

**Korean automatic speech recognition adapted for telephone and customer-service speech.**

Thaki-ASR-KO-1.7B is a supervised fine-tune of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), with an emphasis on Korean narrowband calls, conversational speech, and number-bearing utterances. This repository publishes the implementation, training recipe, evaluation utilities, and limitations of the released checkpoint.

**[Model & weights](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B)** · **[Model card & full evaluation](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B)** · **[Training recipe](docs/training-recipe.md)** · **[Independent evaluation discussion](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B/discussions/1)** · **[License](LICENSE)**

> **Scope of this release:** This is a code and methodology repository, not a dataset release or a hosted transcription service. Model weights are distributed separately on Hugging Face. Restricted training/evaluation audio and transcripts are not included.

## Overview

| Item | Description |
| --- | --- |
| Base model | `Qwen/Qwen3-ASR-1.7B` |
| Released model | `ThakiCloud/Thaki-ASR-KO-1.7B` |
| Primary focus | Korean telephony, consultation/call-center, and conversational speech |
| Adaptation | Supervised fine-tuning; audio encoder frozen, decoder-side adaptation |
| Release checkpoint | R1 `checkpoint-1200`, selected by telephony development-set CER |
| Training mixture | Telephony 65% / KsponSpeech 20% / general Korean 15% (sample proportions) |
| Audio preparation | 16 kHz, mono, 16-bit PCM WAV |
| Evaluated utterance length | **At most 20 seconds**; longer recordings are not characterized by the release evaluation |
| License | Apache-2.0 for the published project code and released model; training corpora retain separate terms |

The historical release decision was **GO under five pre-registered gates** in the Step 5b evaluation. This is a statement about that fixed protocol and its thresholds, **not** a claim of universal superiority, fully notation-invariant numeric recognition, or independently held-out general-Korean performance. A subsequent release audit clarified important limitations below. The model card is the authoritative record for evaluation versions, results, and audit history.

## Evaluation: results and interpretation

### Original sealed comparison

Both arms used the same decoding protocol. The following figures are historical release measurements, included for context; see the [model card](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B) for the full gates, thresholds, and audit.

| Evaluation | Base | Thaki-ASR-KO-1.7B | Interpretation |
| --- | ---: | ---: | --- |
| Telephony TEST-D, 900 utterances — CER | 6.88% | 6.07% | Improvement on the tested short telephone utterances |
| Numeric-focused TEST-E, 250 utterances — CER | 6.83% | 6.35% | Improvement under the original CER scoring protocol |
| General TEST-C, 456 utterances — CER (B1) | 3.46% | 1.95% | **Not a clean held-out generalization estimate**; see the overlap audit below |
| TEST-E ∪ TEST-C, C1 numeric coverage | 66.4% | 88.7% | **Notation-sensitive reference-number coverage**, not general semantic numeric accuracy |

**C1 denominator:** 327 distinct eligible number-containing utterances, each scored in three greedy-decoding repetitions = **981 scored outputs**, not 981 independent utterances. TEST-E is a subset of TEST-D.

### What C1 actually measures

The frozen evaluator, [`eval/num_match_v2.py`](eval/num_match_v2.py) (`match_utterance_v2`), checks whether all **eligible numeric elements in the reference** are covered by the hypothesis. It handles Arabic-digit runs, certain digit-by-digit Korean readings (e.g., `010` / `공일공`), unit-aware matching for `만` / `천` / `억`, and some segmentation differences.

This is **not** complete Korean number-expression normalization. The metric is biased toward particular notation conventions, and additional numbers in the hypothesis do not invalidate a pass.

| Reference | Hypothesis | C1 behavior |
| --- | --- | --- |
| `52만원` | `오십이만 원` | **FAIL**: place-value Hangul expression is not recognized as the corresponding number |
| `3개 주세요` | `3개 주세요` or `세 개 주세요` | **Not scored**: an isolated single digit without an eligible multiplier is outside C1's reference-element selection |
| `52만원` | `52만 4천원` | **PASS**: the reference element is covered; the extra number is tracked separately but does not fail C1 |
| `010-1234-5678` | `공일공 일이삼사 오육칠팔` | **PASS**: supported digit-by-digit reading |

Accordingly, the original **+22.3 percentage-point C1 difference must not be described as a +22.3 pp gain in semantic number recognition**. Output-format changes may contribute to it. For display, normalization, and inverse text normalization (ITN), evaluate transcription content and formatting behavior separately.

### B1: Zeroth-Korean overlap audit

TEST-C uses the Zeroth-Korean test split, while the training mixture included Zeroth-Korean train data. The **2026-11-04 release audit**, documented in [the training recipe](docs/training-recipe.md), reports that a seeded replay found **454 of 456 TEST-C sentences** in the assembled training mixture. The released build's contamination filter was supplied the telephony development set, **not** the subsequently created TEST-C set.

The B1 result therefore remains part of the frozen historical release record, but it should be read as a **limited no-regression check**, not evidence of improvement on unseen general-Korean sentences. The existence of [`data/sft.py`](data/sft.py)'s overlap filter does not retroactively remove this limitation: it only filters against evaluation sets explicitly provided when the training data are built.

### Independent community re-scoring

Kwan-u Choi independently compared the released model with the original base on six Korean test sets using [kocer](https://pypi.org/project/kocer/) alongside raw CER. In the reported **1,936-utterance telephone set**, the difference remained significant after notation normalization (Thaki minus base: **−1.14 pp**, paired-bootstrap 95% CI **[−1.46, −0.79]**). The **408-utterance telephone number subset** did **not** show a significant kocer difference (−0.10 pp), and output numeral conventions varied by speech domain.

These are **external, separately reported findings**, not an additional result of the original sealed Step 5b evaluation. See the [discussion and methodology](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B/discussions/1). Do not compare scores across datasets as if they shared a test population.

## Quick start: inference with the released weights

Run commands from the repository root. The bundled inference example is configured for a **CUDA device with BF16 support**; it is not a CPU-only inference benchmark.

```bash
python -m pip install torch transformers qwen-asr soundfile scipy

# Normalize your own input to the release audio format.
python -c 'from data.audio import normalize_wav; print(normalize_wav("input.wav", "input_16k.wav"))'

# Run the released checkpoint; downloads weights from Hugging Face.
python examples/infer.py input_16k.wav
```

See [`examples/infer.py`](examples/infer.py) for the exact model identifier and inference parameters. The script prints the library's returned transcription output. For application integration, handle the installed `qwen-asr` version's result structure explicitly.

> **Data handling:** Use audio you are authorized to process. Speech transcripts can contain personal, financial, or sensitive information. Do not submit such material in public GitHub issues.

## Reproducing the methodology

There are two distinct reproducibility targets:

1. **Checkpoint inference and scoring:** The published weights, examples, and deterministic CPU scoring functions can be run on appropriately prepared audio and reference text.
2. **Retraining the exact release:** The recipe and implementation are documented, but the original AI-Hub training corpora and sealed evaluation examples are **not redistributed**. This repository alone is therefore insufficient to regenerate the identical checkpoint or independently reproduce every original score.

### Prepare your own licensed data

The SFT builder accepts JSONL with an `audio` path and `text` transcript. Audio paths are resolved relative to the JSONL file. Pass **every evaluation set you intend to use** to the contamination checker before training.

```json
{"audio":"audio/sample.wav","text":"오늘 상담을 시작하겠습니다","duration":2.4}
```

```bash
python -m pip install soundfile scipy datasets qwen-asr librosa

# Example paths: supply data you have rights to use.
python -m data.sft \
  --train ./train.jsonl \
  --eval ./frozen_eval.jsonl \
  --out ./sft.jsonl \
  --report ./data_gate_report.json
```

The builder rejects exact audio SHA-256 overlap and removes training records whose normalized transcripts match the supplied evaluation references. Inspect its report. **This check does not establish zero contamination against unspecified or future evaluation sets.**

### Fine-tune with the published wrapper

```bash
FREEZE_AUDIO=1 FULL_FP32=1 \
python training/sft_train.py \
  --model_path Qwen/Qwen3-ASR-1.7B \
  --train_file ./sft.jsonl \
  --output_dir ./out \
  --batch_size 4 --grad_acc 32 --lr 2e-6 --epochs 1 \
  --save_steps 100 --save_total_limit 5
```

This command is a **minimal adaptation example**, not a complete replay of the release. The released run additionally applied knowledge distillation to selected general-Korean samples (weight `λ = 0.05`), selected `checkpoint-1200` by development-set CER, and used a specified data mixture. See [`docs/training-recipe.md`](docs/training-recipe.md) and [`training/kd_patch.py`](training/kd_patch.py).

The upstream Qwen3-ASR SFT script is preserved in [`training/qwen3_asr_sft.py`](training/qwen3_asr_sft.py), with provenance recorded in [`training/UPSTREAM_PIN.txt`](training/UPSTREAM_PIN.txt). Local changes are implemented in [`training/sft_train.py`](training/sft_train.py) rather than modifying the pinned upstream file.

### Inspect the CPU-only metrics

```bash
python -m eval.metrics_extra
python -m eval.num_match_v2
```

These commands run the built-in metric controls; **they do not reproduce the sealed model evaluation**. The [`eval/`](eval/) package contains CER/WER scoring, numeric extraction, and the frozen C1 evaluator.

## Repository layout

| Location | Contents |
| --- | --- |
| [`examples/infer.py`](examples/infer.py) | Released-checkpoint inference example |
| [`data/audio.py`](data/audio.py) | 16 kHz mono PCM-16 preprocessing functions |
| [`data/sft.py`](data/sft.py) | SFT JSONL builder and explicit evaluation-overlap checks |
| [`training/`](training/) | Pinned upstream script, training wrapper, and optional KD support |
| [`eval/`](eval/) | Deterministic CER/WER and numeric metric implementations |
| [`docs/training-recipe.md`](docs/training-recipe.md) | Source composition, hyperparameters, selection, and release audit |
| [`LICENSE`](LICENSE) | Apache License 2.0 |

## Known limitations and deployment guidance

- **Short utterances only:** Release scores apply to audio segments of **≤20 seconds**. Long-form transcription, diarization, streaming latency, and production service-level targets were not established by these gates.
- **Numeric transcription is not the same as numeral formatting:** Output style varies across domains. Validate both value correctness and ITN/display conventions for your application.
- **Do not treat C1 as a safety or factuality guarantee:** Its reference-coverage definition can pass outputs containing incorrect additional numbers. Validate monetary values, dates, and identifiers before automated actions.
- **Generalization claims must account for overlap:** B1's TEST-C results are subject to the Zeroth-Korean sentence-overlap audit described above.
- **Domain transfer is not established:** Reported telephone gains should not be generalized to every dialect, noise condition, microphone, language, or speaker population without additional testing.
- **Human review is recommended in consequential workflows:** Do not use an unverified transcript as the sole basis for payments, account changes, compliance actions, or other high-stakes decisions.

## Data provenance and redistribution

The release recipe reports a training mixture consisting of AI-Hub **상담 음성** (dataset 100; telephony), AI-Hub **한국어 음성** (dataset 123; KsponSpeech), and permitted general-Korean sources including Zeroth-Korean and the specified Korean subset of Emilia-YODAS. Dataset-specific licenses and usage conditions apply independently of this repository's code license.

**Not included:** AI-Hub or other original audio, transcripts, training metadata, per-utterance predictions from restricted sets, and internal storage/access information. Please obtain data from their authorized distributors and verify the applicable terms for your intended use. **The Apache-2.0 repository license does not grant rights to any underlying dataset.**

For the original AI-Hub sources, public reporting is limited to permitted aggregate results. Do not redistribute restricted per-utterance source content or predictions derived from it without confirming the applicable permissions.

**Required AI-Hub attribution:**

> 이 모델은 한국지능정보사회진흥원(NIA)의 사업결과인 AI-Hub "상담 음성", "한국어 음성" 데이터를 활용해 학습했습니다.

## License, acknowledgments, and feedback

Project code and released weights are published under **Apache License 2.0**, subject to the respective upstream notices and the independently applicable rights in training data. The bundled official Qwen training script retains its original copyright and Apache-2.0 SPDX notice. See [LICENSE](LICENSE), [UPSTREAM_PIN.txt](training/UPSTREAM_PIN.txt), and the [model repository](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B).

We acknowledge the Qwen team and the providers of the data sources cited in the [training recipe](docs/training-recipe.md). We also thank **Kwan-u Choi** for an independent evaluation that clarified the distinction between transcription accuracy and numeral-format effects.

For **model behavior or benchmark discussion**, use the [Hugging Face model discussions](https://huggingface.co/ThakiCloud/Thaki-ASR-KO-1.7B/discussions). For **code and documentation issues**, use this repository's GitHub Issues. Please report reproducible behavior with software versions and **non-sensitive, redistributable** examples.
