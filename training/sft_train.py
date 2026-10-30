"""Thin wrapper over the official Qwen3-ASR SFT script that fixes one upstream
bug and adds the training guards used for this release.

It imports the pinned official script (`training/qwen3_asr_sft.py`, provenance
SHA in `UPSTREAM_PIN.txt`) and patches it in-memory. The vendor file itself is
left byte-for-byte identical to upstream; all changes live here so the pin stays
verifiable.

Why the collator fix (the one real upstream bug):
  The `qwen-asr` processor defaults to `padding_side="left"`, but the official
  collator masks the prompt with `labels[i, :prefix_len] = -100` (from the left).
  For the shorter samples in a batch the mask then shifts by the padding amount,
  so the prompt and the <|AUDIO|> placeholder tokens leak into the labels. With
  the default left padding the run's train_loss sits above chance and eval hits
  NaN. This wrapper forces `padding_side="right"` and masks the right side
  instead, so the prompt is masked correctly regardless of padding.

Optional env knobs:
  FULL_FP32=1         pure fp32 (bf16/fp16 autocast off) + eager attention
  FREEZE_AUDIO=1      freeze the audio encoder; train only the decoder/ASR head
  SAVE_ONLY_MODEL=1   write weights-only checkpoints (no optimizer state)
  SMOKE_STOP_STEP=N   stop cleanly N steps after the latest checkpoint
  KD_TEACHER_ML / KD_TEACHER_KO  (+ KD_* see below)
                      enable the KD preservation loss (see `kd_patch.py`)
"""
import os
import sys

import qwen3_asr_sft as up  # pinned official script (same directory)


class RightPadCollator(up.DataCollatorForQwen3ASRFinetuning):
    """Force right padding so the prompt mask does not shift by the pad length."""

    def __call__(self, features):
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [up.load_audio(p, sr=self.sampling_rate) for p in audio_paths]
        kw = dict(return_tensors="pt", padding=True, padding_side="right", truncation=False)
        full_inputs = self.processor(text=full_texts, audio=audios, **kw)
        prefix_inputs = self.processor(text=prefix_texts, audio=audios, **kw)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        # Fail loudly if right padding did not actually take effect.
        am = full_inputs["attention_mask"]
        if am.shape[0] > 1 and not bool((am[:, 0] == 1).all()):
            raise RuntimeError("collator: attention_mask has a 0 in the first column — left padding was applied (padding_side ignored)")
        full_inputs["labels"] = labels
        return full_inputs


up.DataCollatorForQwen3ASRFinetuning = RightPadCollator

# FREEZE_AUDIO=1 — freeze the audio encoder (thinker.audio_tower.*), train only the
# LLM + projector. The released model froze the audio encoder: domain/numeric
# adaptation is carried by the decoder side, and freezing avoids the audio-tower
# backward NaN that appears on a pure full fine-tune at this precision/attention.
_orig_from_pretrained = up.Qwen3ASRModel.from_pretrained


def _from_pretrained_freeze(*a, **k):
    # FULL_FP32=1 — pure fp32 (no autocast) + eager attention. On a full fine-tune
    # the audio-tower backward is NaN under bf16 at these batch sizes; fp32 +
    # eager is clean. Memory: 1.7B fp32 + Adam ≈ 27 GB plus activations.
    if os.environ.get("FULL_FP32", "0") == "1":
        import torch as _t
        k["dtype"] = _t.float32
        k["attn_implementation"] = "eager"
        print("[sft_train] FULL_FP32=1: dtype=float32, attn=eager (bf16 autocast off)", flush=True)
    w = _orig_from_pretrained(*a, **k)
    if os.environ.get("FULL_FP32", "0") == "1":
        try:
            w.model.config._attn_implementation = "eager"
        except Exception:
            pass
    if os.environ.get("FREEZE_AUDIO", "0") == "1":
        n_f = n_t = 0
        for name, prm in w.model.named_parameters():
            if "audio_tower" in name:
                prm.requires_grad_(False)
                n_f += prm.numel()
            else:
                n_t += prm.numel()
        print(f"[sft_train] FREEZE_AUDIO=1: frozen audio_tower params={n_f / 1e6:.1f}M, trainable={n_t / 1e6:.1f}M", flush=True)
    else:
        print("[sft_train] FREEZE_AUDIO=0: full fine-tune (audio_tower trainable)", flush=True)
    return w


up.Qwen3ASRModel.from_pretrained = staticmethod(_from_pretrained_freeze)


class _TrainingArgumentsFP32(up.TrainingArguments):
    """Turn the Trainer's bf16/fp16 autocast off when FULL_FP32=1 (the official
    script enables bf16 from the GPU capability)."""

    def __init__(self, *a, **k):
        if os.environ.get("FULL_FP32", "0") == "1":
            k["bf16"] = False
            k["fp16"] = False
        if os.environ.get("SAVE_ONLY_MODEL", "0") == "1":
            k["save_only_model"] = True
        super().__init__(*a, **k)


up.TrainingArguments = _TrainingArgumentsFP32


class _GuardCallback(up.TrainerCallback if hasattr(up, "TrainerCallback") else __import__("transformers").TrainerCallback):
    """(a) NaN/Inf loss -> stop cleanly (keep the last checkpoint).
    (b) SMOKE_STOP_STEP=N -> stop cleanly right after the N-th checkpoint.
    (c) every 10 steps, print s/step and a refreshed ETA.
    (d) on resume, log the starting global_step."""

    def __init__(self):
        import time
        self.t0 = None
        self.s0 = None
        self.time = time
        self.stop_at = int(os.environ.get("SMOKE_STOP_STEP", "0") or 0)

    def on_train_begin(self, args, state, control, **kw):
        self.t0 = self.time.time()
        self.s0 = state.global_step
        print(f"[guard] train begin at global_step={state.global_step} max_steps={state.max_steps} (resumed={state.global_step > 0})", flush=True)
        return control

    def on_log(self, args, state, control, logs=None, **kw):
        import math
        loss = (logs or {}).get("loss")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            print(f"[guard] NaN/Inf loss at step {state.global_step} — stopping (last checkpoint kept)", flush=True)
            control.should_training_stop = True
        done = state.global_step - (self.s0 or 0)
        if self.t0 and done > 0 and state.global_step % 10 == 0:
            sps = (self.time.time() - self.t0) / done
            left = max(state.max_steps - state.global_step, 0) * sps
            print(f"[guard] step {state.global_step}/{state.max_steps} {sps:.1f}s/step ETA {left / 60:.0f}min", flush=True)
        return control

    def on_save(self, args, state, control, **kw):
        if self.stop_at and state.global_step >= self.stop_at:
            print(f"[guard] SMOKE_STOP_STEP={self.stop_at}: saving at step {state.global_step}, then stopping", flush=True)
            control.should_training_stop = True
        return control


_orig_trainer_init = up.CastFloatInputsTrainer.__init__


def _trainer_init_guarded(self, *a, **k):
    _orig_trainer_init(self, *a, **k)
    self.add_callback(_GuardCallback())


up.CastFloatInputsTrainer.__init__ = _trainer_init_guarded

import kd_patch  # noqa: E402  (KD preservation loss — active only when KD_TEACHER_* is set)
if kd_patch.enabled():
    kd_patch.install(up, RightPadCollator)

if __name__ == "__main__":
    sys.exit(up.main())
