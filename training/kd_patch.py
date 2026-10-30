"""Knowledge-distillation preservation loss, imported by `sft_train.py`.

Optional (active only when `KD_TEACHER_ML` or `KD_TEACHER_KO` is set). Adds, at
label token positions only, a KL divergence between a frozen teacher's next-token
distribution and the student's:

    loss = sum_tok [ ce_w(lang) * CE  +  kd_w(lang) * T^2 * KL(teacher || student) ] / sum_tok

  teachers : KD_TEACHER_ML = teacher for non-Korean samples (the original
             Qwen3-ASR base); KD_TEACHER_KO = teacher for Korean samples.
             For this release both point at the original Qwen3-ASR-1.7B base.
  ce_w     : Korean 1.0 ; others KD_ML_CE_WEIGHT (default 0.3)
  kd_w     : Korean KD_ALPHA_KO (default 0.5) ; others KD_ALPHA_ML (default 1.0)
  T        : KD_T temperature (default 1.0)

Teachers are frozen, in eval mode, no_grad, fp32 (same dtype as the student, so
no input recast is needed). The released run applied KD to the **general
(preserved-Korean) samples only** — see `KD_FLAG_MARKER` below and the
training-recipe doc for the exact weights used.

Language decision: from the training-label prefix that `data/sft.py` writes
("language Korean<asr_text>..."). If `KD_FLAG_MARKER` is set, the marker in the
audio path decides which samples get KD instead of the language — this is how
the release restricted KD to the preserved general-Korean subset.
"""
import os
import torch
import torch.nn.functional as F


def enabled() -> bool:
    return bool(os.environ.get("KD_TEACHER_ML") or os.environ.get("KD_TEACHER_KO"))


def is_korean_target(target: str) -> int:
    return 1 if target.lstrip().startswith("language Korean") else 0


def _diag(self, mask, s_logits, tgt, tok_is_ko, kd_tok):
    """KD smoke check (first KD_DIAG_STEPS steps): verifies KD is attached to label
    tokens only and the teacher distribution is aligned. PASS criteria:
    rows == n_label_tok ; kd_ko ~= 0 at step 0 when the KO teacher == init ;
    ML-teacher top-1 accuracy on the answer token >= 0.6."""
    import json
    ko = tok_is_ko == 1
    d = {"step": int(self.state.global_step), "n_label_tok": int(mask.sum()), "kd_rows": int(s_logits.size(0)),
         "n_ko_tok": int(ko.sum()), "n_ml_tok": int((~ko).sum()),
         "kd_ko": float(kd_tok[ko].mean()) if bool(ko.any()) else None,
         "kd_ml": float(kd_tok[~ko].mean()) if bool((~ko).any()) else None,
         "student_top1": float((s_logits.argmax(-1) == tgt[mask]).float().mean()),
         "teacher_top1": getattr(self, "_kd_tacc", {}), "cfg": self._kd_cfg}
    if d["kd_rows"] != d["n_label_tok"]:
        raise RuntimeError(f"[kd] KD rows {d['kd_rows']} != label tokens {d['n_label_tok']}")
    print("[kd-diag] " + json.dumps(d), flush=True)
    self._kd_tacc = {}


def install(up, collator_cls):
    orig_call = collator_cls.__call__

    marker = os.environ.get("KD_FLAG_MARKER", "")

    def call_with_lang(self, features):
        out = orig_call(self, features)
        # If a KD_FLAG_MARKER is set, only samples whose audio path contains it get
        # KD (= preserved general subset). Otherwise use the language decision.
        out["kd_is_ko"] = torch.tensor(
            [(1 if marker in f["audio"] else 0) if marker else is_korean_target(f["target"]) for f in features],
            dtype=torch.long)
        return out

    collator_cls.__call__ = call_with_lang

    def load_teacher(path, device):
        w = up.Qwen3ASRModel.from_pretrained(path, dtype=torch.float32, device_map=None).model
        up.patch_outer_forward(w)
        w.to(device).eval()
        for p in w.parameters():
            p.requires_grad_(False)
        print(f"[kd] teacher loaded: {path} params={sum(p.numel() for p in w.parameters()) / 1e9:.2f}B device={device}", flush=True)
        return w

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        dev = next(model.parameters()).device
        is_ko = inputs.pop("kd_is_ko").to(dev)
        if not hasattr(self, "_kd_teachers"):
            self._kd_teachers = {k: (load_teacher(os.environ[e], dev) if os.environ.get(e) else None)
                                 for k, e in (("ko", "KD_TEACHER_KO"), ("ml", "KD_TEACHER_ML"))}
            self._kd_cfg = dict(T=float(os.environ.get("KD_T", "1.0")), a_ko=float(os.environ.get("KD_ALPHA_KO", "0.5")),
                                a_ml=float(os.environ.get("KD_ALPHA_ML", "1.0")), ce_ml=float(os.environ.get("KD_ML_CE_WEIGHT", "0.3")))
            print(f"[kd] cfg {self._kd_cfg}", flush=True)
        labels = inputs["labels"]
        fwd = {k: v for k, v in inputs.items() if k != "labels"}
        outputs = model(**fwd)
        # Memory: pick out label positions only instead of copying the full
        # [B, L, V] logit tensor (V ~ 152k).
        tgt = labels[:, 1:]
        mask = (tgt != -100)
        full = outputs.logits[:, :-1, :]
        s_logits = full[mask].float()                       # [N, V], row order = batch row order
        tok_is_ko = is_ko.unsqueeze(1).expand_as(mask)[mask]  # [N]
        ce_tok = F.cross_entropy(s_logits, tgt[mask], reduction="none")  # [N]
        cfg = self._kd_cfg
        T = cfg["T"]
        kd_tok = torch.zeros_like(ce_tok)
        B = labels.size(0)
        for key, sel in (("ko", is_ko == 1), ("ml", is_ko == 0)):
            tch = self._kd_teachers.get(key)
            if tch is None or not bool(sel.any()):
                continue
            sub = {k: (v[sel] if torch.is_tensor(v) and v.dim() > 0 and v.size(0) == B else v) for k, v in fwd.items()}
            with torch.no_grad():
                t_sel = tch(**sub).logits[:, :-1, :][mask[sel]].float()  # [n_sel, V]
            tok_sel = (tok_is_ko == (1 if key == "ko" else 0))
            s_logp = F.log_softmax(s_logits[tok_sel] / T, -1)
            t_logp = F.log_softmax(t_sel / T, -1)
            kd_tok[tok_sel] = (t_logp.exp() * (t_logp - s_logp)).sum(-1) * (T * T)
            if model.training and self.state.global_step < int(os.environ.get("KD_DIAG_STEPS", "3")):
                # Alignment check: how often does the teacher's top-1 hit the answer
                # token — a one-position label shift makes this drop to ~0.
                self._kd_tacc = getattr(self, "_kd_tacc", {})
                self._kd_tacc[key] = float((t_sel.argmax(-1) == tgt[mask][tok_sel]).float().mean())
            del t_sel, t_logp, s_logp
        if model.training and self.state.global_step < int(os.environ.get("KD_DIAG_STEPS", "3")):
            _diag(self, mask, s_logits, tgt, tok_is_ko, kd_tok)
        one = torch.ones_like(ce_tok)
        ce_w = torch.where(tok_is_ko == 1, one, one * cfg["ce_ml"])
        kd_w = torch.where(tok_is_ko == 1, one * cfg["a_ko"], one * cfg["a_ml"])
        ntok = mask.sum().clamp(min=1)
        loss = ((ce_w * ce_tok).sum() + (kd_w * kd_tok).sum()) / ntok
        if model.training and self.state.global_step % 10 == 0:
            print(f"[kd] step {self.state.global_step} ce {float(ce_tok.detach().sum() / ntok):.4f} kd {float(kd_tok.detach().sum() / ntok):.4f} ko {int(is_ko.sum())}/{B}", flush=True)
        return (loss, outputs) if return_outputs else loss

    up.CastFloatInputsTrainer.compute_loss = compute_loss
    print("[sft_train] KD preservation loss on",
          {k: os.environ.get(k) for k in ("KD_TEACHER_ML", "KD_TEACHER_KO", "KD_ALPHA_ML", "KD_ALPHA_KO", "KD_ML_CE_WEIGHT", "KD_T")},
          flush=True)
