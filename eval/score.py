#!/usr/bin/env python3
"""Korean-aware ASR scoring. The METRIC IS OWNED BY CODE, never by the model.

Rules honoured:
  [[sonnet-format-determinism]] the worker emits text only; CER/WER are computed here.
  [[metric-must-discriminate]]  ships a positive control so we can prove the metric
                                actually catches damage before we compare arms.

Korean specifics that are easy to get wrong and silently ruin a comparison:
  * NFC.   macOS / some datasets hand you NFD (decomposed jamo). '한' is then 3
           codepoints, not 1, and CER becomes meaningless. Normalise BOTH sides.
  * CER is the primary metric. Korean spacing (띄어쓰기) is not standardised
           enough for WER to discriminate models — WER is reported but flagged.
  * Punctuation / numerals are stripped, because ASR arms differ in whether they
           emit them at all, which is a formatting difference, not an accuracy one.
"""
from __future__ import annotations

import re
import unicodedata

# strip punctuation/symbols; keep letters of ANY script + digits + space.
# ⛔ 2026-10-22 실측: 예전 `[^가-힣a-z0-9 ]` 는 한자·가나·아랍 문자를 전부 지워 FLEURS zh/ar CER 이 구조적으로 1.0 이었다(ja 는 잔재만 채점).
#    한국어 참조·가설에는 한글·라틴·숫자 외 글자가 사실상 없어 KO 게이트(G1~G7) 값은 변하지 않는다(eval/test_score.py 로 확인).
_KEEP = re.compile(r"[^\w ]|_")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Deterministic normalisation applied identically to reference and hypothesis."""
    t = unicodedata.normalize("NFC", text or "")
    t = t.lower()
    t = _KEEP.sub(" ", t)
    return _WS.sub(" ", t).strip()


def _edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein over arbitrary token sequences. O(len(ref)) memory."""
    if not ref:
        return len(hyp)
    prev = list(range(len(ref) + 1))
    for j, h in enumerate(hyp, 1):
        cur = [j]
        for i, r in enumerate(ref, 1):
            cur.append(min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (r != h)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> dict:
    """Character error rate with whitespace REMOVED (Korean standard)."""
    r = normalize(reference).replace(" ", "")
    h = normalize(hypothesis).replace(" ", "")
    d = _edit_distance(list(r), list(h))
    return {"errors": d, "length": len(r), "rate": (d / len(r)) if r else None}


def wer(reference: str, hypothesis: str) -> dict:
    """Word error rate over whitespace tokens. Spacing-sensitive — secondary only."""
    r = normalize(reference).split()
    h = normalize(hypothesis).split()
    d = _edit_distance(r, h)
    return {"errors": d, "length": len(r), "rate": (d / len(r)) if r else None}


def corpus_score(pairs) -> dict:
    """Aggregate over (reference, hypothesis) pairs.

    Aggregation is micro-averaged (total errors / total length), NOT the mean of
    per-utterance rates — a 3-character utterance must not carry the same weight
    as a 200-character one.
    """
    ce = cl = we = wl = 0
    n = 0
    for ref, hyp in pairs:
        c, w = cer(ref, hyp), wer(ref, hyp)
        ce += c["errors"]; cl += c["length"]
        we += w["errors"]; wl += w["length"]
        n += 1
    return {
        "utterances": n,
        "cer": (ce / cl) if cl else None,
        "wer": (we / wl) if wl else None,
        "cer_errors": ce, "cer_ref_chars": cl,
        "wer_errors": we, "wer_ref_words": wl,
    }


# ---------------------------------------------------------------------------
# Digit-vs-spelled-out mismatch.
#
# zeroth-korean references are READ-SPEECH transcripts: numbers are written the
# way they are spoken ("지난해 삼 월"), not as digits ("지난해 3월"). An ASR arm
# that emits digits is not less accurate — it formats differently — but raw CER
# punishes it anyway. That would rank arms by output convention, not by hearing.
#
# So we do two things, and we NEVER silently replace the primary number:
#   * report `digit_utt_ratio`, a diagnostic: how much of this arm's output even
#     contains digits. Near zero => the whole issue is moot for this arm.
#   * report a SECONDARY cer, `cer_numnorm`, where sino-Korean digit readings are
#     applied to the hypothesis. Sino-Korean (일이삼...) is the right system for
#     dates/counts in news text, which is what zeroth-korean is.
# ---------------------------------------------------------------------------

_SINO = "영일이삼사오육칠팔구"
_HAS_DIGIT = re.compile(r"[0-9]")


def _read_sino(digits: str) -> str:
    """Digit-by-digit sino-Korean reading: '3' -> '삼', '2026' -> '이영이육'.

    Deliberately digit-by-digit rather than full myriad grouping (이천이십육).
    Read-speech references for years/dates in this corpus use the digit-by-digit
    form, and a wrong-but-confident myriad expansion would inject errors of its
    own. Bounded and predictable beats clever.
    """
    return "".join(_SINO[int(c)] for c in digits)


def numnorm(text: str) -> str:
    """Replace digit runs in a hypothesis with sino-Korean readings."""
    return re.sub(r"[0-9]+", lambda m: _read_sino(m.group()), text or "")


def corpus_score_full(pairs) -> dict:
    """Primary score + the digit-convention diagnostics."""
    out = corpus_score(pairs)
    pairs = list(pairs)
    with_digits = sum(1 for _, h in pairs if _HAS_DIGIT.search(normalize(h) or ""))
    out["digit_utt_ratio"] = (with_digits / len(pairs)) if pairs else 0.0
    out["cer_numnorm"] = corpus_score([(r, numnorm(h)) for r, h in pairs])["cer"]
    # if this gap is large, the raw ranking is a formatting artefact, not accuracy
    if out["cer"] is not None and out["cer_numnorm"] is not None:
        out["numnorm_delta_pp"] = round((out["cer"] - out["cer_numnorm"]) * 100, 3)
    return out
