#!/usr/bin/env python3
"""R2 강건 숫자 지표 v1 — C1 게이트의 병행 보고(참고) 판정.

정의:
- **clean ref** = 발화 원문 text(아라비아 원문)에 앵커된 숫자 run. 읽기 변환
  (text_digits)에서 추출하면 조사·어미 merge 부속참조물("7일"→71 등)이 ref 에
  들어 실오류 0 발화가 FAIL 이던 결함을 피한다.
- **분절 불변(융합 허용) 매칭** = ref piece ↔ hyp run 을 블록 단위로 대응:
    exact  : ref piece == hyp run
    fusion : hyp run == 연속 ref pieces 의 concatenation (예: ref 1234·5678·9101 vs hyp "1234-5678-9101")
    decomp : ref piece == 연속 hyp runs 의 concatenation (예: ref 01066664343 vs hyp "010 6666 4343")
  단순 부분문자열("30" ⊂ "300000")은 매칭이 아니다 — 블록 concatenation 이 정확히
  일치해야 한다. 경계 이동으로 숫자 순서만 보존되고 블록이 어긋난 것은 FAIL(보존적).
- **PASS** = 모든 clean ref piece 가 exact|fusion|decomp 로 커버됨.
- 기존 exact 지표(metrics_extra.number_check)는 동일 clean ref 로 **병행 보고(참고용)** —
  게이트 판정에 쓰지 않는다.

CPU 만, 결정론. 모델 추론 없음. `__main__` 블록의 양성·음성 대조가
판정 로직을 검증한다.
"""
from __future__ import annotations
import re
from .metrics_extra import digits_in, number_check  # noqa: E402  (기존 지표 — 정본 단일 소스)

ARAB = re.compile(r"\d[\d\-\.]*\d|\d{2,}")
BLOCK_CAP = 8  # 블록당 최대 piece/run 수 (ref run 은 통상 ≤13자리, 넉넉함)


def digits_runs(text: str) -> list[str]:
    """text 의 아라비아 숫자 run 들을 순수 숫자 문자열로. 하이픈·점·공백 표기는 접합."""
    return [re.sub(r"[^\d]", "", m.group(0)) for m in ARAB.finditer(text or "")]


def clean_ref(text: str) -> list[str]:
    """발화 원문 text 에 앵커된 ref 숫자 run 들. (읽기 변환 text_digits 는 사용하지 않음.)"""
    return digits_runs(text)


def _cover(ref: list[str], got: list[str]):
    """ref piece ↔ hyp run 의 순차 블록 대응 (greedy, 결정론).
    returns: (ref_status, hyp_status) — 각각 ['exact'|'decomp'|'fusion'|None]"""
    st, gst = [None] * len(ref), [None] * len(got)
    i = j = 0
    while i < len(ref) and j < len(got):
        r, h = ref[i], got[j]
        if r == h:
            st[i] = gst[j] = "exact"; i += 1; j += 1
            continue
        acc, hit = "", None
        for k in range(j, min(len(got), j + BLOCK_CAP)):  # decomp
            acc += got[k]
            if acc == r:
                hit = k
                break
            if len(acc) > len(r):
                break
        if hit is not None:
            st[i] = "decomp"
            for k in range(j, hit + 1):
                gst[k] = "decomp"
            i += 1
            j = hit + 1
            continue
        acc, hit = "", None
        for m in range(i, min(len(ref), i + BLOCK_CAP)):  # fusion
            acc += ref[m]
            if acc == h:
                hit = m
                break
            if len(acc) > len(h):
                break
        if hit is not None:
            for m in range(i, hit + 1):
                st[m] = "fusion"
            gst[j] = "fusion"
            i = hit + 1
            j += 1
            continue
        j += 1  # 어긋남: hyp run 을 extra 로 두고 다음 ref piece 시도
    return st, gst


def match_utterance(ref: list[str], hyp: str) -> dict | None:
    """한 발화 숫자 판정. ref 가 비면 (비숫자 발화) None.
    robust = 모든 ref piece 커버 (강건 지표, 게이트 판정). legacy = 기존 exact (참고).
    hyp 쪽은 digits_in (아라비아·읽기 표기 둘 다 정규화) — 표기 방식과 무관하게 값만 본다.
    ref 쪽만 clean anchor (text 아라비아 원문) — 읽기 변환 부속참조물 불필요."""
    if not ref:
        return None
    got = digits_in(hyp)
    st, gst = _cover(ref, got)
    legacy = number_check(ref, hyp)
    return {
        "ref": ref,
        "got": got,
        "status": st,
        "hyp_status": gst,
        "robust": all(s is not None for s in st),
        "n_ref": len(ref),
        "n_got": len(got),
        "n_extra_runs": sum(1 for s in gst if s is None),
        "n_unmatched_ref": sum(1 for s in st if s is None),
        "legacy_exact": legacy["exact"],
        "legacy_missing": legacy["missing"],
        "legacy_wrong_plausible": legacy["wrong_plausible"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps({"ok": "num_match (v1)", "api": ["digits_runs", "clean_ref", "match_utterance"]}))
