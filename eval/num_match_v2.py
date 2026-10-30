#!/usr/bin/env python3
"""Step 5b 릴리스 검증 숫자 지표 v2 — 단위 인지(value-aware) 의미 매칭. 2026-10-28 동결.

v1 (num_match.py) 의 검증된 맹점 2개를 메운다 — v1 정본은 불변, 본 모듈은
Step 5b 신규 독립 TEST 의 C1 게이트 판정본이다 (평가 실행 전 동결):
  1. ref/hyp 양쪽 아라비아 run 뒤 만·천·억 단위어 확장 ("52만원"→520000, "2만 7천"→[20000,7000])
  2. 단위어가 있는 단일자리 run 포함 ("6천원"→6000 — v1 은 len≥2 로 영점 처리)
  3. 블록 매칭 확장: exact | decomp | fusion — (a) v1 문자열 concatenation (선행 0 보존:
     "010 1234 5678" vs "010-1234-5678") 이나 (b) 단위 관여 블록의 값 합 ("2만 7천" 20000+7000
     vs "27000" = fusion)

PASS 정의는 v1 과 동일하게 "모든 ref 요소 커버" (커버리지 의미, 보존적).
extra hyp 요소(추가 run) 는 n_extra 로 보고만 — PASS 에 영향 없음. v1 robust · legacy exact 는
동일 입력으로 병행 보고(참고)한다. CPU 만, 결정론, 모델 추론 없음.

합(sums) 매칭은 **단위 확장 관여 블록만** 허용 — 무관 블록은 v1 concatenation 대로 유지
(보존적: "02"+"3" 합 5 는 "5" 와 매칭하지 않음).
알려진 보수(과확장) 리스크: run 바로 뒤(여백 제거 후 3자 창) 만·천·억 문자가 무관한 단어
("만화" 등)로 쓰이면 잘못 확장됨 — 상담·금융 도메인에서 희귀하고 양 arm 에 대칭 적용.

`__main__` 블록의 양성·음성 대조가 판정 로직을 검증한다.
"""
from __future__ import annotations
import re
from .metrics_extra import number_check  # noqa: E402
from .num_match import clean_ref, match_utterance  # noqa: E402  (v1 병행 보고)

UNIT_VAL = {"만": 10_000, "천": 1_000, "억": 100_000_000}
_RUN = re.compile(r"\d[\d\-\.]*\d|\d")          # 아라비아 run (단일자리 허용 — len·단위 필터가 걸림)
_KDIG = {"공": "0", "영": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
         "육": "6", "칠": "7", "팔": "8", "구": "9"}
_KRUN = re.compile(r"(?:[공영일이삼사오육칠팔구]\s?){2,}")
BLOCK_CAP = 8


def _unit_mult(suf: str) -> int:
    """run 직후 접미 창 에서 만·천·억 을 찾아 곱. 없으면 1."""
    m = 1
    for ch in suf:
        if ch in UNIT_VAL:
            m *= UNIT_VAL[ch]
    return m


def _suf(text: str, pos: int) -> str:
    """pos 이후 여백 제거 후 앞 3자 (단위 창)."""
    s = text[pos:].lstrip()
    return s[:3]


def elems_in(text: str) -> list[tuple[str, int]]:
    """text 의 (숫자문자열, 단위곱) 요소 목록 (왼쪽→오른쪽 순, 결정론).
    (a) 아라비아 run: len≥2 무조건, len==1 이면 단위어 붙을 때만.
    (b) 한국어 자릿수 읽기 run: mapping 후 len≥2 (len==1 은 단위어 붙을 때만).
    숫자 0 요소는 제외."""
    out: list[tuple[str, int]] = []
    for m in _RUN.finditer(text or ""):
        d = re.sub(r"[^\d]", "", m.group(0))
        mult = _unit_mult(_suf(text, m.end()))
        if len(d) >= 2 or mult > 1:
            if int(d):
                out.append((d, mult))
    for m in _KRUN.finditer(text or ""):
        s = "".join(_KDIG[ch] for ch in m.group(0) if ch in _KDIG)
        if not s:
            continue
        mult = _unit_mult(_suf(text, m.end()))
        if len(s) >= 2 or mult > 1:
            if int(s):
                out.append((s, mult))
    return out


def _val(el: tuple[str, int]) -> int:
    return int(el[0]) * el[1]


def _cover(ref: list[tuple[str, int]], got: list[tuple[str, int]]):
    """요소의 순차 블록 대응 (greedy, 결정론) — v1 concatenation + 단위 관여 블록의 값 합.
    returns: (ref_status, hyp_status) — 각 ['exact'|'decomp'|'fusion'|None]"""
    st, gst = [None] * len(ref), [None] * len(got)
    i = j = 0
    while i < len(ref) and j < len(got):
        r, h = ref[i], got[j]
        if r[0] == h[0] or (max(r[1], h[1]) > 1 and _val(r) == _val(h)):
            st[i] = gst[j] = "exact"; i += 1; j += 1
            continue
        acc_s, acc_v, hit = "", 0, None
        for k in range(j, min(len(got), j + BLOCK_CAP)):      # decomp
            acc_s += got[k][0]
            acc_v += _val(got[k])
            if acc_s == r[0]:
                hit = k
                break
            unit = max(r[1], *(g[1] for g in got[j:k + 1])) > 1
            if unit and acc_v == _val(r):
                hit = k
                break
            if len(acc_s) > len(r[0]) and (not unit or acc_v > _val(r)):
                break
        if hit is not None:
            st[i] = "decomp"
            for k in range(j, hit + 1):
                gst[k] = "decomp"
            i += 1
            j = hit + 1
            continue
        acc_s, acc_v, hit = "", 0, None
        for m in range(i, min(len(ref), i + BLOCK_CAP)):      # fusion
            acc_s += ref[m][0]
            acc_v += _val(ref[m])
            if acc_s == h[0]:
                hit = m
                break
            unit = max(h[1], *(x[1] for x in ref[i:m + 1])) > 1
            if unit and acc_v == _val(h):
                hit = m
                break
            if len(acc_s) > len(h[0]) and (not unit or acc_v > _val(h)):
                break
        if hit is not None:
            for m in range(i, hit + 1):
                st[m] = "fusion"
            gst[j] = "fusion"
            i = hit + 1
            j += 1
            continue
        j += 1
    return st, gst


def match_utterance_v2(text: str, hyp: str) -> dict | None:
    """한 발화 v2 판정. ref(원문 text) 요소가 비면 (비숫자 발화) None.
    v2_robust = 게이트 판정본. v1 robust·legacy 는 동일 입력 병행(참고)."""
    ref_e = elems_in(text)
    if not ref_e:
        return None
    got_e = elems_in(hyp)
    st, gst = _cover(ref_e, got_e)
    v1 = match_utterance(clean_ref(text), hyp)
    return {
        "ref": [(d, m) for d, m in ref_e],
        "got": [(d, m) for d, m in got_e],
        "status": st,
        "hyp_status": gst,
        "v2_robust": all(s is not None for s in st),
        "n_ref": len(ref_e),
        "n_extra": sum(1 for s in gst if s is None),
        "n_unmatched_ref": sum(1 for s in st if s is None),
        "v1_robust": v1["robust"] if v1 else None,
        "legacy_exact": v1["legacy_exact"] if v1 else None,
        "legacy_wrong_plausible": v1["legacy_wrong_plausible"] if v1 else 0,
    }


if __name__ == "__main__":
    import json
    # 양성·음성 대조 (채점기가 틀리는 걸 먼저 잡아내는 discriminative power 검증)
    cases = [
        # (text, hyp, 기대 v2_robust, 설명)
        ("월급은 52만원이에요", "월급은 52만원이에요", True, "단위 정답"),
        ("월급은 52만원이에요", "51만원", False, "단위 값 치환 — 감지"),
        ("2만 7천원", "2만 7천원", True, "단위 분리 정답"),
        ("2만 7천원", "27000원", True, "fusion(합): 20000+7000==27000"),
        ("2만 7천원", "68000원", False, "값 오답 — 감지"),
        ("6천원", "6천원", True, "단일자리 run + 단위 (v1 영점)"),
        ("6천원", "7천원", False, "단일자리 치환 — 감지"),
        ("8000만원", "8000만원", True, "ref 쪽 단위 확장"),
        ("0.4퍼센트", "0.4퍼센트", True, "비율"),
        ("0.4퍼센트", "0.3퍼센트", False, "비율 치환 — 감지"),
        ("전화번호는 010-1234-5678", "010 1234 5678", True, "decomp concatenation (v1 계승, 선행 0)"),
        ("번호는 공일공 일이삼사 오육칠팔", "010-1234-5678", True, "한국어 읽기 vs 아라비아"),
        ("이 계좌는 8000만원", "80000000원", True, "ref 단위 확장 vs hyp 아라비아 풀값"),
        ("52만원", "52만 4천원", True, "커버리지 의미: exact 520000 커버 (extra 4000 보고)"),
    ]
    bad = []
    for text, hyp, expect, why in cases:
        r = match_utterance_v2(text, hyp)
        ok = (r is not None and r["v2_robust"] == expect)
        if not ok:
            bad.append((why, text, hyp, r))
    assert not bad, f"대조 실패: {bad}"
    print(json.dumps({"ok": "num_match_v2", "controls": len(cases)}, ensure_ascii=False))
