"""CER 밖의 숫자 지표 — digits_in(숫자열 추출) + number_check(exact / wrong-plausible).

결정론, 모델 판단 없음. 숫자 매칭(v1·v2)에 쓰이는 `digits_in`(숫자열
추출)·`number_check`(exact / wrong-plausible 판정) 두 함수만 포함한다.
용어 오류·환각 탐지 같은 이 평가와 무관한 지표는 제외했다.
"""
from __future__ import annotations
import re

_KDIG = {"공": "0", "영": "0", "일": "1", "이": "2", "삼": "3", "사": "4", "오": "5",
         "육": "6", "칠": "7", "팔": "8", "구": "9"}
_KRUN = re.compile(r"(?:[공영일이삼사오육칠팔구]\s?){2,}")


def digits_in(text: str) -> list[str]:
    """텍스트의 숫자열(길이≥2). 아라비아 숫자(하이픈·점 연결 포함)와 한국어 자릿수
    읽기(공일공 …) 둘 다 뽑아 숫자만 남긴다."""
    out = []
    for m in re.finditer(r"\d[\d\-\.]*\d|\d{2,}", text):
        out.append(re.sub(r"[^\d]", "", m.group(0)))
    for m in _KRUN.finditer(text):
        s = "".join(_KDIG[ch] for ch in m.group(0) if ch in _KDIG)
        if len(s) >= 2:
            out.append(s)
    return [d for d in out if len(d) >= 2]


def number_check(ref_numbers: list[str], hyp: str) -> dict:
    """정답 숫자 전부가 가설에 있으면 exact. 없는 정답과 길이가 같은 다른 숫자가
    가설에 있으면 wrong_plausible(형식은 유효한데 값이 틀림)."""
    want = [re.sub(r"[^\d]", "", n) for n in ref_numbers]
    got = digits_in(hyp)
    missing = [w for w in want if w not in got]
    wrong_plausible = sum(1 for w in missing if any(len(g) == len(w) and g != w for g in got))
    return {"exact": not missing, "missing": len(missing), "wrong_plausible": wrong_plausible, "n_ref": len(want)}


if __name__ == "__main__":
    # 양성 대조 — 채점기가 틀린 걸 잡는지 먼저 본다
    assert digits_in("제 번호는 공일공 일이삼사 오육칠팔 입니다") == ["01012345678"]
    assert digits_in("010-1234-5678 로") == ["01012345678"]
    assert number_check(["010-1234-5678"], "번호는 공일공 일이삼사 오육칠팔")["exact"]
    r = number_check(["010-1234-5678"], "번호는 공일공 일이사삼 오육칠팔"); assert not r["exact"] and r["wrong_plausible"] == 1, r
    r = number_check(["010-1234-5678"], "번호는 잘 안 들려요"); assert r["missing"] == 1 and r["wrong_plausible"] == 0
    print("metrics_extra: positive controls OK")
