"""강의 자료 일관성 검사기: lecture-notes/ + instructor-scripts/ 전수 점검

강의를 매일 찍지 않다 보면 클립 간 형식이 조용히 어긋납니다. 이 스크립트는
"사람이 눈으로 43개를 다시 읽지 않아도" 잡히는 것들을 자동으로 잡습니다.

검사 항목
    P1  프롬프트 4칸 골격      ```prompt 블록에 [맥락] / [만들 것]|[수정할 것] / [하지 말 것] / [확인]이 모두 있는가
    P2  구형 표기 잔존         (Context) (Deliverable) (Verification) 같은 영문 괄호 표기가 남아 있는가
    C1  차트 형식              Plotly를 배운 클립 09 이후의 프롬프트가 .png 차트를 요구하는가 (→ .html이어야 함)
    H1  HTML 재빌드 누락        .md의 프롬프트 첫 줄이 같은 이름의 .html에 없으면 빌드를 빠뜨린 것
    S1  대본↔교재 파일명 불일치  instructor-scripts가 언급한 차트 파일(.html/.png)이 교재 어디에도 없는가
    F1  종목 유니버스 이탈      Part 3~5에 4대 전략 유니버스 밖의 종목코드가 등장하는가
    F2  고정 위험값 변조        2층 하드 가드레일 값(-0.03 / -0.02)이 다른 값으로 적혀 있는가
    F3  금지 표현              폐기된 수치·용어가 남아 있는가 (아래 BANNED_PHRASES)
    N1  실습 번호 불일치        교재 본문의 '실습 NN'과 curriculum.json의 practice 값이 다른가, 번호가 끊기거나 겹치는가

F1~F3은 "앞에서 A라고 했는데 뒤에서 B라고 하는" 종류의 어긋남을 잡기 위한 **사실 목록**입니다.
새 사실이 확정되면 아래 상수(STRATEGY_UNIVERSE / FIXED_RISK_VALUES / BANNED_PHRASES)에 한 줄 추가하세요.

의도적으로 골격을 따르지 않는 프롬프트(예: 클립 03 이전의 짧은 진단 요청)는
```prompt 펜스 위 5줄 안에 다음 주석을 두면 P1/P2 검사에서 제외됩니다:
    <!-- prompt-check: skip — 이유 -->
주의: 이 주석을 목록 항목 안(들여쓴 줄)에 넣으면 목록이 끊어집니다. 목록 위에 빈 줄로 감싸서 두세요.

실행:
    .venv/bin/python runners/check_lecture_consistency.py
    (실패가 하나라도 있으면 종료 코드 1)
"""

from __future__ import annotations

import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECTURE_DIR = ROOT / "lecture-notes"
SCRIPT_DIR = ROOT / "instructor-scripts"
CURRICULUM_PATH = LECTURE_DIR / "_core" / "curriculum.json"

PROMPT_BLOCK_RE = re.compile(r"^([ \t]*)```prompt[ \t]*\n(.*?)^\1```", re.DOTALL | re.MULTILINE)
SKIP_MARK = "prompt-check: skip"
SKIP_LOOKBACK_LINES = 5

# 클립 03에서 가르치는 두 변형: 신규 생성(만들 것) / 후속 수정(수정할 것)
REQUIRED_HEADERS: list[tuple[str, tuple[str, ...]]] = [
    ("[맥락]", ("[맥락]",)),
    ("[만들 것]|[수정할 것]", ("[만들 것]", "[수정할 것]")),
    ("[하지 말 것]", ("[하지 말 것]",)),
    ("[확인]", ("[확인]",)),
]
LEGACY_TAGS = ("(Context)", "(Deliverable)", "(Verification)")

# Plotly 대화형 HTML 차트는 클립 09에서 도입. 그 전(04, 07)은 matplotlib PNG가 의도된 것.
PLOTLY_INTRO_CLIP = 9
PNG_IN_PROMPT_RE = re.compile(r"\.png\b")

CHART_FILE_RE = re.compile(r"[A-Za-z0-9_\-]+\.(?:html|png)\b")

# ── 사실 목록 (fact registry) ────────────────────────────────────────────────
# Part 3부터 실전 운용까지 한 줄로 이어지는 값들. 여기서 벗어나면 앞뒤 말이 달라진다.

# 클립 16에서 확정한 4대 전략 유니버스. Part 3~5에는 이 종목만 등장해야 한다.
STRATEGY_UNIVERSE = {
    "069500": "KODEX 200 (추세추종)",
    "122630": "KODEX 레버리지 (변동성 돌파)",
    "133690": "TIGER 미국나스닥100 (듀얼 모멘텀)",
    "305080": "KODEX 미국채10년선물 (듀얼 모멘텀)",
    "214980": "KODEX 단기채권PLUS (듀얼 모멘텀)",
    "105560": "KB금융 (페어트레이딩)",
    "086790": "하나금융지주 (페어트레이딩)",
}
UNIVERSE_FROM_CLIP = 16  # Part 3 시작. 그 이전(Part 1~2)은 예시 종목이 자유롭다.

# 교재가 종목코드를 적는 두 가지 형태만 인식한다 (일반 6자리 숫자 오탐 방지).
TICKER_RE = re.compile(r"종목코드[:：]?\s*'?(\d{6})|\((\d{6})\)")

# 2층 하드 가드레일. 클립 01에서 선언했고 어떤 클립에서도 바뀌지 않는다.
FIXED_RISK_VALUES = {
    "hard_stop_loss_pct": "-0.03",
    "daily_portfolio_circuit_breaker_pct": "-0.02",
}
RISK_VALUE_RE = re.compile(r'"(hard_stop_loss_pct|daily_portfolio_circuit_breaker_pct)"\s*:\s*(-?[\d.]+)')

# 폐기된 수치·용어. (표현, 왜 안 되는가)
BANNED_PHRASES = [
    ("8.5년", "Part 3는 Part 2와 같은 2016-01-01~2026-08-31, 약 10.7년입니다"),
    ("8.7년", "Part 3는 Part 2와 같은 2016-01-01~2026-08-31, 약 10.7년입니다"),
    ("과적합", "용어는 '과최적화'로 통일합니다 (클립 17 정의)"),
    ("드로다운", "용어는 '낙폭(Drawdown)'으로 통일합니다 (클립 07 정의)"),
    ("알파", "강의에서 정의하지 않은 용어입니다. '추가 수익', '주력 수익원'처럼 풀어 쓰세요"),
]

# 본문 실습 번호 추출. 제목 뒤에 부제가 붙기도 하고(클립 09) 번호로 끝나기도 한다(클립 05).
#   '## 실습 — 실습 12: ...' / '## 최종 실습 — 실습 31: ...' / '## 실습 — 실습 03'
BODY_PRACTICE_RE = re.compile(r"^##\s.*?실습\s(\d{2})(?:\s*[:：]|\s*$)", re.MULTILINE)


@dataclass
class Finding:
    check: str
    level: str  # FAIL | WARN
    path: Path
    line: int
    message: str

    def render(self) -> str:
        rel = self.path.relative_to(ROOT)
        return f"  [{self.level}] {self.check}  {rel}:{self.line}  {self.message}"


def clip_number(path: Path) -> int:
    m = re.match(r"(\d+)_", path.stem)
    return int(m.group(1)) if m else -1


def iter_prompt_blocks(text: str):
    """(시작 줄 번호, 본문, skip 여부) 를 차례로 돌려준다."""
    lines = text.splitlines()
    for m in PROMPT_BLOCK_RE.finditer(text):
        start_line = text[: m.start()].count("\n") + 1  # 1-based, ```prompt 줄
        lookback = lines[max(0, start_line - 1 - SKIP_LOOKBACK_LINES) : start_line - 1]
        skipped = any(SKIP_MARK in ln for ln in lookback)
        yield start_line, m.group(2), skipped


def comparable_lines(body: str) -> list[str]:
    """HTML과 대조할 수 있는 줄만 고른다.

    빌드 과정에서 `$...$` 수식은 <span class="math-inline">로 바뀌므로 그 줄은 제외한다.
    """
    return [ln.strip() for ln in body.splitlines() if ln.strip() and "$" not in ln]


def check_prompts(md: Path, text: str, findings: list[Finding]) -> None:
    clip = clip_number(md)
    for start, body, skipped in iter_prompt_blocks(text):
        if not skipped:
            missing = [label for label, variants in REQUIRED_HEADERS if not any(v in body for v in variants)]
            if missing:
                findings.append(Finding("P1 4칸 골격", "FAIL", md, start, f"빠진 칸: {', '.join(missing)}"))
            legacy = [t for t in LEGACY_TAGS if t in body]
            if legacy:
                findings.append(Finding("P2 구형 표기", "FAIL", md, start, f"영문 괄호 표기 잔존: {', '.join(legacy)}"))
        if clip >= PLOTLY_INTRO_CLIP and PNG_IN_PROMPT_RE.search(body):
            findings.append(
                Finding("C1 차트 형식", "FAIL", md, start, "클립 09 이후 프롬프트가 .png 차트를 요구함 → 대화형 Plotly .html로")
            )


def check_html_sync(md: Path, text: str, findings: list[Finding]) -> None:
    html_path = md.with_suffix(".html")
    if not html_path.exists():
        findings.append(Finding("H1 HTML 누락", "FAIL", md, 1, "같은 이름의 .html이 없음 → runners/build_lecture_notes.py 실행"))
        return
    rendered = html_path.read_text(encoding="utf-8")
    for start, body, _ in iter_prompt_blocks(text):
        for probe in comparable_lines(body):
            # 빌더는 코드 블록에서 & < > " 를 엔티티로 바꾸지만 ' 는 그대로 둔다.
            escaped = html.escape(probe, quote=False).replace('"', "&quot;")
            if not any(c in rendered for c in (probe, escaped)):
                findings.append(
                    Finding(
                        "H1 HTML 재빌드",
                        "FAIL",
                        md,
                        start,
                        f"프롬프트의 이 줄이 .html에 없음 → 재빌드 누락: {probe[:60]}",
                    )
                )
                return  # 한 파일에 한 번만 보고


def line_of(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


def check_facts(md: Path, text: str, findings: list[Finding]) -> None:
    """F1~F3 — 클립을 건너뛰며 조용히 달라지는 값들을 잡는다."""
    clip = clip_number(md)

    if clip >= UNIVERSE_FROM_CLIP:
        for m in TICKER_RE.finditer(text):
            ticker = m.group(1) or m.group(2)
            if ticker not in STRATEGY_UNIVERSE:
                findings.append(
                    Finding(
                        "F1 종목 유니버스",
                        "FAIL",
                        md,
                        line_of(text, m.start()),
                        f"'{ticker}'은 4대 전략 유니버스 밖의 종목입니다 (백테스트한 적 없는 종목이 실전에 들어옴)",
                    )
                )

    for m in RISK_VALUE_RE.finditer(text):
        key, value = m.group(1), m.group(2)
        expected = FIXED_RISK_VALUES[key]
        if value != expected:
            findings.append(
                Finding("F2 고정 위험값", "FAIL", md, line_of(text, m.start()), f"{key}는 {expected} 고정인데 {value}로 적혀 있음")
            )

    for phrase, why in BANNED_PHRASES:
        idx = text.find(phrase)
        if idx != -1:
            findings.append(Finding("F3 금지 표현", "FAIL", md, line_of(text, idx), f"'{phrase}' — {why}"))


def check_practice_numbers(notes: list[Path], note_texts: dict[Path, str], findings: list[Finding]) -> None:
    """N1 — 본문 실습 번호와 목차(curriculum.json)가 같은가, 번호가 끊기거나 겹치지 않는가."""
    if not CURRICULUM_PATH.exists():
        return
    curriculum = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))

    seen: dict[int, Path] = {}
    for md in notes:
        text = note_texts[md]
        meta = curriculum.get(md.stem)
        if meta is None:
            continue
        body_nums = [int(n) for n in BODY_PRACTICE_RE.findall(text)]
        toc = meta.get("practice", "")
        toc_num = int(toc.split()[-1]) if toc else None

        if not body_nums:
            if toc_num is not None:
                findings.append(
                    Finding("N1 실습 번호", "FAIL", md, 1, f"목차는 '{toc}'인데 본문에 번호 붙은 실습 제목이 없음")
                )
            continue

        num = body_nums[0]
        if toc_num is None:
            findings.append(
                Finding("N1 실습 번호", "FAIL", md, 1, f"본문은 '실습 {num:02d}'인데 curriculum.json의 practice가 비어 있음")
            )
        elif toc_num != num:
            findings.append(
                Finding("N1 실습 번호", "FAIL", md, 1, f"본문 '실습 {num:02d}' ↔ 목차 '{toc}' 불일치")
            )
        if num in seen:
            findings.append(
                Finding("N1 실습 번호", "FAIL", md, 1, f"실습 {num:02d}가 {seen[num].stem}와 중복")
            )
        seen[num] = md

    if seen:
        missing = sorted(set(range(1, max(seen) + 1)) - set(seen))
        if missing:
            nums = ", ".join(f"{n:02d}" for n in missing)
            findings.append(Finding("N1 실습 번호", "FAIL", CURRICULUM_PATH, 1, f"빠진 실습 번호: {nums}"))


def check_script_refs(all_note_text: str, findings: list[Finding]) -> None:
    for script in sorted(SCRIPT_DIR.glob("Part*/*.md")):
        text = script.read_text(encoding="utf-8")
        for lineno, ln in enumerate(text.splitlines(), 1):
            for m in CHART_FILE_RE.finditer(ln):
                name = m.group(0)
                if name not in all_note_text:
                    findings.append(
                        Finding("S1 대본↔교재", "WARN", script, lineno, f"대본이 언급한 차트 파일 '{name}'이 교재 어디에도 없음")
                    )


def main() -> int:
    notes = sorted(LECTURE_DIR.glob("Part*/*.md"))
    if not notes:
        print(f"검사할 노트가 없습니다: {LECTURE_DIR}/Part*/*.md")
        return 2

    findings: list[Finding] = []
    note_texts: dict[Path, str] = {p: p.read_text(encoding="utf-8") for p in notes}

    for md, text in note_texts.items():
        check_prompts(md, text, findings)
        check_html_sync(md, text, findings)
        check_facts(md, text, findings)

    check_practice_numbers(notes, note_texts, findings)
    check_script_refs("\n".join(note_texts.values()), findings)

    prompt_total = sum(len(list(iter_prompt_blocks(t))) for t in note_texts.values())
    skipped_total = sum(1 for t in note_texts.values() for _, _, s in iter_prompt_blocks(t) if s)

    print(f"강의 자료 일관성 검사 — 노트 {len(notes)}개, 프롬프트 블록 {prompt_total}개 (검사 제외 {skipped_total}개)")
    print()

    if not findings:
        print("  이상 없음. 프롬프트 골격·차트 형식·HTML 동기화·대본 참조·사실 목록·실습 번호가 모두 일치합니다.")
        return 0

    by_check: dict[str, list[Finding]] = {}
    for f in findings:
        by_check.setdefault(f.check, []).append(f)
    for check, items in by_check.items():
        print(f"{check} — {len(items)}건")
        for f in items:
            print(f.render())
        print()

    fails = sum(1 for f in findings if f.level == "FAIL")
    warns = len(findings) - fails
    print(f"결과: FAIL {fails}건, WARN {warns}건")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
