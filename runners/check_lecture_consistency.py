"""강의 자료 일관성 검사기: lecture-notes/ + instructor-scripts/ 전수 점검

강의를 매일 찍지 않다 보면 클립 간 형식이 조용히 어긋납니다. 이 스크립트는
"사람이 눈으로 43개를 다시 읽지 않아도" 잡히는 것들을 자동으로 잡습니다.

검사 항목
    P1  프롬프트 4칸 골격      ```prompt 블록에 [맥락] / [만들 것]|[수정할 것] / [하지 말 것] / [확인]이 모두 있는가
    P2  구형 표기 잔존         (Context) (Deliverable) (Verification) 같은 영문 괄호 표기가 남아 있는가
    C1  차트 형식              Plotly를 배운 클립 09 이후의 프롬프트가 .png 차트를 요구하는가 (→ .html이어야 함)
    H1  HTML 재빌드 누락        .md의 프롬프트 첫 줄이 같은 이름의 .html에 없으면 빌드를 빠뜨린 것
    S1  대본↔교재 파일명 불일치  instructor-scripts가 언급한 차트 파일(.html/.png)이 교재 어디에도 없는가

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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LECTURE_DIR = ROOT / "lecture-notes"
SCRIPT_DIR = ROOT / "instructor-scripts"

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


def first_content_line(body: str) -> str:
    for ln in body.splitlines():
        if ln.strip():
            return ln.strip()
    return ""


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
        probe = first_content_line(body)
        # 빌드 시 $...$ 는 수식 span으로 바뀌므로, 수식이 든 줄은 대조에서 제외
        if not probe or "$" in probe:
            continue
        if html.escape(probe, quote=False) not in rendered and probe not in rendered:
            findings.append(
                Finding("H1 HTML 재빌드", "FAIL", md, start, "프롬프트 첫 줄이 .html에 없음 → md 수정 후 재빌드 누락")
            )
            return  # 한 파일에 한 번만 보고


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

    check_script_refs("\n".join(note_texts.values()), findings)

    prompt_total = sum(len(list(iter_prompt_blocks(t))) for t in note_texts.values())
    skipped_total = sum(1 for t in note_texts.values() for _, _, s in iter_prompt_blocks(t) if s)

    print(f"강의 자료 일관성 검사 — 노트 {len(notes)}개, 프롬프트 블록 {prompt_total}개 (검사 제외 {skipped_total}개)")
    print()

    if not findings:
        print("  이상 없음. 43개 클립의 프롬프트 골격·차트 형식·HTML 동기화·대본 참조가 모두 일치합니다.")
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
