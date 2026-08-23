"""강의 노트 빌더: lecture-notes/partXX/*.md → lecture-notes/partXX/*.html + lecture-notes/index.html

MD와 HTML이 각 part 폴더 안에 나란히 생성되도록 빌드합니다.
GDPval 스타일의 전문적인 에디토리얼 테마(DATATRAIN 브랜딩, 인라인 CSS 폴백, 프롬프트 카드)를 적용합니다.

실행:
    source .venv/bin/activate && python runners/build_lecture_notes.py
    (또는 lecture-notes 폴더에서 python build.py)
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent
LECTURE_DIR = ROOT / "lecture-notes"
CURRICULUM_PATH = LECTURE_DIR / "curriculum.json"
TEMPLATE_DIR = LECTURE_DIR / "template"
ASSETS_DIR = LECTURE_DIR / "assets"

MD_EXTENSIONS = ["extra", "sane_lists", "tables"]


@dataclass
class Note:
    source: Path
    part: int
    part_title: str
    chapter: int
    chapter_title: str
    clip: int
    title: str
    duration: int
    practice: str = ""
    body: str = ""
    href: str = field(default="", compare=False)

    @property
    def note_id(self) -> str:
        return f"p{self.part}-ch{self.chapter:02d}-c{self.clip:02d}"

    @property
    def sort_key(self) -> tuple[int, int, int]:
        return (self.part, self.chapter, self.clip)


def parse_note(path: Path, curriculum: dict[str, dict]) -> Note:
    meta = curriculum.get(path.stem)
    if meta is None:
        raise ValueError(f"curriculum.json에 없는 노트입니다: {path.stem} ({path})")

    body = path.read_text(encoding="utf-8").lstrip("\n")
    # 첫 줄의 H1 제목은 페이지 템플릿이 렌더링하므로 본문에서 제거한다.
    if body.startswith("# "):
        body = body.split("\n", 1)[1] if "\n" in body else ""

    return Note(source=path, body=body, **meta)


def render_markdown(body: str) -> str:
    html = markdown.markdown(body, extensions=MD_EXTENSIONS)

    # 1. ```prompt 블록 → 전용 카드 UI (복사 버튼 + 뱃지)
    prompt_pattern = re.compile(
        r'<pre><code class="language-prompt">(.*?)</code></pre>', re.DOTALL
    )
    def make_prompt_card(match: re.Match) -> str:
        code_content = match.group(1)
        return (
            '<div class="prompt-card">'
            '  <div class="prompt-card-header">'
            '    <span class="prompt-badge">AI 프롬프트 템플릿 (목표 중심)</span>'
            '    <button class="copy-btn">프롬프트 복사</button>'
            '  </div>'
            f'  <pre><code>{code_content}</code></pre>'
            '</div>'
        )
    html = prompt_pattern.sub(make_prompt_card, html)

    # 2. "- [ ] 항목" → 체크박스 리스트
    html = re.sub(
        r"<li>\[ \]\s*", '<li class="check-item"><input type="checkbox"> ', html
    )

    # 3. 이미지 슬롯 주석 → 세련된 비주얼 에셋 박스
    html = re.sub(
        r"<!--\s*IMAGE-SLOT:\s*(.*?)\s*-->",
        r'<div class="image-slot">\1</div>',
        html,
    )

    return html


def build() -> None:
    curriculum: dict[str, dict] = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    
    # part01 ~ part05 폴더 안의 .md 파일 검색
    md_files = [p for p in LECTURE_DIR.glob("part*/*.md")]
    notes = sorted(
        (parse_note(p, curriculum) for p in md_files),
        key=lambda n: n.sort_key,
    )
    if not notes:
        raise SystemExit(f"변환할 노트가 없습니다: {LECTURE_DIR}/part*/*.md")
        
    missing = set(curriculum) - {n.note_id for n in notes}
    if missing:
        raise SystemExit(f"curriculum.json에는 있으나 노트 파일이 없는 클립: {sorted(missing)}")

    for note in notes:
        note.href = f"part{note.part:02d}/{note.note_id}.html"

    # 템플릿 및 스타일 준비
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    page_tpl = env.get_template("page.html.j2")
    index_tpl = env.get_template("index.html.j2")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    css_content = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")
    (ASSETS_DIR / "style.css").write_text(css_content, encoding="utf-8")
    
    if (TEMPLATE_DIR / "logo.png").exists():
        shutil.copy2(TEMPLATE_DIR / "logo.png", ASSETS_DIR / "logo.png")

    # 개별 클립 페이지 렌더링
    for i, note in enumerate(notes):
        prev_note = notes[i - 1] if i > 0 else None
        next_note = notes[i + 1] if i < len(notes) - 1 else None
        out_path = LECTURE_DIR / note.href
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            page_tpl.render(
                note=note,
                content=render_markdown(note.body),
                css_path="../assets/style.css",
                inline_css=css_content,
                logo_path="../assets/logo.png",
                index_path="../index.html",
                prev={"href": f"../{prev_note.href}", "title": prev_note.title} if prev_note else None,
                next={"href": f"../{next_note.href}", "title": next_note.title} if next_note else None,
            ),
            encoding="utf-8",
        )

    # 목차 페이지 구성
    parts: list[dict] = []
    for note in notes:
        if not parts or parts[-1]["number"] != note.part:
            parts.append(
                {"number": note.part, "title": note.part_title, "chapters": [], "clip_count": 0, "minutes": 0}
            )
        part = parts[-1]
        if not part["chapters"] or part["chapters"][-1]["number"] != note.chapter:
            part["chapters"].append({"number": note.chapter, "title": note.chapter_title, "clips": []})
        part["chapters"][-1]["clips"].append(
            {
                "id": note.note_id,
                "title": note.title,
                "href": note.href,
                "duration": note.duration,
                "practice": note.practice,
            }
        )
        part["clip_count"] += 1
        part["minutes"] += note.duration

    (LECTURE_DIR / "index.html").write_text(
        index_tpl.render(
            parts=parts,
            css_path="assets/style.css",
            inline_css=css_content,
            total_clips=len(notes),
            total_minutes=sum(n.duration for n in notes),
        ),
        encoding="utf-8",
    )

    print(f"빌드 완료: 노트 {len(notes)}개 → {LECTURE_DIR}/partXX/*.html")
    print(f"전체 목차: {LECTURE_DIR / 'index.html'}")


if __name__ == "__main__":
    build()
