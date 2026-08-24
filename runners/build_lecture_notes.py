"""강의 노트 빌더: lecture-notes/PartXX_*/*.md → lecture-notes/PartXX_*/*.html + lecture-notes/00_AI트레이딩_전체교재_목차.html

MD와 HTML이 각 한국어 Part 폴더 안에 나란히 생성되도록 빌드합니다.
내부 설정/템플릿/에셋은 lecture-notes/_core/ 폴더에서 중앙 관리합니다.
GDPval 스타일의 전문적인 에디토리얼 테마(DATATRAIN Base64 로고, 인라인 CSS, GIF 뷰어, 프롬프트 카드)를 적용합니다.

실행:
    source .venv/bin/activate && python runners/build_lecture_notes.py
    (또는 lecture-notes 폴더에서 python _core/build.py)
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent.parent
LECTURE_DIR = ROOT / "lecture-notes"
CORE_DIR = LECTURE_DIR / "_core"
CURRICULUM_PATH = CORE_DIR / "curriculum.json"
TEMPLATE_DIR = CORE_DIR / "template"
ASSETS_DIR = CORE_DIR / "assets"
INDEX_FILENAME = "00_AI트레이딩_전체교재_목차.html"

MD_EXTENSIONS = ["extra", "sane_lists", "tables"]


@dataclass
class Note:
    source: Path
    part: int
    part_title: str
    part_folder: str
    chapter: int
    chapter_title: str
    clip: int
    title: str
    filename: str
    duration: int
    clip_num: int = 0
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

    # 3. 이미지 슬롯 주석 → 에셋이 존재할 경우 실제 GIF/이미지 figure로 렌더링
    def make_image_figure(match: re.Match) -> str:
        raw = match.group(1).strip()
        parts = raw.split("—", 1) if "—" in raw else raw.split("-", 1)
        file_part = parts[0].strip()
        caption = parts[1].strip() if len(parts) > 1 else ""
        filename = Path(file_part).name
        asset_file = ASSETS_DIR / filename
        if asset_file.exists():
            return (
                f'<figure class="visual-figure">'
                f'  <img src="../_core/assets/{filename}" alt="{caption}">'
                f'  <figcaption><strong>[시스템 조감도]</strong> {caption}</figcaption>'
                f'</figure>'
            )
        return f'<div class="image-slot">{raw}</div>'

    html = re.sub(
        r"<!--\s*IMAGE-SLOT:\s*(.*?)\s*-->",
        make_image_figure,
        html,
    )

    return html


def get_logo_data_uri() -> str:
    logo_file = ASSETS_DIR / "logo.png"
    if not logo_file.exists() and (TEMPLATE_DIR / "logo.png").exists():
        shutil.copy2(TEMPLATE_DIR / "logo.png", logo_file)
    if logo_file.exists():
        b64 = base64.b64encode(logo_file.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    return ""


def build() -> None:
    if not CURRICULUM_PATH.exists():
        raise SystemExit(f"curriculum.json을 찾을 수 없습니다: {CURRICULUM_PATH}")

    curriculum: dict[str, dict] = json.loads(CURRICULUM_PATH.read_text(encoding="utf-8"))
    
    # Part01_* ~ Part05_* 폴더 안의 .md 파일 검색
    md_files = [p for p in LECTURE_DIR.glob("Part*/*.md")]
    notes = sorted(
        (parse_note(p, curriculum) for p in md_files),
        key=lambda n: n.sort_key,
    )
    if not notes:
        raise SystemExit(f"변환할 노트가 없습니다: {LECTURE_DIR}/Part*/*.md")
        
    missing = set(curriculum) - {n.filename for n in notes}
    if missing:
        raise SystemExit(f"curriculum.json에는 있으나 노트 파일이 없는 클립: {sorted(missing)}")

    for note in notes:
        note.href = f"{note.part_folder}/{note.filename}.html"

    # 템플릿 및 스타일 준비
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    page_tpl = env.get_template("page.html.j2")
    index_tpl = env.get_template("index.html.j2")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    css_content = (TEMPLATE_DIR / "style.css").read_text(encoding="utf-8")
    (ASSETS_DIR / "style.css").write_text(css_content, encoding="utf-8")
    logo_data_uri = get_logo_data_uri()

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
                css_path="../_core/assets/style.css",
                inline_css=css_content,
                logo_data_uri=logo_data_uri,
                index_path=f"../{INDEX_FILENAME}",
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
                "clip_num": note.clip_num,
                "title": note.title,
                "href": note.href,
                "duration": note.duration,
                "practice": note.practice,
            }
        )
        part["clip_count"] += 1
        part["minutes"] += note.duration

    out_index_path = LECTURE_DIR / INDEX_FILENAME
    out_index_path.write_text(
        index_tpl.render(
            parts=parts,
            css_path="_core/assets/style.css",
            inline_css=css_content,
            logo_data_uri=logo_data_uri,
            total_clips=len(notes),
            total_minutes=sum(n.duration for n in notes),
            index_filename=INDEX_FILENAME,
        ),
        encoding="utf-8",
    )

    # 브라우저/웹서버가 / 로 열 수 있도록 index.html도 생성
    (LECTURE_DIR / "index.html").write_text(
        out_index_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # GitHub Pages가 _core/ 폴더를 무시하지 않도록 .nojekyll 파일 생성
    (LECTURE_DIR / ".nojekyll").touch(exist_ok=True)

    print(f"빌드 완료: 노트 {len(notes)}개 → {LECTURE_DIR}/PartXX_*/*.html")
    print(f"전체 목차: {out_index_path}")


if __name__ == "__main__":
    build()
