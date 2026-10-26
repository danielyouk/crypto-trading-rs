"""실습 프롬프트 모음집 빌더: lecture-notes/PartXX_*/*.md 의 ```prompt 블록을 한 권으로 묶습니다.

산출물(기본 artifacts/benefits/):
    AI트레이딩_실습_프롬프트_모음.md   복사용
    AI트레이딩_실습_프롬프트_모음.pdf  열람용 (Windows Chrome 헤드리스로 변환, 없으면 LibreOffice)
    script_clip_map.json              프롬프트에 등장하는 스크립트 이름 → 클립 번호 (package_reference_code.py 가 씀)
    클립별_txt/ClipNN_제목.txt          클립 하나에 프롬프트만 담은 복붙용 텍스트
    AI트레이딩_부록_실습_프롬프트_모음.zip  위 PDF·md·클립별 txt·README 를 묶은 전달용 ZIP

실행:
    .venv/bin/python runners/build_prompt_collection.py [--out-dir artifacts/benefits] [--keep-html]
교재를 고치면 다시 실행하기만 하면 됩니다.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import subprocess
import zipfile
from datetime import date
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
LECTURE_DIR = ROOT / "lecture-notes"
CURRICULUM = LECTURE_DIR / "_core" / "curriculum.json"
SITE = "https://danielyouk.github.io/crypto-trading-rs/"
CHROME_CANDIDATES = [
    Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    Path("/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe"),
]
HEADS = ["[맥락]", "[만들 것]", "[수정할 것]", "[하지 말 것]", "[확인]"]
SCRIPT_RE = re.compile(r"(?<![\w/])((?:docs/)?[\w\-]+\.(?:py|bat|sh|md))\b")


def load_notes() -> list[dict]:
    cur = json.loads(CURRICULUM.read_text(encoding="utf-8"))
    notes = []
    for stem, meta in cur.items():
        path = LECTURE_DIR / meta["part_folder"] / f"{stem}.md"
        if not path.exists():
            raise FileNotFoundError(path)
        notes.append({**meta, "path": path})
    notes.sort(key=lambda n: n["clip_num"])
    return notes


def extract_prompts(text: str) -> list[dict]:
    """프롬프트 블록마다 {heading, notes, skip, body} 를 돌려준다."""
    out: list[dict] = []
    heading = ""
    notes: list[str] = []
    skip_reason = ""
    in_prompt = False
    buf: list[str] = []
    for line in text.splitlines():
        if in_prompt:
            if line.strip().startswith("```"):
                out.append({"heading": heading, "notes": notes, "skip": skip_reason, "body": "\n".join(buf).strip("\n")})
                notes, skip_reason, buf, in_prompt = [], "", [], False
            else:
                buf.append(line)
            continue
        s = line.strip()
        if s.startswith("```prompt"):
            in_prompt = True
            continue
        if s.startswith("#"):
            heading = s.lstrip("#").strip()
            notes = []
            continue
        m = re.match(r"<!--\s*prompt-check:\s*skip\s*[—-]*\s*(.*?)\s*-->", s)
        if m:
            skip_reason = m.group(1) or "골격 연습용 조각"
            continue
        if s.startswith(">") and ("📁" in s or "📌" in s):
            notes.append(re.sub(r"^>\s*", "", s))
    return out


def md_inline_to_html(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    return s


def prompt_to_html(body: str) -> str:
    esc = html.escape(body)
    for h in HEADS:
        esc = esc.replace(html.escape(h), f'<b class="head">{html.escape(h)}</b>')
    return esc


CSS = """
@page { size: A4; margin: 18mm 16mm 20mm 16mm; }
body { font-family: "Malgun Gothic", "맑은 고딕", sans-serif; color: #1f2937; font-size: 10.5pt; line-height: 1.55; }
h1, h2, h3, h4 { color: #1e3a8a; margin: 0; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: center; }
.cover .brand { color: #6b7280; font-size: 11pt; letter-spacing: .08em; margin-bottom: 18mm; }
.cover h1 { font-size: 30pt; line-height: 1.3; }
.cover .sub { font-size: 14pt; color: #374151; margin-top: 8mm; }
.cover .meta { color: #6b7280; margin-top: 22mm; font-size: 10pt; }
.part { break-before: page; }
.part > h2 { font-size: 20pt; padding-bottom: 4mm; border-bottom: 3px solid #1e3a8a; margin-bottom: 8mm; }
.clip { margin-top: 10mm; }
.clip > h3, .clip h4 { break-after: avoid; }
.clip > h3 { font-size: 14pt; }
.clip .link { color: #6b7280; font-size: 9pt; margin: 1mm 0 3mm; word-break: break-all; }
.clip .link a { color: #6b7280; text-decoration: none; }
.clip h4 { font-size: 11pt; color: #374151; margin: 4mm 0 2mm; }
.note { background: #eff6ff; border-left: 4px solid #60a5fa; padding: 2mm 3mm; margin: 2mm 0; font-size: 9.5pt; color: #1e3a8a; }
.skip { color: #9a3412; font-size: 9.5pt; margin: 2mm 0; }
pre.prompt { white-space: pre-wrap; word-break: break-all; font-family: "Malgun Gothic", sans-serif; font-size: 10pt;
  background: #f8fafc; border: 1px solid #cbd5e1; border-left: 5px solid #1e3a8a; border-radius: 4px; padding: 4mm 5mm; margin: 2mm 0 4mm; }
pre.prompt b.head { color: #1d4ed8; }
code { font-family: Consolas, "Malgun Gothic", monospace; font-size: 9.5pt; background: #f1f5f9; padding: 0 2px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 3mm 0; }
th, td { border: 1px solid #e5e7eb; padding: 1.6mm 2.5mm; text-align: left; vertical-align: top; }
th { background: #f1f5f9; }
.howto h2 { font-size: 18pt; margin-bottom: 5mm; }
.howto h3 { font-size: 12.5pt; margin: 6mm 0 2mm; }
.howto p, .howto li { margin: 1.2mm 0; }
.four td:first-child { white-space: nowrap; font-weight: bold; color: #1d4ed8; width: 24mm; }
"""

HOWTO_MD = '''## 읽는 법

이 모음집은 핸드북(교재)의 36개 클립에 들어 있는 실습 프롬프트 47개를 **글자 그대로** 옮긴 것입니다. 핸드북의 각 클립 페이지 '실습' 절에도 같은 프롬프트가 '프롬프트 복사' 버튼과 함께 있습니다.

### 네 칸 골격

모든 프롬프트는 Clip 03에서 배운 네 칸으로 되어 있습니다. 새로 만들 때는 `[맥락] → [만들 것] → [하지 말 것] → [확인]`, 이미 있는 파일을 고칠 때는 둘째 칸만 `[수정할 것]`으로 바뀝니다.

| 칸 | 적는 것 |
|---|---|
| [맥락] | 지금 어떤 작업을 왜 하는지, 어떤 파일이 이미 있는지 |
| [만들 것] / [수정할 것] | 작업이 끝났을 때 화면에 무엇이 어떻게 보여야 하는지 (방법이 아니라 결과) |
| [하지 말 것] | AI가 흔히 저지르는 우회(가짜 값 넣기, 되던 부분 고치기, 주문 보내기 등)를 막는 규칙 |
| [확인] | 사람이 눈으로 볼 것. 확인을 AI에게 맡기지 않습니다 |

### 쓰는 요령

1. **클립 순서대로** 쓰세요. 뒤 클립의 프롬프트는 앞 클립에서 만든 파일(예: `top3_drawdowns.csv`, `signals.json`)이 있다고 전제합니다. 그런 파일은 프롬프트 위의 📁 메모에 적어 두었습니다.
2. 프롬프트 속 **날짜·종목·숫자는 본인 화면에 맞게** 바꾸세요. 특히 `"""` 사이에는 본인 화면의 표나 오류 로그를 그대로 붙여 넣습니다.
3. 결과가 다르거나 오류가 나면 새 프롬프트를 처음부터 쓰지 말고, Clip 05에서 배운 **후속 수정 네 칸**으로 고칩니다.
4. 영상과 교재가 다르면 **교재(핸드북)가 최신**입니다. 이 모음집은 교재에서 뽑았습니다.
5. 같은 프롬프트를 보내도 **AI가 쓰는 코드는 사람마다, 날마다 다를 수 있습니다.** 강사 화면과 코드가 같은지가 아니라, 화면에 나온 결과(차트·표·알림·동작)가 [확인] 칸대로 맞는지로 판단하세요.
6. 증권사에 실제로 주문을 보내는 프롬프트(Part 4·5)는 **모의투자 계좌**를 전제로 합니다.
'''


def build(out_dir: Path, keep_html: bool) -> None:
    notes = load_notes()
    today = date.today().isoformat()
    parts: dict[int, dict] = {}
    script_map: dict[str, list[int]] = {}
    total = 0
    for n in notes:
        prompts = extract_prompts(n["path"].read_text(encoding="utf-8"))
        if not prompts:
            continue
        total += len(prompts)
        for p in prompts:
            for m in SCRIPT_RE.finditer(p["body"]):
                name = m.group(1)
                if name.endswith(".md") and not name.startswith("docs/"):
                    continue
                script_map.setdefault(name, [])
                if n["clip_num"] not in script_map[name]:
                    script_map[name].append(n["clip_num"])
        href = f"{SITE}{quote(n['part_folder'])}/{quote(n['filename'])}.html"
        href_text = f"{SITE}{n['part_folder']}/{n['filename']}.html"
        parts.setdefault(n["part"], {"title": n["part_title"], "clips": []})
        parts[n["part"]]["clips"].append({**n, "prompts": prompts, "href": href, "href_text": href_text})

    clip_count = sum(len(p["clips"]) for p in parts.values())
    title = "AI 트레이딩 시스템 만들기: 실습 프롬프트 모음"
    sub = f"핸드북 {clip_count}개 클립의 프롬프트 {total}개"

    # ---------- Markdown ----------
    md: list[str] = [f"# {title}", "", f"{sub} · 만든 날 {today} · 핸드북 {SITE}", "", HOWTO_MD.replace("36개 클립", f"{clip_count}개 클립").replace("47개", f"{total}개"), "## 차례", "", "| Part | Clip | 제목 | 프롬프트 |", "|---|---|---|---|"]
    for pn, part in parts.items():
        for c in part["clips"]:
            md.append(f"| {pn} | {c['clip_num']:02d} | {c['title']} | {len(c['prompts'])} |")
    md.append("")
    for pn, part in parts.items():
        md += [f"## Part {pn}. {part['title']}", ""]
        for c in part["clips"]:
            md += [f"### Clip {c['clip_num']:02d}. {c['title']}", "", f"핸드북: {c['href_text']}", ""]
            for i, p in enumerate(c["prompts"], 1):
                label = re.sub(r"^실습\s*[—-]\s*", "", p["heading"]) if p["heading"] else f"프롬프트 {i}"
                if len(c["prompts"]) > 1:
                    label = f"{label} ({i}/{len(c['prompts'])})"
                md += [f"#### {label}", ""]
                for note in p["notes"]:
                    md += [f"> {note}", ""]
                if p["skip"]:
                    md += [f"*({p['skip']})*", ""]
                md += ["```text", p["body"], "```", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "AI트레이딩_실습_프롬프트_모음.md"
    md_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    (out_dir / "script_clip_map.json").write_text(json.dumps(script_map, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- HTML → PDF ----------
    import markdown  # 교재 빌더와 같은 패키지

    howto_html = markdown.markdown(HOWTO_MD.replace("36개 클립", f"{clip_count}개 클립").replace("47개", f"{total}개"), extensions=["tables"])
    howto_html = howto_html.replace("<table>", '<table class="four">', 1)
    h: list[str] = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>"]
    h.append(f"<section class='cover'><div class='brand'>AI 트레이딩 강의 핸드북 부록</div><h1>{html.escape(title)}</h1>"
             f"<div class='sub'>{html.escape(sub)}</div><div class='meta'>만든 날 {today}<br>핸드북 {SITE}<br>교재가 바뀌면 이 모음집도 다시 만듭니다.</div></section>")
    h.append(f"<section class='part howto'>{howto_html}</section>")
    h.append("<section class='part'><h2>차례</h2><table><tr><th>Part</th><th>Clip</th><th>제목</th><th>프롬프트</th></tr>")
    for pn, part in parts.items():
        for c in part["clips"]:
            h.append(f"<tr><td>{pn}</td><td>{c['clip_num']:02d}</td><td>{html.escape(c['title'])}</td><td>{len(c['prompts'])}</td></tr>")
    h.append("</table></section>")
    for pn, part in parts.items():
        h.append(f"<section class='part'><h2>Part {pn}. {html.escape(part['title'])}</h2>")
        for c in part["clips"]:
            h.append(f"<div class='clip'><h3>Clip {c['clip_num']:02d}. {html.escape(c['title'])}</h3><div class='link'>핸드북 <a href='{c["href"]}'>{html.escape(c["href_text"])}</a></div>")
            for i, p in enumerate(c["prompts"], 1):
                label = re.sub(r"^실습\s*[—-]\s*", "", p["heading"]) if p["heading"] else f"프롬프트 {i}"
                if len(c["prompts"]) > 1:
                    label = f"{label} ({i}/{len(c['prompts'])})"
                h.append(f"<h4>{html.escape(label)}</h4>")
                for note in p["notes"]:
                    h.append(f"<div class='note'>{md_inline_to_html(note)}</div>")
                if p["skip"]:
                    h.append(f"<div class='skip'>({html.escape(p['skip'])})</div>")
                h.append(f"<pre class='prompt'>{prompt_to_html(p['body'])}</pre>")
            h.append("</div>")
        h.append("</section>")
    h.append("</body></html>")
    html_path = out_dir / "AI트레이딩_실습_프롬프트_모음.html"
    html_path.write_text("\n".join(h), encoding="utf-8")
    pdf_path = out_dir / "AI트레이딩_실습_프롬프트_모음.pdf"
    to_pdf(html_path, pdf_path)
    if not keep_html:
        html_path.unlink(missing_ok=True)

    # ---------- 클립별 txt + README + ZIP ----------
    txt_dir = out_dir / "클립별_txt"
    if txt_dir.exists():
        shutil.rmtree(txt_dir)
    txt_dir.mkdir()
    for pn, part in parts.items():
        for c in part["clips"]:
            lines = [f"Clip {c['clip_num']:02d}. {c['title']}", f"Part {pn}. {part['title']}", f"핸드북: {c['href_text']}", "", "=" * 60, ""]
            for i, p in enumerate(c["prompts"], 1):
                label = re.sub(r"^실습\s*[—-]\s*", "", p["heading"]) if p["heading"] else f"프롬프트 {i}"
                lines.append(f"[{i}/{len(c['prompts'])}] {label}")
                for note in p["notes"]:
                    lines.append(re.sub(r"\*\*|`", "", note))
                if p["skip"]:
                    lines.append(f"({p['skip']})")
                lines += ["", "-" * 60, p["body"], "-" * 60, ""]
            safe = re.sub(r"[\\/:*?\"<>|]", "_", c["title"]).replace(" ", "_")
            (txt_dir / f"Clip{c['clip_num']:02d}_{safe}.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8-sig")

    readme = f"""AI 트레이딩 시스템 만들기: 실습 프롬프트 모음 (만든 날 {today})

들어 있는 것
  AI트레이딩_실습_프롬프트_모음.pdf  열람용. 읽는 법(네 칸 골격), 차례, Part·클립 순서의 프롬프트 전문
  AI트레이딩_실습_프롬프트_모음.md   같은 내용의 복사용(마크다운)
  클립별_txt/                        클립 하나에 그 클립의 프롬프트만 담은 텍스트 {clip_count}개. 강의 클립마다 붙이기 좋습니다

쓰는 법
  1. 핸드북({SITE})의 각 클립 페이지 '실습' 절에도 같은 프롬프트가 '프롬프트 복사' 버튼과 함께 있습니다.
  2. 프롬프트는 클립 순서대로 쓰세요. 뒤 클립은 앞 클립에서 만든 파일이 있다고 전제합니다.
  3. 날짜·종목·숫자는 본인 화면에 맞게 바꾸고, \"\"\" 사이에는 본인 화면의 표나 오류 로그를 붙여 넣습니다.
  4. 영상과 다르면 교재(핸드북)가 최신입니다. 이 모음집은 교재에서 뽑았습니다.
  5. 같은 프롬프트를 보내도 AI가 쓰는 코드는 사람마다, 날마다 다를 수 있습니다. 코드가 강사와 같은지가 아니라 화면 결과(차트·표·알림·동작)가 맞는지로 판단하세요.
  6. 증권사에 주문을 보내는 프롬프트(Part 4·5)는 모의투자 계좌를 전제로 합니다.
"""
    (out_dir / "README.txt").write_text(readme, encoding="utf-8-sig")

    zip_path = out_dir / "AI트레이딩_부록_실습_프롬프트_모음.zip"
    inner = "AI트레이딩_실습_프롬프트_모음"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (out_dir / "README.txt", pdf_path, md_path):
            zf.write(f, f"{inner}/{f.name}")
        for f in sorted(txt_dir.iterdir()):
            zf.write(f, f"{inner}/클립별_txt/{f.name}")
    print(f"클립 {clip_count}개, 프롬프트 {total}개 → {md_path.name}, {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB), "
          f"클립별 txt {clip_count}개, script_clip_map.json({len(script_map)}개 이름)")
    print(f"전달용 ZIP: {zip_path} ({zip_path.stat().st_size // 1024} KB)")


def to_win(p: Path) -> str:
    s = str(p.resolve())
    m = re.match(r"^/mnt/([a-z])/(.*)$", s)
    return f"{m.group(1).upper()}:\\{m.group(2).replace('/', chr(92))}" if m else s


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = next((c for c in CHROME_CANDIDATES if c.exists()), None)
    if chrome and str(html_path.resolve()).startswith("/mnt/"):
        url = "file:///" + to_win(html_path).replace("\\", "/")
        subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={to_win(pdf_path)}", url], check=True, capture_output=True, timeout=180)
        return
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("PDF 변환 도구가 없습니다 (Windows Chrome/Edge 또는 LibreOffice 필요)")
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(pdf_path.parent), str(html_path)], check=True, capture_output=True, timeout=300)
    produced = pdf_path.parent / (html_path.stem + ".pdf")
    if produced != pdf_path:
        produced.replace(pdf_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(ROOT / "artifacts" / "benefits"))
    ap.add_argument("--keep-html", action="store_true")
    a = ap.parse_args()
    build(Path(a.out_dir), a.keep_html)
