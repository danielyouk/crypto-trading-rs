"""강사 완성본(참고용 코드) ZIP 포장기 — 표준 라이브러리만 사용

촬영 PC의 작업 폴더(예: ~/ai-trading)를 받아, 계좌·비밀·기록 파일을 걷어낸 사본을 만들고
.env.example 과 README_시작하기.md 를 넣어 ZIP 으로 묶습니다. 남은 파일에서 비밀로 보이는
문자열을 훑어 보고서로 알려 주되, 파일을 고치지는 않습니다(사람이 확인).

실행 예:
    python runners/package_reference_code.py ~/ai-trading \
        --out artifacts/benefits/AI트레이딩_참고용_완성코드.zip \
        --map artifacts/benefits/script_clip_map.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path

# 통째로 빼는 폴더
EXCLUDE_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".claude",
                "logs", "price_cache", "replay", ".idea", ".vscode", "node_modules"}
# 빼는 파일(이름 패턴). 비밀·계좌 기록·개인 일지
EXCLUDE_FILES = [
    ".env", ".env.*", "*.env", ".kis_token.json", ".kis_rate.lock", "*.lock", "*.flag",
    "orders_history.json", "approvals_log.json", "orders_*.json", "signals_*.json", "signals.json",
    "holdings_*.json", "monitor_*.json", "day_start_asset.json", "daily_journal_*.md",
    "daily_performance_review_*.md", "*.log", "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db",
    "*.zip", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.mp4",
]
KEEP_EVEN_IF_MATCH = {".env.example"}

# 프롬프트가 파일 이름을 못 박지 않아 script_clip_map.json 에 없지만, 그 클립에서 만들어지는 파일
EXTRA_CLIP_MAP = {
    "run_trading.bat": [35], "run_trading.sh": [35],      # Clip 35: 내 컴퓨터에 맞는 실행 파일 두 개
    "daily_performance_review.py": [36],                   # Clip 36: 장 마감 후 매매 결과 분석
    "kis_auth_safety_check.py": [27],                      # Clip 27: 모의투자 전용 안전장치 확인
}

# 남은 텍스트 파일에서 비밀처럼 보이는 것
SECRET_PATTERNS = [
    (re.compile(r"(?i)(app\s*key|appkey|app\s*secret|appsecret|secret|token|password|passwd|pw)\s*[=:]\s*['\"]?[A-Za-z0-9+/=_-]{12,}"), "키·비밀번호 값으로 보임"),
    (re.compile(r"\b\d{8}-\d{2}\b"), "계좌번호 형식(8자리-2자리)"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "이메일 주소"),
    (re.compile(r"PS[A-Za-z0-9]{20,}"), "한국투자증권 앱키 형식"),
]
TEXT_EXT = {".py", ".md", ".txt", ".json", ".csv", ".bat", ".sh", ".ps1", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".html"}


def excluded(rel: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
        return True
    name = rel.name
    if name in KEEP_EVEN_IF_MATCH:
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_FILES)


def make_env_example(src_env: Path) -> str:
    lines = ["# 비밀값은 비워 두었습니다. 본인 값을 채운 뒤 파일 이름을 .env 로 바꾸세요.", ""]
    if not src_env.exists():
        lines += ["KIS_APP_KEY=", "KIS_APP_SECRET=", "KIS_ACCOUNT_NO=", "KIS_ACCOUNT_PRODUCT_CODE=01",
                  "KIS_MODE=paper", "GMAIL_ADDRESS=", "GMAIL_APP_PASSWORD=", "ALERT_TO="]
        return "\n".join(lines) + "\n"
    for raw in src_env.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            lines.append(raw)
            continue
        key = s.split("=", 1)[0].strip()
        lines.append(f"{key}=")
    return "\n".join(lines) + "\n"


def scan_secrets(root: Path) -> list[tuple[str, int, str, str]]:
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT or p.name == ".env.example":
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for pat, why in SECRET_PATTERNS:
                if pat.search(line):
                    hits.append((str(p.relative_to(root)), i, why, line.strip()[:120]))
                    break
    return hits


def build_readme(files: list[Path], clip_map: dict[str, list[int]], today: str) -> str:
    rows = []
    for f in sorted(files, key=lambda x: (x.suffix != ".py", str(x))):
        if f.suffix not in {".py", ".bat", ".sh", ".md"} or f.name.startswith("README"):
            continue
        clips = clip_map.get(f.name) or clip_map.get(str(f).replace("\\", "/")) or EXTRA_CLIP_MAP.get(f.name)
        if clips:
            made = f"Clip {clips[0]:02d}"
            used = ", ".join(f"{c:02d}" for c in clips[1:])
            rows.append(f"| `{f}` | {made} | {used} |")
        else:
            rows.append(f"| `{f}` | 핸드북 프롬프트에 없음(강사가 따로 만든 파일) | |")
    table = "\n".join(["| 파일 | 만든 클립 | 이후 쓰는 클립 |", "|---|---|---|"] + rows)
    return f"""# 참고용 완성 코드 — 시작하기

이 폴더는 강사가 강의를 촬영하며 **Claude Code에 요청해 만든 결과물의 한 예**입니다.
정답이 아닙니다. 같은 프롬프트를 보내도 AI가 쓰는 코드는 사람마다, 날마다 조금씩 다릅니다.
강의의 목표는 이 코드를 그대로 쓰는 것이 아니라, **본인이 프롬프트로 만들고 눈으로 확인하는 것**입니다.
막혔을 때 "강사는 어떻게 만들었나" 비교해 보는 용도로 쓰세요. 비교할 때도 코드가 같은지가 아니라
화면에 나온 결과(차트·표·알림·동작)가 각 클립의 [확인] 칸대로 맞는지를 보세요. (포장일: {today})

## 시작 순서

1. 압축을 풀고 VS Code에서 **폴더 열기**로 이 폴더를 엽니다.
2. `.env.example`을 복사해 `.env`로 이름을 바꾸고, 한국투자증권 모의투자 앱키·계좌번호와 Gmail 앱 비밀번호를 채웁니다(Clip 27, 32).
3. Claude Code 대화창에 이렇게 요청합니다.

```
[맥락] 이 폴더는 강의에서 받은 참고용 완성 코드야. 아직 실행 환경이 없어.
[만들 것] 가상환경(.venv)을 만들고 requirements.txt가 있으면 설치해줘. 없으면 각 .py가 쓰는 패키지를 찾아 requirements.txt를 만들고 설치해줘.
[하지 말 것] .env 값을 바꾸거나 증권사에 주문을 보내는 스크립트는 실행하지 마.
[확인] 설치가 끝나면 market_data.py 처럼 조회만 하는 스크립트 하나를 실행해 결과를 보여줘.
```

4. 그다음부터는 핸드북(교재)의 클립 순서대로, 각 클립의 프롬프트를 본인 폴더에서 직접 실행하는 것을 권합니다.

## 이 폴더에 없는 것 (일부러 뺐습니다)

- `.env`, `.kis_token.json` 같은 비밀값과 토큰
- `logs/`, `price_cache/`, 주문 목록·승인 기록·체결 이력·일지 등 강사 계좌의 기록
- 그래서 처음 실행하면 "이전 신호 파일이 없음" 같은 안내가 나오는 것이 정상입니다. Clip 29부터 순서대로 만들면 채워집니다.
- `docs/` 안의 일지·보고서·안내 문서는 프로그램이 실행될 때 만들어지는 결과물이라 넣지 않았습니다.
- Part 1~3(Clip 2~19)의 백테스트 스크립트는 이 묶음에 없습니다. 그 클립들은 프롬프트 한 번으로 완성되는 독립 스크립트라 본인이 만든 것을 쓰면 됩니다.

## 파일과 클립 대응

{table}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="강사 작업 폴더 (예: ~/ai-trading)")
    ap.add_argument("--out", required=True, help="만들 ZIP 경로")
    ap.add_argument("--map", help="script_clip_map.json (build_prompt_collection.py 가 만듦)")
    ap.add_argument("--inner-name", default="ai-trading", help="ZIP 안 최상위 폴더 이름")
    ap.add_argument("--notes", help="README 끝에 그대로 붙일 마크다운 파일(예: 교재와 다른 점)")
    args = ap.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.is_dir():
        print(f"폴더가 없습니다: {src}", file=sys.stderr)
        return 1
    clip_map = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else {}
    today = date.today().isoformat()

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / args.inner_name
        kept: list[Path] = []
        dropped: list[Path] = []
        for p in sorted(src.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(src)
            if excluded(rel):
                dropped.append(rel)
                continue
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            kept.append(rel)
        if not (stage / ".env.example").exists():
            (stage / ".env.example").write_text(make_env_example(src / ".env"), encoding="utf-8")
        readme = build_readme(kept, clip_map, today)
        if args.notes:
            readme += "\n" + Path(args.notes).read_text(encoding="utf-8").rstrip() + "\n"
        (stage / "README_시작하기.md").write_text(readme, encoding="utf-8")
        hits = scan_secrets(stage)

        out = Path(args.out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(stage.parent))

    print(f"ZIP: {out} ({out.stat().st_size // 1024} KB)")
    print(f"넣은 파일 {len(kept)}개, 뺀 파일 {len(dropped)}개")
    print("\n[뺀 파일]")
    for d in dropped:
        print("  -", d)
    print("\n[넣은 파일]")
    for k in kept:
        print("  +", k)
    if hits:
        print("\n⚠️ 비밀처럼 보이는 줄 — 사람이 확인하세요 (ZIP 에 그대로 들어 있습니다)")
        for f, ln, why, line in hits:
            print(f"  {f}:{ln}  [{why}]  {line}")
    else:
        print("\n비밀처럼 보이는 줄 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
