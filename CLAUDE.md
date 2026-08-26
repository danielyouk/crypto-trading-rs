# CLAUDE.md

이 저장소에는 여러 작업 영역이 있습니다. 이 문서는 그중 **강의 자료 제작**에 대해서만 다룹니다.
다른 영역(Rust 트레이딩 시스템, `python/`, `pairs_eda/`, WFA 실행기 등)은 여기 정리되어 있지 않으니 필요할 때 직접 확인하세요.

---

## 강의 자료를 만지기 전에 반드시 읽을 것

**[`docs/course/집필_원칙.md`](docs/course/집필_원칙.md)**

`lecture-notes/` 또는 `instructor-scripts/` 아래 파일을 새로 쓰거나 고칠 때는 **먼저 이 문서를 읽고 그 규칙을 따르세요.** 43개 클립 전체에 적용되는 규칙이며, 실제 사례가 함께 적혀 있습니다.

특히 자주 어기게 되는 것 두 가지:

1. **프롬프트에 정답을 심지 않는다** — 프롬프트에는 목표와 판단 기준만 씁니다. 강사가 이미 아는 해법(특정 파일 경로, 특정 함정의 이름)을 넣으면 그 순간은 통하지만 수강생은 아무것도 못 배웁니다. 그 이야기는 해설에 씁니다.
2. **정의 없이 용어를 쓰지 않는다** — 첫 등장 자리에서 한 줄로 풉니다. 뒤로 미루거나 별도 용어집으로 빼지 않습니다.

---

## 폴더 구조

| 경로 | 용도 | 공개 |
|---|---|---|
| `lecture-notes/PartXX_*/NN_*.md` | **수강생용 교재 원본.** 여기를 고칩니다 | GitHub Pages로 배포됨 |
| `lecture-notes/PartXX_*/NN_*.html` | 위 md에서 **생성된 결과물** | 배포됨 |
| `lecture-notes/_core/` | 템플릿·CSS·커리큘럼 정의 | — |
| `instructor-scripts/PartXX_*/NN_*.md` | **강사용 촬영 대본** (멘트, 구간별 시간) | 배포 안 됨 |
| `instructor-scripts/00_촬영_셋업_가이드.md` | OBS·마이크·Windows 촬영 환경 설정 | 배포 안 됨 |
| `docs/course/` | 집필 원칙, 커리큘럼 기획 문서 | 배포 안 됨 |

**교재를 고치면 대본도 같이 봐야 합니다.** Step 번호, 파일 이름, 프롬프트 내용이 양쪽에 중복되어 있어 한쪽만 고치면 어긋납니다.

---

## HTML은 직접 고치지 마세요

`.html` 파일은 **빌드 산출물**입니다. `.md`를 고친 뒤 반드시 재빌드하세요.

```bash
.venv/bin/python runners/build_lecture_notes.py
```

43개 노트와 전체 목차가 한 번에 생성됩니다. 재빌드를 빠뜨리면 커밋된 HTML이 md와 어긋난 채로 배포됩니다.

---

## 배포

`main`에 push하면 [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)이 GitHub Pages로 배포합니다.

**단, 아래 경로가 바뀐 경우에만 실행됩니다:**

```
lecture-notes/**
runners/build_lecture_notes.py
.github/workflows/deploy-pages.yml
```

그래서 `instructor-scripts/**`나 `docs/**`만 바꾼 커밋은 **실행 기록이 안 생기는 게 정상**입니다. "워크플로가 안 돌았다 = 고장"으로 판단하지 마세요.

배포되는 것은 `lecture-notes/` 폴더뿐입니다. 강사용 대본은 저장소에는 있지만 사이트에는 올라가지 않습니다.

Actions 실행 목록에 새 run이 **바로 안 잡힐 수 있습니다.** 등록이 늦을 뿐이니 몇 분 기다렸다 다시 확인하고, 성급하게 "트리거되지 않았다"고 결론 내리지 마세요.

---

## 사이트 확인

https://danielyouk.github.io/crypto-trading-rs/

내용이 실제로 반영됐는지 확인할 때는 브라우저 캐시를 피해 `curl`로 직접 받아 확인하는 편이 확실합니다.
