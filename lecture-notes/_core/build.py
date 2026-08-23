#!/usr/bin/env python3
"""강의 노트 재빌더 (수강생 실행용)

사용법:
    python _core/build.py
    (또는 _core 폴더 안에서 python build.py)

수강생 분들이 각 Part 폴더 안의 마크다운(.md) 파일을 수정한 뒤,
이 스크립트를 실행하면 즉시 HTML 교재가 나란히 업데이트됩니다.
"""

from pathlib import Path
import sys

# 프로젝트 루트 탐색
FILE_PATH = Path(__file__).resolve()
CORE_DIR = FILE_PATH.parent
LECTURE_DIR = CORE_DIR.parent
ROOT = LECTURE_DIR.parent

sys.path.insert(0, str(ROOT))

from runners.build_lecture_notes import build

if __name__ == "__main__":
    build()
