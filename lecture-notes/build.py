#!/usr/bin/env python3
"""강의 노트 빌더 (수강생 실행용)

사용법:
    python build.py

수강생 분들이 각 part 폴더 안의 마크다운(.md) 파일을 수정한 뒤,
이 스크립트를 실행하면 즉시 HTML 교재가 나란히 업데이트됩니다.
"""

from pathlib import Path
import sys

# 상위 폴더의 runners 모듈 또는 독립 실행 지원
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runners.build_lecture_notes import build

if __name__ == "__main__":
    build()
