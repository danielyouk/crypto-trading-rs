# AI 트레이딩 강의 데모 운영 안내

## 정식 제작

```bash
source .venv/bin/activate
python runners/build_course_demo.py --full-run
```

한 명령이 다음 작업을 순서대로 수행합니다.

1. 고정된 미국 S&P 500 WFA 실행
2. 검증 결과 JSON 번들 저장
3. 세미나 PNG·SVG·발표 메모 생성
4. 상세페이지 GIF·MP4 생성
5. 한국형 KOSPI 200 ETF·현금 WFA 생성
6. 수강생 완성 폴더 ZIP 생성
7. `상세페이지 자료` 탭에 로컬 검증 링크 기록
8. 수치·해상도·GIF 2MB·ZIP 구성 검증

정식 결과 번들이 이미 있고 설정이 `us-sp500-course-v1`과 같다면 다음 명령은 장시간 WFA를 건너뛰고 시각 자료만 다시 생성합니다.

```bash
python runners/build_course_demo.py
```

## 강사용 화면

```bash
streamlit run dashboards/course_demo.py
```

화면은 네 가지로 분리됩니다.

- 검증 완료 결과 재생
- 30~60초 빠른 실제 계산
- 한국형 지수–현금 연구실
- 신호 승인과 가상 매매 장부

## 검증

```bash
pytest -q
python runners/verify_course_demo.py
```

검증 결과는 `artifacts/course_demo/verification-report.json`에 저장됩니다.

## 공개 표현

- 미국 WFA 결과를 한국 시장 실적으로 표현하지 않습니다.
- 재생 화면을 즉시 계산으로 표현하지 않습니다.
- 가상 매매 장부를 실제 API 주문으로 표현하지 않습니다.
- 모든 수익 그래프에는 과거 시뮬레이션 면책 문구를 표시합니다.
