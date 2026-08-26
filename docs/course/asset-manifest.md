# 강의·상세페이지 자산 Manifest

이 문서는 검증 결과 번들과 최종 전달 파일의 대응 관계를 기록합니다. 파일이 생성될 때 상태, 해시, 크기와 링크를 갱신합니다.

## 상태 정의

- `planned`: 스토리보드만 확정
- `generated`: 로컬 파일 생성
- `verified`: 수치·표현·파일 크기 검증
- `delivered`: 공유 링크를 XLSX에 반영

## 미국 WFA 기준 결과

| ID | 파일 | 용도 | 출처 번들 | 상태 |
|---|---|---|---|---|
| bundle-us-wfa | `artifacts/course_demo/bundles/us_wfa_verified/` | 모든 미국 WFA 시각화의 단일 데이터 원본 | `runners/run_wfa.py` | planned |
| chart-equity | `artifacts/course_demo/seminar/01_equity_curve.png/.svg` | 전체 누적수익 비교 | bundle-us-wfa | planned |
| chart-drawdown | `artifacts/course_demo/seminar/02_drawdown.png/.svg` | MDD 비교 | bundle-us-wfa | planned |
| chart-timeline | `artifacts/course_demo/seminar/03_wfa_timeline.png/.svg` | WFA와 regime 설명 | bundle-us-wfa | planned |
| chart-loss | `artifacts/course_demo/seminar/04_loss_period.png/.svg` | 대표 손실 구간 확대 | bundle-us-wfa | planned |
| chart-before-after | `artifacts/course_demo/seminar/05_before_after.png/.svg` | 허용 변경 전후 재검증 | bundle-us-wfa | planned |

## 상세페이지

| ID | 파일 | 화면 이야기 | 최대 크기 | 상태 | 공유 링크 |
|---|---|---|---:|---|---|
| gif-wfa-replay | `artifacts/course_demo/detail/01_wfa_replay.gif` | 워크포워드 진행 → bear regime → equity·drawdown | 2 MB | planned |  |
| gif-loss-review | `artifacts/course_demo/detail/02_loss_review.gif` | 손실 구간 정지 → 제한된 보고서 → 사용자 선택 → 비교 | 2 MB | planned |  |
| gif-paper-approval | `artifacts/course_demo/detail/03_paper_approval.gif` | 신호 → 승인 → 가상 매매 장부 체결 | 2 MB | planned |  |
| gif-korea-lab | `artifacts/course_demo/detail/04_korea_cash_regime.gif` | ETF 보유와 위험 신호 시 현금 대기 비교 | 2 MB | planned |  |

## 무료 세미나

| ID | 파일 | 발표 순서 | 핵심 메시지 | 상태 |
|---|---|---:|---|---|
| seminar-readme | `artifacts/course_demo/seminar/SPEAKER_NOTES.md` | 0 | 숫자보다 그래프에서 볼 질문을 먼저 제시 | planned |
| seminar-equity | `01_equity_curve.png/.svg` | 1 | 전체 기간에 전략 전환이 어떤 차이를 만들었는가 | planned |
| seminar-drawdown | `02_drawdown.png/.svg` | 2 | 수익률과 MDD를 함께 판단 | planned |
| seminar-timeline | `03_wfa_timeline.png/.svg` | 3 | 당시까지의 데이터로만 다음 구간을 평가 | planned |
| seminar-loss | `04_loss_period.png/.svg` | 4 | 손실은 실패가 아니라 분석 입력 | planned |
| seminar-before-after | `05_before_after.png/.svg` | 5 | 변경은 즉시 배포하지 않고 다시 검증 | planned |
| seminar-korea-map | `06_korea_extension.png/.svg` | 6 | 미국 검증 결과와 한국 적용 구조를 구분 | planned |

## 한국형 지수-현금 연구실

| ID | 파일 | 용도 | 상태 |
|---|---|---|---|
| bundle-korea-cash | `artifacts/course_demo/bundles/korea_cash_regime/` | ETF 보유·현금 대기 WFA 결과 | planned |
| korea-summary | `artifacts/course_demo/korea_cash_regime/summary.json` | 수익률·MDD·회복기간·참여율·전환·비용 | planned |

## 수강생 배포

| ID | 파일 | 포함 내용 | 상태 |
|---|---|---|---|
| student-guide | `artifacts/course_demo/student/시작_안내.md` | 실행, 프롬프트, 사람이 확인할 체크리스트 | planned |
| student-package | `artifacts/course_demo/ai-trading-course-demo.zip` | 오프라인 결과 재생 가능한 완성 폴더 | planned |

## 제작 메타데이터

최종 생성 시 다음 항목을 `artifacts/course_demo/manifest.json`에도 기계 판독 가능하게 기록합니다.

- 파일 경로와 SHA-256
- 생성일
- 파일 크기
- 이미지·영상 해상도와 프레임 수
- 출처 결과 번들의 설정 해시
- 데이터 기간과 거래비용 가정
- 검증 상태
