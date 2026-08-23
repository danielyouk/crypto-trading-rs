# (Advanced) 주식시장 운영 시간에 맞춰 시스템을 자동으로 실행하기

## 이번 클립에서 만들 것

이 클립이 끝나면, 여러분은 리눅스/Mac의 **크론탭(Crontab)**이나 윈도우의 **작업 스케줄러(Task Scheduler)**를 활용하여, 한국 주식 시장의 운영 시간표에 맞춰 **아침 08:30 시그널 생성, 09:00~15:20 실시간 매매/하드 손절 감시, 15:40 일일 매매 일지 작성**을 100% 무인으로 스스로 실행하는 완전한 **'자동화 오케스트레이션(Automation Orchestration)'**을 구축하게 됩니다.

> 💡 **수강생 안심 안내**: 크론탭 설정이 낯설더라도 걱정하지 마세요. Claude Code에게 내 운영체제(Mac/Linux/Windows)를 알려주면 정확한 스케줄러 등록 명령어와 쉘 스크립트(`run_pipeline.sh`)를 자동으로 완성해 줍니다.

- 한국 주식 시장(KRX) 표준 운영 타임라인과 시스템 작업 맵핑
- 크론탭(Cron Expression) 문법과 파이썬 가상환경(`.venv`) 자동 연동 기법
- 단일 마스터 실행기 `run_pipeline.py` / `run_pipeline.sh`의 설계
- 로그 파일 자동 분할 및 일자별 보관(`logs/system_YYYYMMDD.log`)
- **실습 23**: 전체 파이프라인 무인 스케줄링 등록 및 통합 시뮬레이션

---

## 이론 핵심

### 1. 주식시장 타임테이블과 시스템 작업 맵핑

```text
[08:30] ──> ① 사전 점검 & 당일 시그널 생성 (generate_signals.py)
              • 토큰 갱신, 계좌 잔고 확인, 주간 정책(weekly_policy.json) 유효성 검사

[09:00] ──> ② 개장 즉시 시가 주문 및 변동성 돌파 감시 시작 (execute_orders.py)
              • 추세추종/자산배분 진입, 변동성 돌파 실시간 목표가 폴링

[09:00 ~ 15:20] ──> ③ 상시 하드 가드레일 감시 (risk_guard.py)
                      • 0.5초 간격 개별 종목 -3% 손절 및 포트폴리오 -2% 서킷브레이커 감시

[15:20] ──> ④ 데이트레이딩(VBO) 당일 포지션 종가 청산
[15:40] ──> ⑤ 장 마감 후 일일 AI 매매 일지 작성 (daily_journal.py)
```

---

### 2. 크론탭(Crontab) 설정 예시

```bash
# 월~금(1-5) 평일 자동 실행 스케줄
30 8 * * 1-5 /path/to/ai-trading/.venv/bin/python /path/to/ai-trading/generate_signals.py >> /path/to/ai-trading/logs/signals.log 2>&1
0 9 * * 1-5 /path/to/ai-trading/.venv/bin/python /path/to/ai-trading/execute_orders.py >> /path/to/ai-trading/logs/orders.log 2>&1
40 15 * * 1-5 /path/to/ai-trading/.venv/bin/python /path/to/ai-trading/daily_journal.py >> /path/to/ai-trading/logs/journal.log 2>&1
```

---

## 실습 — 실습 23: 무인 자동화 오케스트레이터 구축

Claude Code 대화창에 아래 프롬프트를 입력하세요.

```prompt
한국 주식 시장 운영 시간에 맞춰 전체 퀀트 시스템을 자동으로 스케줄링하는 마스터 오케스트레이터를 완성해줘. (Context)

다음 요구사항에 맞춰 스크립트를 작성하고 실행해줘: (Deliverable)
1. 아침 시그널 생성(08:30) ➔ 장중 매매/하드손절(09:00~15:20) ➔ 장 마감 일지(15:40)를 순차적으로 제어하는 마스터 파이썬 스크립트 'orchestrator.py'를 작성해줘.
2. 현재 OS 환경에 맞는 스케줄러 등록 가이드(Mac/Linux의 경우 crontab 파일, Windows의 경우 schtasks 스크립트)를 'docs/scheduler_setup.md'로 작성해줘.
3. 모든 터미널 출력과 에러가 날짜별 로그 파일('logs/system_YYYYMMDD.log')로 안전하게 자동 저장되도록 로깅 시스템을 구성해줘.
4. 전체 파이프라인의 1회 전체 순환 드라이런(Dry-run) 테스트를 실행하여 정상 작동을 증명해줘.

실행 완료 후 터미널에 드라이런 완료 로그를 출력하고 스케줄러 가이드 경로를 알려줘. (Verification)
```

---

### 내 눈으로 확인할 체크리스트

- [ ] 마스터 오케스트레이터(`orchestrator.py`)가 전체 모듈을 순서대로 에러 없이 호출했다.
- [ ] `logs/` 폴더에 날짜별 통합 시스템 로그가 기록되는 것을 확인했다.
- [ ] 평일 아침부터 장 마감까지 컴퓨터가 스스로 시스템을 안전하게 운용할 준비를 마쳤다.

---

## 다음 클립 예고

축하합니다! 이제 여러분은 완전한 무인 자동 모의투자 시스템을 가동할 수 있게 되었습니다.  
하지만 시스템을 지속적으로 운영하면서 자산을 불려 나가려면 **실전 운영에서의 비용 관리, 시장 레짐 체인지 대응, 그리고 1인 퀀트의 일간/주간/월간 루틴**이 뒷받침되어야 합니다.  
마지막 **Part 5 (결과 분석과 안전한 실전 운영)**에서 1인 퀀트 헤지펀드의 완벽한 마침표를 찍어보겠습니다!
