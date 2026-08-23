# 전체 구조 이해하기: 2-Tier 주간 정책 승인과 일간 자동 실행

## 이번 클립에서 만들 것

이 클립이 끝나면, 여러분은 1인 퀀트 투자자가 직장 생활이나 일상을 온전히 누리면서도 기관급 헤지펀드처럼 완벽한 규율로 자산을 운용할 수 있는 **'2-Tier 하이브리드 트레이딩 시스템(주간 거버넌스 + 일간 무인 자동 실행)'**의 전체 아키텍처와 6단계 파이프라인의 청사진을 완벽하게 이해하게 됩니다.

> 💡 **수강생 사전 안내**: 지금 당장 증권사 계좌를 켤 필요는 없습니다. 오늘은 우리가 구축할 실전 무인 매매 시스템의 **전체 설계도와 작동 규칙**을 뼈대부터 탄탄하게 세우는 시간입니다. 증권사 API 연동은 다음 클립(Clip 27)에서 단계별로 안전하게 진행합니다.

- 기존 100% 전자동 봇과 건별 메신저 승인 봇의 **3대 치명적 실패 원인**
- **Tier 1 (주간 전략 거버넌스)**: 주 1회 주말 30분, Claude Code와 함께 시장 국면을 진단하고 `weekly_policy.json` 최종 승인
- **Tier 2 (일간 무인 자동 실행 & 하드 가드레일)**: 평일 장중 인간 개입 없는 무인 자동 발주 및 0.1초 즉시 강제 손절
- 전체 6대 핵심 서브시스템 파이프라인 조감도

---

## 이론 핵심

### 1. 왜 '매 건마다 스마트폰 승인' 방식은 실패하는가?

시중의 많은 강의가 "매매 신호가 오면 텔레그램으로 알림을 보내고 사람이 버튼을 눌러 승인하는 봇"을 가르칩니다. 하지만 이는 실전에서 3가지 치명적 문제를 일으킵니다.

<div class="compare-box">
  <div class="compare-card bad">
    <div class="tag">❌ 건별 메신저 승인 방식의 함정</div>
    <div class="steps">
      • <strong>타이밍 지연 & 알림 피로</strong>: 회의나 운전 중에 알림을 놓쳐 진입 기회 상실.<br>
      • <strong>심리적 공포 개입</strong>: 연속 3번 손절 후 4번째 유효 신호가 왔을 때 무서워서 승인 거부 ➔ 이후 대세 상승을 놓침.<br>
      • <strong>노이즈 과적합</strong>: 장중 흔들림에 맞춰 매일 전략을 바꾸다 계좌 파탄.
    </div>
  </div>
  <div class="compare-card good">
    <div class="tag">✅ 2-Tier 주간 거버넌스 모델의 우수성</div>
    <div class="steps">
      • <strong>주말 30분 의사결정</strong>: 차분한 주말에 한 주의 전략 비중(Policy)을 승인.<br>
      • <strong>장중 100% 무인 집행</strong>: 승인된 룰에 따라 기계가 감정 없이 0.1초 만에 실행.<br>
      • <strong>하드 가드레일 안전망</strong>: 급락 시 사람에게 묻지 않고 기계가 즉각 손절.
    </div>
  </div>
</div>

---

### 2. 2-Tier 시스템의 역할 분담 체계

| 계층 (Tier) | 주체 | 실행 주기 | 구체적 담당 업무 |
|---|---|---|---|
| **Tier 1: 주간 거버넌스** | **사람 (CIO) + Claude Code** | **주 1회 (주말 30분)** | 거시 시장 국면 평가, 4대 전략 가중치 결정, 주간 정책(`weekly_policy.json`) 최종 승인 |
| **Tier 2: 무인 자동 실행** | 시스템 (실행 엔진) | 평일 장중 (09:00~15:30) | 승인된 정책에 따라 KIS Open API로 모의주문 자동 발주 (사람 개입 없음) |
| **Tier 2: 하드 가드레일** | 기계적 코드 규칙 | 상시 감시 (0.1초 주기) | 개별 종목 -3% 즉각 손절, 계좌 -2% 일일 서킷브레이커 강제 청산 (질문 없이 집행) |
| **Post-Market: 일일 저널** | AI 리포터 | 장 마감 후 (15:40) | 오늘의 체결·손익 분석 및 일일 매매 일지(`daily_journal.md`) 자동 작성 |

---

### 3. 실전 시스템 6대 서브시스템 파이프라인

<div class="arch-grid">
  <div class="arch-card">
    <div class="card-title">① 계좌 인증 & 연결 (Auth)</div>
    <div class="card-desc">한국투자증권 Open API 모의투자 계좌 인증 및 OAuth2 토큰 자동 갱신 (Clip 27)</div>
  </div>
  <div class="arch-card">
    <div class="card-title">② 시장 데이터 파이프라인 (Data)</div>
    <div class="card-desc">유니버스 종목들의 실시간 시세 및 기술적 지표 계산 (Clip 28)</div>
  </div>
  <div class="arch-card">
    <div class="card-title">③ 주간 정책 합의 (Governance)</div>
    <div class="card-desc">Claude Code와 주간 국면 분석 후 <code>weekly_policy.json</code> 서명 승인 (Clip 29~30)</div>
  </div>
  <div class="arch-card">
    <div class="card-title">④ 무인 자동 발주 엔진 (Execution)</div>
    <div class="card-desc">승인된 조건 만족 시 KIS API로 주문 전송 및 체결 확인 (Clip 31)</div>
  </div>
  <div class="arch-card">
    <div class="card-title">⑤ 하드 안전 가드레일 (Safety)</div>
    <div class="card-desc">0.1초 즉시 손절 및 일일 서킷브레이커, 킬스위치 비상 정지 (Clip 32~33)</div>
  </div>
  <div class="arch-card">
    <div class="card-title">⑥ 일일 AI 저널링 (Journal)</div>
    <div class="card-desc">장 마감 후 AI가 체결 내역 및 손익을 분석해 자동 일지 기록 (Clip 34~35)</div>
  </div>
</div>

---

## 관전 포인트 및 체크리스트

- [ ] 주간 전략 정책 승인(Tier 1)과 장중 무인 실행(Tier 2)의 분리 원리를 이해했다.
- [ ] 장중 급락 시 시스템이 사람에게 승인을 묻지 않고 즉시 하드 손절을 집행해야 하는 이유를 파악했다.
- [ ] 6단계 파이프라인이 앞으로 Part 4에서 어떻게 구현될지 전체 흐름을 확인했다.

---

## 다음 클립 예고

전체 아키텍처를 이해했습니다.  
이제 첫 번째 엔지니어링 단계로, **한국투자증권 Open API에 접속하여 모의투자 계좌를 연결하고 인증 토큰을 발급받아 내 계좌의 잔고를 조회**해 보겠습니다.
