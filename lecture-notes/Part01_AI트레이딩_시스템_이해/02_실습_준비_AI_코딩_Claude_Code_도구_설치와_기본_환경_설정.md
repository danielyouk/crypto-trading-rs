# 실습 준비: AI 코딩 Claude Code 도구 설치와 기본 환경 설정

## 이번 클립에서 만들 것

이 클립이 끝나면, 여러분의 컴퓨터에 앞으로 5개 파트 동안 나만의 트레이딩 시스템을 함께 만들 **강력한 AI 페어 프로그래머(Claude Code)**와 **한국 주식 데이터 분석 환경**이 완벽하게 갖춰집니다.

> 💡 **수강생 안심 안내**: 복잡한 파이썬 버전 관리나 패키지 충돌 때문에 걱정하실 필요가 없습니다. 우리는 파이썬 패키지를 하나하나 수동으로 설치하지 않습니다. **개발 환경 구축 자체를 Claude Code에게 위임**하고, 우리는 화면으로 정상 작동 여부만 시각적으로 확인합니다.

- 터미널 기반 자율 AI 에이전트 **Claude Code의 핵심 작동 메커니즘** 이해
- **"어떤 AI 도구를 써도 본질은 같다"** — 범용 도구 철학과 단일 작업 파이프라인
- Claude Code 설치 및 인증, 투명한 비용 관리(API vs 정액제) 이해
- Claude Code를 통해 **Python 가상환경(`.venv`)과 FinanceDataReader 설치**를 완전 자동화
- **실습 01**: 삼성전자(005930) 최근 종가 및 데이터프레임 무결성 출력 검증

---

## 이론 핵심

### 1. 왜 Claude Code인가? — 터미널 기반 에이전트의 혁신

AI 코딩 도구는 크게 웹 채팅(ChatGPT, Claude.ai), 에디터 통합형(Cursor, Copilot), 그리고 **자율 터미널 에이전트(Claude Code)**로 발전해 왔습니다.

<div class="compare-box">
  <div class="compare-card info">
    <div class="tag">기존 웹 채팅 & 에디터 확장</div>
    <div class="steps">
      • 코드를 복사해서 파이썬 파일에 직접 붙여넣어야 함.<br>
      • 패키지 설치, 파일 생성, 실행, 디버깅을 사람이 수동 반복.<br>
      • 폴더 구조가 커지면 전체 맥락(Context)을 놓치기 쉬움.
    </div>
  </div>
  <div class="compare-card good">
    <div class="tag">Claude Code (자율 터미널 에이전트)</div>
    <div class="steps">
      • <strong>파일 생성/수정/실행/검증을 스스로 자율 수행</strong>.<br>
      • 프로젝트 전체 파일 구조를 한눈에 파악하고 스스로 테스트.<br>
      • 오류가 나면 터미널 출력을 보고 스스로 수정(Self-healing).
    </div>
  </div>
</div>

---

### 2. 범용 도구 철학 — 어떤 AI 도구를 써도 본질은 동일하다

이 강의는 모든 실습을 **Claude Code** 기준으로 명료하게 진행합니다. 하지만 여러분이 기억해야 할 가장 중요한 원칙은 **"도구는 바뀔 수 있어도, 아키텍트의 작업 원리는 영원하다"**는 점입니다.

<div class="arch-grid">
  <div class="arch-card">
    <div class="card-title">① 목표 정의 (Goal Specification)</div>
    <div class="card-desc">코드 문법이 아니라 "산출물의 형태(차트, 표, JSON)"를 명확히 선언하는 능력은 모든 도구에서 100% 동일합니다.</div>
  </div>
  <div class="arch-card">
    <div class="card-title">② 환경 격리 (Workspace Isolation)</div>
    <div class="card-desc">독립된 폴더와 가상환경 위에서 안전하게 코드가 실행되도록 통제하는 원칙은 도구와 무관하게 동일합니다.</div>
  </div>
  <div class="arch-card">
    <div class="card-title">③ 인간 감독관 검증 (Human-in-the-Loop)</div>
    <div class="card-desc">AI의 결과물을 맹신하지 않고 사람 눈으로 차트와 수치를 교차 검증하는 루프는 Cursor, Codex 어디서나 똑같이 적용됩니다.</div>
  </div>
</div>

> 📌 **핵심 원칙**: 본 강의의 모든 프롬프트와 검증 프로토콜은 표준 터미널 및 파일 기반으로 설계되어 있으므로, 향후 어떤 새로운 AI 도구가 등장하더라도 동일한 논리로 100% 적용할 수 있습니다.

---

### 3. 계정 및 비용 구조의 투명한 이해

Claude Code를 사용하기 위해 계정을 설정할 때 알아두어야 할 비용 체계는 매우 단순하고 투명합니다.

| 방식 | 과금 구조 | 추천 대상 | 특징 및 장점 |
|---|---|---|---|
| **Anthropic Console (API Key)** | 쓴 만큼 후불 결제 ($5~$10 충전) | 가끔 집중적으로 작업하는 사용자 | **프롬프트 캐싱(Prompt Caching)**이 자동 적용되어 반복 호출 시 비용이 최대 90% 절감됨 |
| **Claude Pro / Team / Max (정액제)** | 월 $20~$30 구독형 | 일상적으로 대화 및 코딩을 함께 쓰는 사용자 | 한도 내에서 추가 비용 걱정 없이 자유롭게 사용 가능 |

- **이 강의 전체 실습 비용**: 5개 파트의 모든 실습을 완료하는 데 소요되는 API 비용은 **커피 한두 잔 가격($3~$7 내외)**에 불과합니다.
- 토큰 낭비를 줄이는 실전 팁은 **Part 5 (Clip 41)**에서 체계적으로 배웁니다.

---

### 4. 3단계 환경 준비 로드맵

우리가 직접 수행할 설치 작업은 딱 1단계뿐이며, 2단계와 3단계는 AI가 스스로 수행합니다.

<div class="step-flow">
  <div class="step-card">
    <span class="step-num">Step 1</span>
    <div class="step-body">
      <div class="step-title">사람이 직접 수행: AI 도구 실행 및 작업 폴더 연결</div>
      <div class="step-desc">데스크톱 앱(Cursor / Claude Desktop)을 실행하거나 터미널에서 Claude Code를 켭니다.</div>
    </div>
  </div>
  <div class="step-card ai">
    <span class="step-num">Step 2</span>
    <div class="step-body">
      <div class="step-title">AI에게 전권 위임: Python 가상환경 및 라이브러리 자동 구축</div>
      <div class="step-desc">표준 프롬프트 1줄을 입력하면 AI가 <code>.venv</code> 생성 및 <code>pandas</code>, <code>FinanceDataReader</code> 설치를 스스로 완료합니다.</div>
    </div>
  </div>
  <div class="step-card verify">
    <span class="step-num">Step 3</span>
    <div class="step-body">
      <div class="step-title">사람이 눈으로 확인: 삼성전자 주가 데이터 출력 대조</div>
      <div class="step-desc">화면에 삼성전자의 최근 5일치 일봉 표가 정상적으로 떠 있는지 눈으로 최종 확인합니다.</div>
    </div>
  </div>
</div>

---

## 실습 — 실습 01: AI 도구 환경 구축

> 💡 **진입 방식 선택 가이드**:  
> • **초보자 추천 (데스크톱 앱 방식)**: Cursor나 Claude Desktop 등의 앱을 켜고 `ai-trading` 폴더를 연 뒤 바로 대화창에 프롬프트를 입력하세요.  
> • **터미널 중심 (Claude Code CLI 방식)**: 아래 CLI 명령어로 터미널에서 직접 실행하세요. 어떤 방식을 쓰든 이후의 파이썬 코드 생성과 트레이딩 전략 검증 프롬프트는 **100% 동일**합니다.

### Step 1. 작업 폴더 생성 및 실행

컴퓨터의 원하는 위치(예: 바탕화면 또는 작업 드라이브)에 `ai-trading` 폴더를 만들고 AI 도구를 시작합니다.

#### [방식 A] 데스크톱 앱에서 바로 실행 (가장 간편)
1. 데스크톱 앱(Cursor 또는 Claude Desktop)을 실행합니다.
2. `File` → `Open Folder`를 눌러 새로 만든 `ai-trading` 폴더를 엽니다.
3. AI 대화창을 열고 Step 2의 프롬프트를 입력합니다.

#### [방식 B] 터미널 CLI(Claude Code)에서 실행
1. 터미널(PowerShell, Terminal, WSL 등)을 열고 폴더로 이동합니다.
```bash
# 1. 작업 폴더 생성 및 이동
mkdir ai-trading
cd ai-trading

# 2. Claude Code 설치 (최초 1회)
npm install -g @anthropic-ai/claude-code

# 3. Claude Code 실행 및 로그인
claude
```
> 터미널에 `claude`를 입력하면 웹 브라우저가 열리며 Anthropic 계정 로그인이 진행됩니다. 로그인이 완료되면 터미널에 대화창 프롬프트(`>`)가 나타납니다.

---

### Step 2. 환경 구축 프롬프트 입력

대화창이 열리면 아래의 표준 프롬프트를 복사하여 그대로 입력하세요.

```prompt
지금부터 한국 주식 데이터를 분석하고 퀀트 트레이딩 시스템을 구축하는 프로젝트를 시작할 거야.

다음 환경을 준비해줘:
1. 이 폴더 안에 Python 가상환경(.venv)을 생성하고 활성화할 수 있도록 구성해줘.
2. 주식 데이터를 수집하고 분석할 수 있도록 pandas, matplotlib, FinanceDataReader 라이브러리를 설치해줘.
3. 설치가 완료되면 삼성전자(종목코드: 005930)의 최근 5일치 일봉 데이터(시가/고가/저가/종가/거래량)를 터미널에 깔끔한 표로 출력해서 정상 동작을 증명해줘.
```

---

### 내 눈으로 확인할 체크리스트

Claude Code가 스스로 명령어를 실행하고 설치를 진행하는 과정을 지켜본 뒤, 마지막에 터미널에 출력된 화면을 확인합니다.

- [ ] **가상환경 및 패키지 설치 완료**: 에러 메시지 없이 pandas, FinanceDataReader 설치가 끝났다.
- [ ] **표 형태의 데이터 출력**: 터미널에 `Date, Open, High, Low, Close, Volume` 컬럼을 가진 5행의 데이터 표가 보인다.
- [ ] **종가 숫자 정상성**: 출력된 삼성전자의 최근 종가(예: 70,000~80,000원대)가 실제 시세와 유사한 수준이다.

---

## 자주 발생하는 문제 및 트러블슈팅

### 1. `npm` 명령어를 찾을 수 없다는 에러 (`command not found: npm`)
- **원인**: 컴퓨터에 Node.js가 설치되어 있지 않습니다.
- **해결**: [nodejs.org](https://nodejs.org)에 접속하여 LTS(안정화) 버전을 다운로드받아 설치한 후 터미널을 재시작하세요.

### 2. 가상환경 권한 오류 (Windows PowerShell `ExecutionPolicy` 오류)
- **원인**: Windows의 스크립트 실행 제한 정책 때문입니다.
- **해결**: Claude Code 대화창에 에러 메시지를 그대로 붙여넣으면, AI가 권한 우회 명령어(`Set-ExecutionPolicy -Scope Process RemoteSigned`)를 적용하거나 직접 적절한 방식으로 실행해 줍니다.

### 3. 방화벽이나 네트워크 연결 문제
- **해결**: 다음 프롬프트를 입력하여 대체 패키지 인덱스 미러를 사용하도록 지시하세요.

```prompt
패키지 다운로드 중 타임아웃 에러가 발생했어.
pip install 타임아웃 시간을 늘리고 다시 FinanceDataReader를 설치한 뒤 삼성전자 종가를 출력해줘.
```

---

## 다음 클립 예고

축하합니다! 이제 여러분의 컴퓨터에는 퀀트 트레이딩 개발을 위한 최고의 AI 조수가 준비되었습니다.  
다음 클립에서는 AI에게 **내가 원하는 결과를 모호함 없이 단번에 이끌어내는 '결과 중심 프롬프트 작성법'**을 마스터하겠습니다.
