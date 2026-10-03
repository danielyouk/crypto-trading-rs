# 같은 작업 방식을 Codex에도 적용하기

## 이번 클립에서 만들 것

이 클립이 끝나면, 여러분은 지금까지 익힌 **'생각 ➔ 네 칸 요청([맥락]·[만들 것]·[하지 말 것]·[확인]) ➔ 화면 확인 ➔ 피드백 개선'**이라는 제로코드 퀀트 아키텍트의 작업 루프가 **Codex(OpenAI가 만든 코딩용 AI 도구로, Claude Code와 비슷한 역할), Cursor(AI 기능이 들어간 코드 편집기) 등 지금과 앞으로의 어떤 AI 도구에서도 똑같이 통한다**는 것을 확인하고, 도구의 변화에 흔들리지 않는 영구적인 시스템 구축 역량을 완성하게 됩니다.

> 💡 **아키텍트의 궁극적 무기**: AI 모델과 도구는 6개월마다 새로운 이름으로 쏟아져 나옵니다. 하지만 **"정해진 형식의 데이터 파일(JSON), 엄격한 안전장치, 사람이 매수를 승인하는 절차"**라는 퀀트 시스템의 본질적 설계도는 도구가 무엇으로 바뀌든 영원히 동일합니다.

- **어떤 도구에서도 통하는 작업 방식**: 도구는 껍데기일 뿐이며, 아키텍트의 논리적 사고와 목표 선언이 본질이다
- Claude Code에서 만든 코드와 프롬프트를 **Codex나 다른 AI 도구로 옮겨 쓰는 3단계 규칙**
- 프로그램끼리 정해진 형식의 파일(`signals.json`, `orders_날짜.json`)로만 주고받기 때문에, 어떤 AI 도구로 만들었든 시스템이 그대로 돌아가는 이유
- 미래에 어떤 초지능 AI가 등장하더라도 변하지 않는 **'사람이 마지막에 확인하고 승인하는 역할'의 가치**

---

## 이론 핵심

### 1. 도구는 바뀌어도 시스템 설계는 그대로다

<div class="compare-box">
  <div class="compare-card info">
    <div class="tag">도구에 기대는 접근법</div>
    <div class="steps">
      • 특정 에디터의 단축키나 특정 AI의 고유 명령어에 집착.<br>
      • 새로운 도구가 나오면 처음부터 다시 공부해야 함.<br>
      • <strong>결과</strong>: 도구 유행에 휘둘리며 시스템 본질을 놓침.
    </div>
  </div>
  <div class="compare-card good">
    <div class="tag">시스템 설계자의 접근법 (도구와 무관)</div>
    <div class="steps">
      • <strong>누구나 쓸 수 있는 재료</strong>: 파이썬과 누구나 무료로 설치할 수 있는 공개 라이브러리(pandas, FinanceDataReader 등), 프로그램끼리는 JSON 파일로 주고받기.<br>
      • <strong>네 칸 프롬프트 골격</strong>: 어떤 AI 도구에 넣어도 요청이 같은 뜻으로 전달됨.<br>
      • <strong>결과</strong>: Claude Code, Codex, Cursor 어디서든 1분 만에 시스템 재현 가능!
    </div>
  </div>
</div>

---

### 2. 다른 AI 도구로 작업 방식을 확장하는 3단계 원칙

<div class="step-flow">
  <div class="step-card">
    <span class="step-num">Step 1</span>
    <div class="step-body">
      <div class="step-title">[맥락] 그대로 전달</div>
      <div class="step-desc">"우리는 신호를 만들고, 매수는 사람이 승인하고, 주문과 손절은 프로그램이 실행하는 주식 트레이딩 시스템을 만들고 있어. signals.json과 날짜별 주문 목록(orders_날짜.json)을 기반으로 동작해."</div>
    </div>
  </div>
  <div class="step-card ai">
    <span class="step-num">Step 2</span>
    <div class="step-body">
      <div class="step-title">[만들 것]·[하지 말 것] 그대로 선언</div>
      <div class="step-desc">"generate_signals.py의 목표 변동성을 허용 범위(12~20%) 안에서 15%에서 12%로 낮추고, 바뀐 값이 목표 수량에 어떻게 반영되는지 표로 보여줘. 허용 범위 밖의 값은 쓰지 말고, 다른 파일은 고치지 마."</div>
    </div>
  </div>
  <div class="step-card verify">
    <span class="step-num">Step 3</span>
    <div class="step-body">
      <div class="step-title">[확인] 그대로 검증</div>
      <div class="step-desc">"완성되면 스크립트를 실행해서 화면에 검증 로그를 출력하고 파일 저장 경로를 알려줘."</div>
    </div>
  </div>
</div>

---

## 실습 — 실습 30: 다른 도구에서도 그대로 돌아가는지 확인

Claude Code 대화창에 아래 프롬프트를 입력하여, 우리 시스템이 특정 AI 도구 없이도 파이썬과 JSON 파일만으로 돌아가는지 최종 확인해 보세요.

> 📁 **공유 파일**: 이 프롬프트는 `signals.json`과 가장 최근 날짜의 `orders_날짜.json`을 읽기만 하고 고치거나 새로 만들지 않습니다. 승인 상태를 고쳐 둔 주문 목록도 그대로 남습니다.

```prompt
[맥락] 우리가 만든 AI 트레이딩 시스템의 프로그램 파일들(kis_auth, market_data, generate_signals, prepare_orders, approve_orders, notifier, execute_orders, risk_guard, order_validator, kill_switch, daily_journal, 그리고 만들었다면 run_after_close, run_market_open)이 Claude Code 없이도, 다른 AI 코딩 도구(Codex, Cursor 등)나 일반 터미널에서 그대로 돌아가는지 점검하고 싶어.

[만들 것] 다음 작업을 수행해줘:
1. 이 시스템이 따로 설치해서 쓰는 라이브러리 목록을 정리해줘. 목록 파일(requirements.txt)이 있으면 그것을 보고, 없으면 없다고 보고한 뒤 코드에서 실제로 불러 쓰는 라이브러리를 찾아 정리해. 그리고 전부 누구나 무료로 설치할 수 있는 공개 라이브러리인지, 특정 AI 도구에서만 쓸 수 있는 것은 없는지 점검해줘.
2. signals.json과 가장 최근 날짜의 orders_날짜.json 두 파일에, 그 파일을 읽는 코드가 필요로 하는 항목이 빠짐없이 있고 값 형식(날짜·숫자·비중 합계)이 맞는지 검사해줘. 파일이 없으면 새로 만들지 말고 '파일 없음'으로 적어.
3. 다른 AI 도구(Codex 등)를 처음 사용하는 사람이 이 프로젝트를 인계받았을 때 1분 만에 실행할 수 있도록 작성된 'docs/universal_tool_guide.md' 매뉴얼을 생성해줘. 명령마다 주문이 나가는지 표시하고, 처음 확인할 때 쓸 주문 없는 실행('--dry-run' 등)도 함께 적어줘.

[하지 말 것]
- 점검한다고 우리 프로그램 파일(.py)을 실행하지 마. 주문이 나가거나, 내가 승인 상태를 고쳐 둔 주문 목록이 바뀔 수 있어. 라이브러리와 JSON 파일은 읽어서 점검해.
- 점검 중 발견한 문제를 소스 코드나 JSON 파일을 고쳐서 해결하지 마. 발견 사항만 기록해.
- 특정 AI 도구 전용 설정 파일이나 플러그인을 새로 만들지 마. 가이드는 파이썬과 터미널 명령만으로 써.
- 검증 결과를 '통과'로 뭉뚱그리지 말고, 항목별로 통과/미통과와 근거를 적어.

[확인] 대화창에 설치 라이브러리·두 JSON 파일의 항목과 형식·실행 가이드 3개 항목의 점검 결과가 출력되었는지, 그리고 'docs/universal_tool_guide.md' 파일이 저장되었는지 확인해줘.
```

---

### 내 눈으로 확인할 체크리스트

- [ ] 시스템이 쓰는 라이브러리 목록과 두 JSON 파일(`signals.json`, `orders_날짜.json`)의 점검 결과가 항목별 통과/미통과로 나왔다.
- [ ] `docs/universal_tool_guide.md`만 보고 다른 도구나 터미널에서 주문 없이(`--dry-run`) 실행해 볼 수 있는지 확인했다.

---

## 다음 클립 예고

이제 43개 클립의 대장정이 마지막 1개 클립만을 남겨두고 있습니다!  
다음 마지막 클립에서는 **직장인/1인 퀀트로서 평생 유지 가능한 '일간 10분, 주간 30분, 월간 1시간 운영 루틴'**을 최종 정립하고 대단원의 막을 내리겠습니다.
