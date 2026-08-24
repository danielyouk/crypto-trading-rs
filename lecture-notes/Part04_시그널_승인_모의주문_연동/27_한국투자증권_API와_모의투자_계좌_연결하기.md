# 한국투자증권 API와 모의투자 계좌 연결하기

## 이번 클립에서 만들 것

이 클립이 끝나면, 여러분은 **한국투자증권(KIS) Developers 포털에서 모의투자 API 키를 발급**받고, 보안 환경설정 파일(`.env`)을 통해 인증 정보를 안전하게 격리한 뒤, **파이썬 스크립트로 OAuth2 접근 토큰을 발급받아 모의투자 계좌의 예수금 및 보유 주식 잔고를 화면에 깔끔한 표로 출력**하는 데 성공하게 됩니다.

> 💡 **수강생 안심 안내**: 실계좌(진짜 돈)가 아니라 **가상 머니 5억 원이 지급되는 '모의투자 계좌'**로 진행하므로 금전적 위험이 전혀 없습니다. 또한 API Key와 Secret은 절대로 외부에 공개되지 않도록 `.env` 환경변수로 철저히 보호합니다.

- KIS Developers 포털에서 모의투자 계좌 등록 및 `AppKey`, `AppSecret` 발급 3단계 가이드
- 인증 정보 보안의 절대 규칙: `.env` 파일 관리와 `.gitignore` 격리
- OAuth2 `access_token` 발급 및 만료 시간 자동 관리 원리
- **실습 15**: `kis_auth.py` 작성 및 모의투자 계좌 잔고(예수금, 평가손익) 조회 검증

---

## 이론 핵심

### 1. 증권사 Open API 인증 아키텍처

한국투자증권 Open API는 현대적인 REST API 및 OAuth2 토큰 방식을 사용합니다.

```text
[내 컴퓨터 / Python] 
   ──> AppKey + AppSecret을 KIS 인증 서버로 POST 전송
   <── 24시간 유효한 'access_token' 수신
   ──> 매 요청마다 헤더에 'Bearer access_token'을 실어서 시세/잔고/주문 API 호출
```

---

### 2. 보안 제1원칙: 환경변수(`.env`) 격리

API 키를 파이썬 코드 안에 하드코딩하면 GitHub 등에 소스코드를 올렸을 때 키가 유출되는 심각한 보안 사고가 발생합니다.

```text
ai-trading/
├── .env                  # [보안 절대 파일] APP_KEY, APP_SECRET, CANO 등 보관
├── .gitignore            # .env가 Git에 커밋되지 않도록 차단
├── kis_auth.py           # dotenv 라이브러리로 .env의 값을 읽어서 사용
```

```env
# .env 파일 예시
KIS_APP_KEY=PSxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIS_APP_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
KIS_CANO=50123456
KIS_ACNT_PRDT_CD=01
KIS_IS_PAPER=True
```

---

## 실습 — 실습 15: KIS API 인증 및 계좌 잔고 조회

### Step 1. 모의투자 API 키 발급 (웹 브라우저)
1. [한국투자증권 KIS Developers](https://apiportal.koreainvestment.com/)에 접속하여 로그인합니다.
2. [KOSCOM Open API ➔ 모의투자] 메뉴에서 앱을 등록하고 **App Key**와 **App Secret**을 발급받습니다.
3. 내 모의투자 계좌번호 8자리(CANO)와 상품코드 2자리(01)를 확인합니다.

---

### Step 2. Claude Code에게 연동 모듈 작성 요청

Claude Code 대화창에 아래 프롬프트를 입력하세요.

```prompt
한국투자증권(KIS) 모의투자 Open API와 연동하여 계좌 잔고를 조회하는 모듈을 만들고 싶어. (Context)

다음 요구사항에 맞춰 파이썬 스크립트를 작성하고 실행해줘: (Deliverable)
1. python-dotenv와 requests 패키지를 확인 및 설치해줘.
2. .env 파일 템플릿을 생성하고, 사용자가 입력한 KIS_APP_KEY, KIS_APP_SECRET, KIS_CANO, KIS_ACNT_PRDT_CD 환경변수를 안전하게 로드하도록 구현해줘.
3. KIS 모의투자 서버(https://openapivts.koreainvestment.com:29443)의 OAuth2 토큰 발급 엔드포인트(/oauth2/tokenP)를 호출해 access_token을 발급받아줘.
4. 발급된 토큰으로 '주식 잔고조회(TTTC8434R)' API를 호출하여 내 계좌의 총 예수금, 총 평가금액, 총 손익률을 대화창에 요약 표로 출력해줘.
5. 전체 코드는 'kis_auth.py'로 저장해줘.

실행 완료 후 화면에 내 모의투자 계좌의 예수금 요약표를 출력해서 정상 연결을 증명해줘. (Verification)
```

---

### 내 눈으로 확인할 체크리스트

- [ ] `.env` 파일에 발급받은 모의투자 키와 계좌번호가 정상 저장되었다.
- [ ] `kis_auth.py` 실행 시 OAuth2 접근 토큰(Bearer 토큰)이 에러 없이 발급되었다.
- [ ] 화면에 내 모의투자 계좌의 총 예수금(예: 500,000,000원)과 평가 잔고 표가 성공적으로 출력되었다.

---

## 자주 발생하는 문제

- **"유효하지 않은 AppKey/Secret" 에러 (EGW00133)**: 실전용 키와 모의투자용 키가 다릅니다. KIS Developers 포털에서 [모의투자] 탭에서 발급받은 키인지 확인하세요.
- **도메인 접속 포트 오류**: 모의투자는 `https://openapivts.koreainvestment.com:29443` (포트 29443)을 사용해야 합니다.

---

## 다음 클립 예고

증권사 계좌와의 통신 파이프라인이 뚫렸습니다!  
다음 클립에서는 KIS API를 통해 **실제 주식 시장의 실시간 가격(현재가, 호가) 데이터를 파이썬으로 수신하고 지연 없이 가공하는 방법**을 배워보겠습니다.
