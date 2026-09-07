"""
위험관리 규칙(오버레이)을 Part 2 전략 위에 얹었을 때 MDD가 어떻게 바뀌는지 확인하는 백테스트
- 기간: 2018-01-01 ~ 2026-08-31 (2026년 7월 코스피 급락 포함)
- 기본 전략 (Part 2 그대로, 위험관리 없음):
    추세추종: KODEX 200(069500) 종가 > 200일선이면 보유, 아니면 현금. 신호 오늘 종가, 체결 내일 시가
    추세추종 + 완충 밴드 1%: 종가 > 200일선 x 1.01 이면 진입, 종가 < 200일선 이면 청산 (클립 22 전략 B)
    듀얼 모멘텀: 4 ETF(나스닥100 133690 / 코스피200 069500 / 미국채10년 305080 / 단기채 214980), 월말 판정, 익월 첫 거래일 시가 교체
- 위험관리 규칙 (매매 규칙 3요소 중 ③, 두 전략에 동일하게 적용):
    R1 고점 대비 손절: 보유 중 종가가 보유 기간 고점 대비 -N% 아래로 내려오면 익일 시가에 전량 매도.
       손절 후에는 다음 월말 점검일까지 현금. 월말에 기본 전략 신호가 다시 '보유'면 익월 첫 거래일 시가 재진입
    R2 변동성 목표 비중: 최근 20일 실현 변동성(연율)이 목표(예: 15%)보다 크면 비중 = 목표/실현변동성 (최대 100%).
       오늘 종가로 계산, 내일 시가에 조정. 비중 변화가 10%p 이상일 때만 실제 매매(잔거래 방지)
    조합: R1만 / R2만 / R1+R2
- 비용: 편도 0.125% (왕복 0.25%, 강의 기준) x 거래 비중
- 산출: 전략별 x 규칙별 CAGR / MDD / 샤프 / 2026년 7월 낙폭 / 거래 수 표 (Markdown)
"""

import sys

import numpy as np
import pandas as pd

import dual_momentum_backtest as dm

START, END = "2018-01-01", "2026-08-31"
COST = 0.00125            # 편도 0.125% = 왕복 0.25%
STOP = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10      # 고점 대비 손절폭
TARGET_VOL = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15  # 변동성 목표(연율)
VOL_WIN, REBAL_STEP = 20, 0.10
BAND = 0.01                # 완충 밴드 (클립 22 전략 B)
TF = "069500"


def base_holdings(closes: pd.DataFrame) -> dict[str, pd.Series]:
    """일별 '기본 전략이 원하는 보유 자산' (None=현금). 값은 그날 시가부터 적용되는 상태."""
    out = {}
    # 추세추종
    sma = closes[TF].rolling(200).mean()
    sig = (closes[TF] > sma).astype(int).shift(1).fillna(0).astype(int)
    out["추세추종 (KODEX 200 200일선)"] = sig.map({1: TF, 0: None})
    # 추세추종 + 완충 밴드 1%: 종가 > 200일선 x 1.01 이면 진입, 종가 < 200일선 이면 청산 (히스테리시스)
    c, up, dn = closes[TF].values, (sma * (1 + BAND)).values, sma.values
    state, states = 0, np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if np.isnan(dn[i]):
            states[i] = 0; continue
        if state == 0 and c[i] > up[i]:
            state = 1
        elif state == 1 and c[i] < dn[i]:
            state = 0
        states[i] = state
    sigb = pd.Series(states, index=closes.index).shift(1).fillna(0).astype(int)
    out[f"추세추종 + 완충 밴드 {BAND:.0%}"] = sigb.map({1: TF, 0: None})
    # 듀얼 모멘텀
    sigm = dm.monthly_signals(closes)
    h = dm.build_holdings(sigm, closes.index)
    out["듀얼 모멘텀 (4 ETF)"] = h
    return out


def simulate(opens, closes, want: pd.Series, reentry_ok: pd.Series, use_stop: bool, use_vol: bool):
    """
    want: 일별 기본 전략의 목표 자산(None=현금). 오늘 시가부터 적용되는 상태.
    반환: 일수익률, 거래 수(편도)
    """
    idx = closes.index
    vol = closes.pct_change().rolling(VOL_WIN).std() * np.sqrt(252)

    def vol_weight(asset: str, day) -> float:
        v = vol.at[day, asset]
        return 1.0 if (np.isnan(v) or v <= TARGET_VOL) else TARGET_VOL / v

    daily = np.zeros(len(idx)); n_trades = 0
    asset, w = None, 0.0            # 어제 종가 시점의 보유 자산과 비중
    peak = None                     # 보유 기간 고점(종가)
    stopped = False                 # 손절 후 다음 월말 점검일까지 대기
    pending_w = None                # 오늘 시가에 조정할 비중 (어제 종가에 결정)
    for i, d in enumerate(idx):
        prev = idx[i - 1] if i > 0 else None
        target = want[d] if isinstance(want[d], str) else None

        # 1) 오늘 시가에 무엇을 할지 결정 (어제 종가까지의 정보만 사용)
        if stopped:
            if target is not None and reentry_ok[d]:
                stopped = False
                new_asset = target
                new_w = vol_weight(target, prev) if (use_vol and prev is not None) else 1.0
            else:
                new_asset, new_w = None, 0.0
        elif target is None:
            new_asset, new_w = None, 0.0
        elif target != asset:
            new_asset = target
            new_w = vol_weight(target, prev) if (use_vol and prev is not None) else 1.0
        else:
            new_asset, new_w = asset, (pending_w if pending_w is not None else w)

        # 2) 체결과 오늘 수익률: 어제 종가->오늘 시가는 기존 보유분, 시가->종가는 새 보유분
        r_gap = opens.at[d, asset] / closes.at[prev, asset] - 1 if (asset is not None and prev is not None) else 0.0
        if new_asset != asset:
            turnover = w + new_w
            n_trades += int(w > 0) + int(new_w > 0)
        else:
            turnover = abs(new_w - w)
            n_trades += int(turnover > 1e-9)
        r_day = closes.at[d, new_asset] / opens.at[d, new_asset] - 1 if new_asset is not None else 0.0
        daily[i] = (1 + w * r_gap) * (1 - COST * turnover) * (1 + new_w * r_day) - 1

        # 3) 오늘 종가로 내일 시가의 행동 결정
        fresh_entry = new_asset is not None and new_asset != asset
        asset, w = new_asset, new_w
        pending_w = None
        if asset is None:
            peak = None
            continue
        c = closes.at[d, asset]
        peak = c if (fresh_entry or peak is None) else max(peak, c)
        if use_stop and c / peak - 1 <= -STOP:
            stopped, peak = True, None          # 내일 시가 전량 매도
        elif use_vol:
            tw = vol_weight(asset, d)
            if abs(tw - w) >= REBAL_STEP or (tw >= 1.0 and w < 1.0 - 1e-9):
                pending_w = tw
    return pd.Series(daily, index=idx), n_trades


def metrics(daily: pd.Series) -> dict:
    eq = (1 + daily).cumprod(); dd = eq / eq.cummax() - 1
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    sd = daily.std()
    jul = eq.loc["2026-06-01":"2026-08-31"]; jul_dd = (jul / jul.cummax() - 1).min()
    return {"CAGR": eq.iloc[-1] ** (1 / years) - 1, "MDD": dd.min(), "MDDDate": dd.idxmin(),
            "Sharpe": daily.mean() / sd * np.sqrt(252) if sd > 0 else np.nan, "2026-07": jul_dd,
            "Total": eq.iloc[-1] - 1}


def main():
    opens, closes = dm.load_prices()
    opens, closes = opens.loc[:END], closes.loc[:END]
    # 손절 후 재진입 허용일: 월말 점검일 다음 거래일
    me = closes.groupby(closes.index.to_period("M")).tail(1).index
    reentry_ok = pd.Series(False, index=closes.index)
    for d in me:
        nxt = closes.index[closes.index > d]
        if len(nxt):
            reentry_ok[nxt[0]] = True
    wants = base_holdings(closes)
    rows = []
    for name, want in wants.items():
        want = want.reindex(closes.index)
        for rule, (s, v) in {"위험관리 없음 (Part 2 그대로)": (False, False),
                             f"R1 고점 대비 -{STOP:.0%} 손절": (True, False),
                             f"R2 변동성 목표 {TARGET_VOL:.0%}": (False, True),
                             "R1 + R2": (True, True)}.items():
            daily, n = simulate(opens, closes, want, reentry_ok, s, v)
            daily = daily.loc[START:END]
            m = metrics(daily)
            rows.append({"전략": name, "규칙": rule, "CAGR": m["CAGR"], "MDD": m["MDD"], "MDD일": m["MDDDate"].date(),
                         "샤프": m["Sharpe"], "2026년 6~8월 낙폭": m["2026-07"], "누적": m["Total"], "거래(편도)": n})
    df = pd.DataFrame(rows)
    print(f"### 위험관리 규칙 적용 전후 ({START} ~ {END}, 편도 비용 {COST:.3%}(왕복 {2*COST:.2%}), 손절 -{STOP:.0%}, 변동성 목표 {TARGET_VOL:.0%})\n")
    print("| 전략 | 규칙 | CAGR | MDD (날짜) | 샤프 | 2026년 6~8월 낙폭 | 누적 | 거래(편도) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        print(f"| {r['전략']} | {r['규칙']} | {r['CAGR']:+.1%} | {r['MDD']:.1%} ({r['MDD일']}) | {r['샤프']:.2f} | "
              f"{r['2026년 6~8월 낙폭']:.1%} | {r['누적']:+.0%} | {r['거래(편도)']} |")


if __name__ == "__main__":
    main()
