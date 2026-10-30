"""
클립 22 개선안(A/B/C)을 카드가 보지 않은 기간에서 확인하는 재현 스크립트 (강사용, 2026-10-30)

- 규칙·값·비용·체결은 risk_overlay_backtest.py의 시뮬레이터를 그대로 쓰고, 가격만 바꿉니다.
    A: 200일선 원본 / B: A + 완충 밴드 1% / C: B + 고점 대비 -10% 손절(월말 재진입) + 변동성 목표 15%
    (참고) A + 변동성 목표 15%
- KODEX 200 일봉은 FinanceDataReader로 2014년부터만 받아지므로, 그 전은 KOSPI 200 지수('KS200')로 대신합니다.
- 출력: 세 기간의 CAGR / MDD / 칼마(CAGR ÷ |MDD|)
    ① 2016-01 ~ 2026-08, KODEX 200 (카드를 만든 기간)
    ② 2016-01 ~ 2026-08, KOSPI 200 지수 (대용 지수로도 ①의 결론이 나오는지)
    ③ 2003-01 ~ 2015-12, KOSPI 200 지수 (카드가 보지 않은 기간)

실행:
    .venv/bin/python risk_overlay_oos_check.py
"""

import sys

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

import types  # noqa: E402

# risk_overlay_backtest.py는 듀얼 모멘텀용으로 dual_momentum_backtest를 import합니다.
# 이 스크립트는 추세추종만 쓰므로, 그 파일이 없어도 돌도록 빈 모듈을 넣어 둡니다.
sys.modules.setdefault("dual_momentum_backtest", types.ModuleType("dual_momentum_backtest"))
_argv = sys.argv
sys.argv = [sys.argv[0]]          # risk_overlay_backtest.py는 import할 때 sys.argv를 손절폭으로 읽습니다
import risk_overlay_backtest as ro  # noqa: E402
sys.argv = _argv

TF = ro.TF


def split(px: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (px[["Open"]].rename(columns={"Open": TF}).astype(float),
            px[["Close"]].rename(columns={"Close": TF}).astype(float))


def holdings(closes: pd.DataFrame) -> dict[str, pd.Series]:
    """risk_overlay_backtest.base_holdings의 추세추종 두 줄과 같은 규칙 (듀얼 모멘텀 제외)."""
    sma = closes[TF].rolling(200).mean()
    a = (closes[TF] > sma).astype(int).shift(1).fillna(0).astype(int).map({1: TF, 0: None})
    c, up, dn = closes[TF].values, (sma * (1 + ro.BAND)).values, sma.values
    state, states = 0, np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if np.isnan(dn[i]):
            states[i] = 0
            continue
        if state == 0 and c[i] > up[i]:
            state = 1
        elif state == 1 and c[i] < dn[i]:
            state = 0
        states[i] = state
    b = pd.Series(states, index=closes.index).shift(1).fillna(0).astype(int).map({1: TF, 0: None})
    return {"A": a, "B": b}


def month_end_reentry(closes: pd.DataFrame) -> pd.Series:
    me = closes.groupby(closes.index.to_period("M")).tail(1).index
    ok = pd.Series(False, index=closes.index)
    for d in me:
        nxt = closes.index[closes.index > d]
        if len(nxt):
            ok[nxt[0]] = True
    return ok


def stats(daily: pd.Series) -> tuple[float, float, float]:
    eq = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    return cagr, mdd, cagr / abs(mdd)


def table(px: pd.DataFrame, start: str, end: str, label: str) -> None:
    opens, closes = split(px)
    h = holdings(closes)
    ok = month_end_reentry(closes)
    rows = [("A 200일선 원본", h["A"], False, False), ("B 밴드 1%", h["B"], False, False),
            ("C 밴드 + 손절 -10% + 변동성 15%", h["B"], True, True), ("(참고) A + 변동성 15%", h["A"], False, True)]
    print(f"\n### {label} ({start} ~ {end})\n\n| 전략 | CAGR | MDD | 칼마 |\n|---|---:|---:|---:|")
    for name, want, stop, vol in rows:
        daily, _ = ro.simulate(opens, closes, want.reindex(closes.index), ok, stop, vol)
        c, m, k = stats(daily.loc[start:end])
        print(f"| {name} | {c:+.1%} | {m:.1%} | {k:.2f} |")
    c, m, k = stats(px.Close.pct_change().loc[start:end].fillna(0.0))
    print(f"| 단순 보유 | {c:+.1%} | {m:.1%} | {k:.2f} |")


if __name__ == "__main__":
    etf = fdr.DataReader("069500", "2014-01-01", "2026-08-31")[["Open", "Close"]].astype(float)
    ks = fdr.DataReader("KS200", "2002-01-01", "2026-08-31")[["Open", "Close"]].astype(float)  # 교재 7번과 같은 시작일
    ks = ks[ks.Open > 0]
    table(etf, "2016-01-01", "2026-08-31", "① KODEX 200, 카드를 만든 기간")
    table(ks, "2016-01-01", "2026-08-31", "② KOSPI 200 지수, 같은 기간 (대용 확인)")
    table(ks, "2003-01-01", "2015-12-31", "③ KOSPI 200 지수, 카드가 보지 않은 기간")
