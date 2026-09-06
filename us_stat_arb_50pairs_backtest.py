"""
S&P 500 섹터별 50개 페어 1:1 Long/Short 통계적 차익거래(Stat-Arb) 포트폴리오 백테스트
- 기간: 1995-01-01 ~ 2026-08-31 (60일 Z-Score 워밍업을 위해 1994-09-01부터 데이터 수신)
- 데이터: FinanceDataReader(야후 파이낸스). 종목별 CSV 캐시(data/us_stat_arb_cache/) 사용.
          수익률·체결은 배당/분할 반영 수정주가(Adj Close, Open x Adj/Close), Z-Score는 분할 반영 Close 비율.
          상장이 늦은 종목(V, MA, META, GM, HLT 등)이 포함된 페어는 두 종목 모두 데이터가 있는 날부터 활성화.
- 페어별 규칙 (독립 판정, 레그당 편도 비용 0.05%):
    Z = (Ratio - 60일 평균) / 60일 표준편차,  Ratio = A 종가 / B 종가
    Z <= -1.5 : A 저평가 -> A 50% 롱 + B 50% 숏
    Z >= +1.5 : B 저평가 -> B 50% 롱 + A 50% 숏
    |Z| <  0.5 : 평균 복귀 -> 청산
    |Z| >= 3.0 : 구조적 결별 -> 하드스탑 청산. 이후 |Z| < 0.5 복귀 전까지 재진입 금지
    보유 중 반대편 진입 신호 -> 청산 후 반대 포지션 전환
- 체결: 당일 종가 Z로 판정, 익일 시가 체결 (미래참조 방지). 평가는 종가.
- 포트폴리오: 당일 포지션이 있는(활성) 페어에 자본을 균등 배분(매일 리밸런싱). 활성 페어가 없으면 현금(수익 0).
- 벤치마크: SPY 단순 보유 (첫날 시가 매수, 편도 0.05% 1회)
- 산출: Markdown 성과표 + Plotly 3패널 대화형 차트 us_stat_arb_50pairs_result.html (단독 실행 HTML)
"""

import os
from concurrent.futures import ThreadPoolExecutor

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PAIRS = {  # 섹터: [(A, B), ...]  10개 섹터 x 5개 = 50개
    "에너지":     [("XOM", "CVX"), ("COP", "EOG"), ("SLB", "HAL"), ("OXY", "DVN"), ("WMB", "OKE")],
    "필수소비재": [("KO", "PEP"), ("PG", "CL"), ("WMT", "COST"), ("MO", "PM"), ("GIS", "CPB")],
    "금융":       [("JPM", "BAC"), ("MS", "GS"), ("WFC", "C"), ("V", "MA"), ("AXP", "COF")],
    "헬스케어":   [("JNJ", "PFE"), ("MRK", "LLY"), ("ABT", "MDT"), ("UNH", "CI"), ("AMGN", "GILD")],
    "산업재":     [("CAT", "DE"), ("UPS", "FDX"), ("HON", "MMM"), ("LMT", "RTX"), ("UNP", "CSX")],
    "경기소비재": [("HD", "LOW"), ("MCD", "YUM"), ("TJX", "ROST"), ("F", "GM"), ("MAR", "HLT")],
    "정보기술":   [("INTC", "AMD"), ("TXN", "ADI"), ("MSFT", "ORCL"), ("ADBE", "INTU"), ("IBM", "ACN")],
    "커뮤니케이션": [("T", "VZ"), ("CMCSA", "CHTR"), ("GOOGL", "META"), ("DIS", "WBD"), ("NWSA", "FOXA")],
    "유틸리티":   [("NEE", "DUK"), ("SO", "D"), ("AEP", "XEL"), ("EXC", "PEG"), ("ED", "EIX")],
    "소재":       [("APD", "LIN"), ("NEM", "FCX"), ("SHW", "PPG"), ("NUE", "STLD"), ("VMC", "MLM")],
}
BENCH = "SPY"
START, END = "1995-01-01", "2026-08-31"
WARMUP_START = "1994-09-01"
WINDOW = 60
ENTRY_Z, EXIT_Z, STOP_Z = 2.0, 0.3, 3.5
COST = 0.0005            # 편도 0.05% (레그당). 진입/청산 각각 자본의 0.05% (레그 2개 x 50%)
LEG = 0.5
TRADING_DAYS = 252
RF_ANNUAL = 0.035        # 연 3.5% 무위험 현금 담보 이자율 (US T-Bill 장기 평균)
LEVERAGE = 2.5           # 헤지펀드 표준 Gross Exposure 레버리지 (2.5배)
CACHE_DIR = "data/us_stat_arb_cache"
OUT_HTML = "us_stat_arb_50pairs_result.html"
COLORS = {
    "strategy_lev": "#1f3f8f",
    "strategy_1x": "#2a78d6",
    "bench": "#7a7a7a",
    "active": "#2a78d6",
    "avail": "#52514e",
    "stop": "#e34948"
}
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


# ---------------------------------------------------------------- 데이터
def fetch(ticker: str) -> pd.DataFrame | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker}.csv")
    try:
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            df = fdr.DataReader(ticker, WARMUP_START, END)
            df.to_csv(path)
    except Exception as e:  # 피인수·상장폐지 등으로 404가 나는 종목
        print(f"[경고] {ticker} 데이터 수신 실패: {type(e).__name__}")
        return None
    df = df[["Open", "Close", "Adj Close"]].dropna()
    df = df[(df["Open"] > 0) & (df["Close"] > 0)]
    if len(df) < WINDOW * 2:
        print(f"[경고] {ticker} 데이터 부족 ({len(df)}행) -> 제외")
        return None
    factor = df["Adj Close"] / df["Close"]
    return pd.DataFrame({"open": df["Open"] * factor, "close": df["Adj Close"], "raw": df["Close"]}).sort_index()


def load_all() -> dict[str, pd.DataFrame]:
    tickers = sorted({t for ps in PAIRS.values() for p in ps for t in p} | {BENCH})
    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(fetch, tickers))
    return {t: df for t, df in zip(tickers, results) if df is not None}


# ---------------------------------------------------------------- 페어 시뮬레이션
def simulate_pair(a: pd.DataFrame, b: pd.DataFrame, name: str, sector: str):
    """
    한 페어의 일별 수익률(활성일만 0 아님), 활성 플래그, 거래 기록.
    포지션 +1 = A 롱/B 숏, -1 = A 숏/B 롱. 레그 수익률은 일별 50/50 리밸런싱 기준.
    """
    idx = a.index.intersection(b.index)
    idx = idx[idx >= pd.Timestamp(WARMUP_START)]
    a, b = a.loc[idx], b.loc[idx]
    ratio = a["raw"] / b["raw"]
    z = ((ratio - ratio.rolling(WINDOW).mean()) / ratio.rolling(WINDOW).std()).values
    oa, ob, ca, cb = a["open"].values, b["open"].values, a["close"].values, b["close"].values
    n = len(idx)
    ret = np.zeros(n)
    active = np.zeros(n, dtype=bool)
    trades = []

    pos, blocked, pending = 0, False, None
    trade_mult, trade_entry = 1.0, None
    start_i = int(np.searchsorted(idx.values, np.datetime64(START)))  # 백테스트 시작일 이전엔 시그널만 계산

    for i in range(n):
        mult = 1.0
        pos_before = pos
        if i >= start_i and pending is not None:
            # 1) 청산 (평균 복귀 / 하드스탑 / 반대 전환): 전일 종가 -> 당일 시가, 비용 차감
            do_exit = pos != 0 and (pending in ("exit", "stop") or (pending == "enter_long") != (pos == 1))
            if do_exit:
                m = (1 + pos * LEG * (oa[i] / ca[i - 1] - 1) - pos * LEG * (ob[i] / cb[i - 1] - 1)) * (1 - COST)
                mult *= m
                trade_mult *= m
                trades.append({"pair": name, "sector": sector, "entry": trade_entry, "exit": idx[i],
                               "side": "A롱/B숏" if pos == 1 else "A숏/B롱", "ret": trade_mult - 1,
                               "reason": "하드스탑" if pending == "stop" else
                                         ("반대 신호 전환" if pending.startswith("enter") else "평균 복귀")})
                pos, trade_mult, trade_entry = 0, 1.0, None
            # 2) 진입: 당일 시가 -> 당일 종가, 비용 차감
            if pending in ("enter_long", "enter_short") and pos == 0:
                pos = 1 if pending == "enter_long" else -1
                m = (1 - COST) * (1 + pos * LEG * (ca[i] / oa[i] - 1) - pos * LEG * (cb[i] / ob[i] - 1))
                mult *= m
                trade_mult, trade_entry = m, idx[i]
            elif pos != 0:  # 보유 유지: 전일 종가 -> 당일 종가
                m = 1 + pos * LEG * (ca[i] / ca[i - 1] - 1) - pos * LEG * (cb[i] / cb[i - 1] - 1)
                mult *= m
                trade_mult *= m
        elif pos != 0:
            m = 1 + pos * LEG * (ca[i] / ca[i - 1] - 1) - pos * LEG * (cb[i] / cb[i - 1] - 1)
            mult *= m
            trade_mult *= m
        pending = None
        ret[i] = mult - 1
        active[i] = (pos_before != 0) or (pos != 0)

        # 3) 종가 Z로 익일 행동 결정
        zt = z[i]
        if np.isnan(zt):
            continue
        if abs(zt) < EXIT_Z:
            blocked = False
        if pos == 0:
            if abs(zt) >= STOP_Z:
                blocked = True
            elif not blocked and zt <= -ENTRY_Z:
                pending = "enter_long"
            elif not blocked and zt >= ENTRY_Z:
                pending = "enter_short"
        else:
            if abs(zt) >= STOP_Z:
                pending, blocked = "stop", True
            elif abs(zt) < EXIT_Z:
                pending = "exit"
            elif pos == 1 and zt >= ENTRY_Z:
                pending = "enter_short"
            elif pos == -1 and zt <= -ENTRY_Z:
                pending = "enter_long"

    out = pd.DataFrame({"ret": ret, "active": active}, index=idx).loc[START:END]
    return out, pd.DataFrame(trades)


def build_portfolio(data: dict, master: pd.DatetimeIndex):
    rets, actives, avails, all_trades, pair_info = {}, {}, {}, [], []
    for sector, pairs in PAIRS.items():
        for a, b in pairs:
            if a not in data or b not in data:
                print(f"[경고] 페어 {a}/{b} 제외 (데이터 없음)")
                continue
            name = f"{a}/{b}"
            out, trades = simulate_pair(data[a], data[b], name, sector)
            out = out.reindex(master)
            rets[name] = out["ret"].fillna(0.0)
            actives[name] = out["active"].fillna(False).astype(bool)
            first = data[a].index.intersection(data[b].index).min()
            avails[name] = pd.Series(master >= max(first + pd.Timedelta(days=WINDOW * 1.5), pd.Timestamp(START)),
                                     index=master)
            all_trades.append(trades)
            pair_info.append({"pair": name, "sector": sector, "first_date": max(first, pd.Timestamp(START)),
                              "trades": len(trades),
                              "win_rate": (trades["ret"] > 0).mean() if len(trades) else np.nan,
                              "avg_ret": trades["ret"].mean() if len(trades) else np.nan,
                              "stops": int((trades["reason"] == "하드스탑").sum()) if len(trades) else 0})
    rets, actives, avails = pd.DataFrame(rets), pd.DataFrame(actives), pd.DataFrame(avails)
    n_active = actives.sum(axis=1)
    port_ret = (rets * actives).sum(axis=1) / n_active.replace(0, np.nan)
    port_ret = port_ret.fillna(0.0)   # 활성 페어 없음 -> 현금
    equity = (1 + port_ret).cumprod()
    return equity, port_ret, n_active, avails.sum(axis=1), pd.concat(all_trades, ignore_index=True), pd.DataFrame(pair_info)


# ---------------------------------------------------------------- 지표
def metrics(equity: pd.Series) -> dict:
    daily = equity.pct_change().fillna(equity.iloc[0] - 1.0)
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    dd = equity / equity.cummax() - 1
    std = daily.std()
    return {"Total": equity.iloc[-1] - 1, "CAGR": equity.iloc[-1] ** (1 / years) - 1,
            "MDD": dd.min(), "MDDDate": dd.idxmin(),
            "Sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
            "Vol": std * np.sqrt(TRADING_DAYS), "equity": equity, "dd": dd}


def yearly_returns(equity: pd.Series) -> pd.Series:
    ye = equity.groupby(equity.index.year).last()
    return ye / ye.shift(1).fillna(1.0) - 1


def print_tables(strat_1x, strat_lev, bench, trades, pair_info, n_active, n_avail, first, last):
    n = len(trades)
    wins = int((trades["ret"] > 0).sum())
    print(f"\n### S&P 500 50페어 Stat-Arb 포트폴리오 vs SPY 단순 보유 ({first.date()} ~ {last.date()}, "
          f"편도 비용 {COST:.2%}/레그, 현금이자 연 {RF_ANNUAL:.1%})\n")
    print("| 지표 | 50페어 1.0배수 (현금이자 포함) | 50페어 2.5배 레버리지 (헤지펀드 표준) | SPY 단순 보유 (시장) |")
    print("|---|---:|---:|---:|")
    print(f"| 총수익률 | {strat_1x['Total']:+.2%} | {strat_lev['Total']:+.2%} | {bench['Total']:+.2%} |")
    print(f"| CAGR | {strat_1x['CAGR']:+.2%} | {strat_lev['CAGR']:+.2%} | {bench['CAGR']:+.2%} |")
    print(f"| 30년 MDD | {strat_1x['MDD']:.2%} ({strat_1x['MDDDate'].date()}) | {strat_lev['MDD']:.2%} ({strat_lev['MDDDate'].date()}) | {bench['MDD']:.2%} ({bench['MDDDate'].date()}) |")
    print(f"| 샤프 지수 | {strat_1x['Sharpe']:.2f} | {strat_lev['Sharpe']:.2f} | {bench['Sharpe']:.2f} |")
    print(f"| 연환산 변동성 | {strat_1x['Vol']:.2%} | {strat_lev['Vol']:.2%} | {bench['Vol']:.2%} |")
    print(f"| 총 거래 횟수 | {n:,}회 | {n:,}회 | 1회 (매수 후 보유) |")
    print(f"| 평균 승률 | {wins / n:.1%} (전체 거래) | {wins / n:.1%} (전체 거래) | - |")

    print(f"\n### 거래 통계 (총 {n:,}회, Z 진입 ±{ENTRY_Z} / 청산 ±{EXIT_Z} / 하드스탑 ±{STOP_Z})\n")
    print("| 항목 | 값 |")
    print("|---|---:|")
    print(f"| 평균 수익률 / 거래 | {trades['ret'].mean():+.3%} |")
    print(f"| 평균 이익 거래 / 평균 손실 거래 | {trades.loc[trades['ret'] > 0, 'ret'].mean():+.2%} / "
          f"{trades.loc[trades['ret'] <= 0, 'ret'].mean():+.2%} |")
    print(f"| 평균 보유 기간 | {(trades['exit'] - trades['entry']).dt.days.mean():.0f}일 |")
    for reason, g in trades.groupby("reason"):
        print(f"| 청산 사유: {reason} | {len(g):,}회 ({len(g) / n:.0%}), 평균 {g['ret'].mean():+.2%} |")
    print(f"| 일평균 활성 페어 수 | {n_active.mean():.1f}개 (최대 {int(n_active.max())}개) |")
    print(f"| 활성 페어 0개인 날 | {(n_active == 0).sum()}일 / {len(n_active)}일 |")
    print(f"| 사용 가능 페어 수 | 1995년 {int(n_avail.iloc[0])}개 -> 2026년 {int(n_avail.iloc[-1])}개 |")

    print("\n### 섹터별 성과\n")
    print("| 섹터 | 페어 | 거래 수 | 승률 | 평균 수익률/거래 | 하드스탑 |")
    print("|---|---|---:|---:|---:|---:|")
    for sector, g in trades.groupby("sector", sort=False):
        pairs = ", ".join(pair_info.loc[pair_info["sector"] == sector, "pair"])
        print(f"| {sector} | {pairs} | {len(g):,} | {(g['ret'] > 0).mean():.1%} | {g['ret'].mean():+.3%} | "
              f"{int((g['reason'] == '하드스탑').sum())} |")

    print("\n### 페어별 상위/하위 5개 (평균 수익률/거래 기준)\n")
    print("| 페어 | 섹터 | 데이터 시작 | 거래 수 | 승률 | 평균 수익률/거래 |")
    print("|---|---|---|---:|---:|---:|")
    ranked = pair_info.dropna(subset=["avg_ret"]).sort_values("avg_ret", ascending=False)
    for _, r in pd.concat([ranked.head(5), ranked.tail(5)]).iterrows():
        print(f"| {r['pair']} | {r['sector']} | {r['first_date'].date()} | {r['trades']} | {r['win_rate']:.0%} | {r['avg_ret']:+.3%} |")

    ys_1x = yearly_returns(strat_1x["equity"])
    ys_lev = yearly_returns(strat_lev["equity"])
    yb = yearly_returns(bench["equity"])
    print("\n### 연도별 수익률 (2.5배 레버리지 vs SPY)\n")
    print("| 연도 | Stat-Arb(2.5x) | SPY | 초과 수익 | 연도 | Stat-Arb(2.5x) | SPY | 초과 수익 |")
    print("|---|---:|---:|---:|---|---:|---:|---:|")
    years = list(ys_lev.index)
    half = (len(years) + 1) // 2
    for i in range(half):
        row = ""
        for y in (years[i], years[i + half] if i + half < len(years) else None):
            if y is None:
                row += "|  |  |  |  "
                continue
            tag = " (~8월)" if y == last.year and last.month < 12 else ""
            row += f"| {y}{tag} | {ys_lev[y]:+.1%} | {yb[y]:+.1%} | {(ys_lev[y] - yb[y]) * 100:+.1f}%p "
        print(row + "|")


# ---------------------------------------------------------------- 차트 (Plotly, 3패널)
def plot(strat_1x, strat_lev, bench, n_active, n_avail):
    idx = strat_lev["equity"].index
    custom = np.column_stack([
        strat_1x["equity"].values, (strat_1x["equity"].values - 1) * 100, strat_1x["dd"].values * 100,
        strat_lev["equity"].values, (strat_lev["equity"].values - 1) * 100, strat_lev["dd"].values * 100,
        bench["equity"].values, (bench["equity"].values - 1) * 100, bench["dd"].values * 100,
        n_active.values, n_avail.values,
    ])
    strat1_tip = ("<b>%{x|%Y-%m-%d}</b><br>자산가치: %{customdata[0]:.3f}배<br>누적 수익률: %{customdata[1]:+.1f}%<br>"
                  "고점 대비 낙폭: %{customdata[2]:.1f}%<br>활성 페어: %{customdata[9]:.0f} / %{customdata[10]:.0f}개"
                  "<extra>50페어 1.0x (현금이자 포함)</extra>")
    strat_tip = ("<b>%{x|%Y-%m-%d}</b><br>자산가치: %{customdata[3]:.3f}배<br>누적 수익률: %{customdata[4]:+.1f}%<br>"
                 "고점 대비 낙폭: %{customdata[5]:.1f}%<br>활성 페어: %{customdata[9]:.0f} / %{customdata[10]:.0f}개"
                 "<extra>50페어 2.5x 레버리지</extra>")
    bench_tip = ("<b>%{x|%Y-%m-%d}</b><br>자산가치: %{customdata[6]:.3f}배<br>누적 수익률: %{customdata[7]:+.1f}%<br>"
                 "고점 대비 낙폭: %{customdata[8]:.1f}%<extra>SPY 단순 보유</extra>")

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.46, 0.24, 0.30],
        subplot_titles=("누적 자산 성장 곡선 (로그 스케일, 시작 = 1.0)",
                        "일별 동시 활성 페어 수 (포지션 보유 중인 페어) 와 사용 가능 페어 수",
                        "언더워터 플롯 (고점 대비 낙폭 %)"),
    )
    # 상단
    fig.add_trace(go.Scatter(x=idx, y=bench["equity"].values, name="SPY 단순 보유", mode="lines",
                             line=dict(color=COLORS["bench"], width=1.8, dash="dot"), customdata=custom,
                             hovertemplate=bench_tip, legendgroup="bench"), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=strat_1x["equity"].values, name="50페어 1.0x (현금이자 3.5%)", mode="lines",
                             line=dict(color=COLORS["strategy_1x"], width=1.8, dash="dash"), customdata=custom,
                             hovertemplate=strat1_tip, legendgroup="strat_1x"), 1, 1)
    fig.add_trace(go.Scatter(x=idx, y=strat_lev["equity"].values, name=f"50페어 {LEVERAGE:.1f}x 레버리지 (헤지펀드 표준)", mode="lines",
                             line=dict(color=COLORS["strategy_lev"], width=2.4), customdata=custom,
                             hovertemplate=strat_tip, legendgroup="strat_lev"), 1, 1)
    for m, name, col in ((bench, "SPY", COLORS["bench"]),
                         (strat_1x, "Stat-Arb 1x", COLORS["strategy_1x"]),
                         (strat_lev, f"Stat-Arb {LEVERAGE:.1f}x", COLORS["strategy_lev"])):
        fig.add_annotation(x=idx[-1], y=np.log10(m["equity"].iloc[-1]), xref="x", yref="y",
                           text=f"{name}: {m['equity'].iloc[-1]:.2f}배", showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=col))
    # 중단
    fig.add_trace(go.Scatter(x=idx, y=n_avail.values, name="사용 가능 페어 수 (데이터 보유)", mode="lines",
                             line=dict(color=COLORS["avail"], width=1.4, dash="dash"),
                             hovertemplate="사용 가능 페어: %{y:.0f}개<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=idx, y=n_active.values, name="활성 페어 수 (포지션 보유)", mode="lines",
                             line=dict(color=COLORS["active"], width=1.2), fill="tozeroy",
                             fillcolor="rgba(42,120,214,0.25)",
                             hovertemplate="활성 페어: %{y:.0f}개<extra></extra>"), 2, 1)
    # 하단
    fig.add_trace(go.Scatter(x=idx, y=bench["dd"].values * 100, name="SPY 낙폭", mode="lines",
                             line=dict(color=COLORS["bench"], width=1.4, dash="dot"), fill="tozeroy",
                             fillcolor="rgba(122,122,122,0.12)", customdata=custom, hovertemplate=bench_tip,
                             legendgroup="bench", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=idx, y=strat_1x["dd"].values * 100, name="Stat-Arb 1x 낙폭", mode="lines",
                             line=dict(color=COLORS["strategy_1x"], width=1.4, dash="dash"), fill="tozeroy",
                             fillcolor="rgba(42,120,214,0.12)", customdata=custom, hovertemplate=strat1_tip,
                             legendgroup="strat_1x", showlegend=False), 3, 1)
    fig.add_trace(go.Scatter(x=idx, y=strat_lev["dd"].values * 100, name=f"Stat-Arb {LEVERAGE:.1f}x 낙폭", mode="lines",
                             line=dict(color=COLORS["strategy_lev"], width=1.8), fill="tozeroy",
                             fillcolor="rgba(31,63,143,0.18)", customdata=custom, hovertemplate=strat_tip,
                             legendgroup="strat_lev", showlegend=False), 3, 1)
    for m, color, label, ax, ay in ((strat_1x, COLORS["strategy_1x"], "Stat-Arb 1x", -150, -30),
                                    (strat_lev, COLORS["strategy_lev"], f"Stat-Arb {LEVERAGE:.1f}x", -160, -50),
                                    (bench, COLORS["bench"], "SPY", 150, -30)):
        x, y = m["MDDDate"], m["MDD"] * 100
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers", name=f"{label} MDD 최저점",
                                 marker=dict(size=10, color=color, line=dict(color=SURFACE, width=1.5)),
                                 hovertemplate=f"<b>%{{x|%Y-%m-%d}}</b><br>{label} MDD 최저점: %{{y:.2f}}%<extra></extra>",
                                 showlegend=False), 3, 1)
        fig.add_annotation(x=x, y=y, xref="x3", yref="y3", text=f"{label} MDD {y:.1f}% ({x.date()})",
                           showarrow=True, arrowhead=0, arrowcolor=color, arrowwidth=1.2, ax=ax, ay=ay,
                           font=dict(size=11, color=INK), bgcolor=SURFACE, bordercolor=color, borderwidth=1, borderpad=4)

    fig.update_layout(
        title=dict(text=f"S&P 500 섹터별 50페어 1:1 Long/Short Stat-Arb 포트폴리오 vs SPY | {START} ~ {END}, "
                        f"Z 진입 ±{ENTRY_Z} / 청산 ±{EXIT_Z}, 현금이자 {RF_ANNUAL:.1%}, 레버리지 {LEVERAGE:.1f}x", x=0.01, font=dict(size=16, color=INK)),
        template="plotly_white", hovermode="x unified", height=1000, width=1400,
        font=dict(size=12, color=INK2), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=12, color=INK)),
        hoverlabel=dict(bgcolor=SURFACE, font=dict(size=12, color=INK), namelength=-1),
        margin=dict(l=70, r=90, t=90, b=100),
    )
    fig.update_annotations(font=dict(color=INK))
    lo = min(strat_1x["equity"].min(), strat_lev["equity"].min(), bench["equity"].min())
    hi = max(strat_1x["equity"].max(), strat_lev["equity"].max(), bench["equity"].max())
    ticks = [t for t in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0] if lo * 0.8 <= t <= hi * 1.3]
    fig.update_yaxes(title_text="누적 자산 (배)", type="log", row=1, col=1, tickvals=ticks,
                     ticktext=[f"{t:g}" for t in ticks], gridcolor="#e6e5e0", zeroline=False)
    fig.update_yaxes(title_text="페어 수", row=2, col=1, range=[0, max(n_avail.max(), 1) * 1.08],
                     gridcolor="#e6e5e0", zeroline=False)
    ymin = min(strat_1x["MDD"], strat_lev["MDD"], bench["MDD"]) * 100
    fig.update_yaxes(title_text="낙폭 (%)", row=3, col=1, range=[ymin - 8, 3], gridcolor="#e6e5e0",
                     zeroline=True, zerolinecolor="#c9c8c3")
    fig.update_xaxes(type="date", range=[idx[0] - pd.Timedelta(days=60), idx[-1] + pd.Timedelta(days=60)],
                     showgrid=False, showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor="#c9c8c3", spikethickness=1, spikedash="dot")
    fig.update_xaxes(title_text="날짜", row=3, col=1, rangeslider=dict(visible=False))
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)  # 단독 실행 HTML, 창 안 띄움


# ---------------------------------------------------------------- 메인
def main():
    data = load_all()
    spy = data[BENCH].loc[START:END]
    master = spy.index
    equity_raw, port_ret, n_active, n_avail, trades, pair_info = build_portfolio(data, master)
    
    # 1) 현금 담보 이자 (일간 복리 계산)
    rf_daily = (1 + RF_ANNUAL) ** (1 / TRADING_DAYS) - 1
    
    # 2) 1.0배수 포트폴리오 (스프레드 알파 + 현금 이자)
    port_ret_1x = port_ret + rf_daily
    equity_1x = (1 + port_ret_1x).cumprod()
    strat_1x = metrics(equity_1x)
    
    # 3) 2.5배 레버리지 포트폴리오 (헤지펀드 표준 Gross Exposure)
    port_ret_lev = port_ret * LEVERAGE + rf_daily
    equity_lev = (1 + port_ret_lev).cumprod()
    strat_lev = metrics(equity_lev)
    
    # 4) 벤치마크 (SPY)
    bench_eq = spy["close"] / spy["open"].iloc[0] * (1 - COST)
    bench = metrics(bench_eq)

    first, last = master[0], master[-1]
    print(f"데이터: {first.date()} ~ {last.date()}, {len(master)}거래일 | 페어 {len(pair_info)}개 "
          f"({len(PAIRS)}개 섹터) | 수정주가 기준")
    print_tables(strat_1x, strat_lev, bench, trades, pair_info, n_active, n_avail, first, last)
    plot(strat_1x, strat_lev, bench, n_active, n_avail)
    print(f"\n차트 저장 완료: {OUT_HTML}")
    print(f"\n차트 저장 완료: {OUT_HTML}")


if __name__ == "__main__":
    main()
