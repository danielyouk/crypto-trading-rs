"""
KB금융(105560) vs 신한지주(055550) 60일 롤링 Z-Score 롱 온리 페어 스위칭 백테스트
- 기간: 2019-01-01 ~ 2026-08-31 (60일 Z-Score 워밍업을 위해 2018-09-01부터 데이터 수신)
- Price Ratio = KB금융 종가 / 신한지주 종가
- Z_t = (Ratio_t - 어제까지 60일 롤링 평균) / (어제까지 60일 롤링 표준편차)
        -> 평균·표준편차 창에 당일 비율은 포함하지 않음 (rolling 뒤 shift(1)).
           "과거 분포에 비추어 오늘이 얼마나 이례적인가"를 묻기 위함. 클립 10~11의 200일선
           ("전일 종가까지")과 같은 규약.
- 규칙 (공매도 없음, 항상 한 종목 100% 보유):
    Z <= -1.5 : KB금융 저평가  -> KB금융 100%
    Z >= +1.5 : 신한지주 저평가 -> 신한지주 100%
    그 외      : 기존 포지션 유지
    첫 시그널 발생 전에는 현금(수익률 0) 대기
- 체결: 시그널은 당일 종가 Z로 판단, 체결은 익일 시가 (미래참조 방지). 클립 09 추세추종과 같은 규약.
- 비용: 편도 0.15% -> 스위칭 1회 = 매도 0.15% + 매수 0.15% = 0.30%, 최초 진입 0.15%
- 벤치마크: KB금융 단순 보유, 신한지주 단순 보유 (첫날 시가 매수, 편도 0.15% 1회)
- 산출: Markdown 비교표 + Plotly 대화형 2패널 차트 pair_trading_backtest_result.html
"""

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

A, B = "105560", "055550"
NAMES = {A: "KB금융", B: "신한지주"}
START, END = "2019-01-01", "2026-08-31"
WARMUP_START = "2018-09-01"
WINDOW = 60
ENTRY_Z = 1.5
COST = 0.0015  # 편도 0.15%
TRADING_DAYS = 252
OUT_HTML = "pair_trading_backtest_result.html"
COLORS = {A: "#2a78d6", B: "#eb6834", "strategy": "#1f3f8f", "z": "#52514e"}


# ---------------------------------------------------------------- 데이터/시그널
def load_prices() -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for code in (A, B):
        df = fdr.DataReader(code, WARMUP_START, END)
        opens[code], closes[code] = df["Open"], df["Close"]
    opens, closes = pd.DataFrame(opens), pd.DataFrame(closes)
    idx = closes.dropna().index
    return opens.loc[idx].sort_index(), closes.loc[idx].sort_index()


def zscore(closes: pd.DataFrame) -> pd.DataFrame:
    ratio = closes[A] / closes[B]
    mean = ratio.rolling(WINDOW).mean().shift(1)  # 어제까지의 60일 평균 (오늘 비율 제외)
    std = ratio.rolling(WINDOW).std().shift(1)    # 어제까지의 60일 표준편차
    return pd.DataFrame({"ratio": ratio, "mean": mean, "std": std, "z": (ratio - mean) / std})


def build_holdings(z: pd.Series) -> pd.Series:
    """시그널 당일 종가 Z -> 익일부터 보유. None = 현금 대기."""
    signal = pd.Series(np.where(z <= -ENTRY_Z, A, np.where(z >= ENTRY_Z, B, None)), index=z.index)
    return signal.shift(1).ffill()  # 익일 적용 + 시그널 없으면 유지


# ---------------------------------------------------------------- 시뮬레이션
def simulate_strategy(opens, closes, holdings) -> tuple[pd.Series, int, pd.Timestamp]:
    idx = holdings.index
    daily = pd.Series(0.0, index=idx)
    prev, switches, first_entry = None, 0, None
    for i, d in enumerate(idx):
        cur = holdings[d]
        if pd.isna(cur):
            continue  # 현금 대기
        if prev is None:  # 최초 진입: 시가 매수 + 편도 비용
            daily[d] = (1 - COST) * closes.at[d, cur] / opens.at[d, cur] - 1
            first_entry = d
        elif cur == prev:
            daily[d] = closes.at[d, cur] / closes.at[idx[i - 1], cur] - 1
        else:  # 스위칭: 구 종목 시가 매도(편도) + 신 종목 시가 매수(편도)
            sell_leg = opens.at[d, prev] / closes.at[idx[i - 1], prev] * (1 - COST)
            buy_leg = (1 - COST) * closes.at[d, cur] / opens.at[d, cur]
            daily[d] = sell_leg * buy_leg - 1
            switches += 1
        prev = cur
    return daily, switches, first_entry


def simulate_buy_hold(opens, closes, code, idx) -> pd.Series:
    daily = closes.loc[idx, code].pct_change().fillna(0.0)
    daily.iloc[0] = (1 - COST) * closes.at[idx[0], code] / opens.at[idx[0], code] - 1
    return daily


def metrics(daily: pd.Series, switches) -> dict:
    equity = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    dd = equity / equity.cummax() - 1
    std = daily.std()
    return {
        "Total": equity.iloc[-1] - 1,
        "CAGR": equity.iloc[-1] ** (1 / years) - 1,
        "MDD": dd.min(), "MDDDate": dd.idxmin(),
        "Sharpe": daily.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else np.nan,
        "Switches": switches, "equity": equity, "dd": dd,
    }


# ---------------------------------------------------------------- 출력
def print_table(results: dict, first, last) -> None:
    print(f"\n### KB금융 vs 신한지주 롱 온리 페어 스위칭 ({first.date()} ~ {last.date()}, "
          f"60일 Z-Score, 진입 |Z| >= {ENTRY_Z}, 편도 비용 {COST:.2%})\n")
    print("| 전략 | 총수익률 | CAGR | MDD | 샤프 지수 | 총 스위칭 횟수 |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        sw = "-" if m["Switches"] is None else f"{m['Switches']}"
        print(f"| {name} | {m['Total']:+.2%} | {m['CAGR']:+.2%} | "
              f"{m['MDD']:.2%} ({m['MDDDate'].date()}) | {m['Sharpe']:.2f} | {sw} |")


def plot(results: dict, zs: pd.DataFrame, holdings: pd.Series) -> None:
    idx = holdings.index
    hold_kr = holdings.map(lambda c: "현금 대기" if pd.isna(c) else NAMES[c])
    z = zs.loc[idx, "z"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.62, 0.38],
        subplot_titles=("정규화 누적 자산 (시작 = 100)",
                        "60일 롤링 Z-Score (KB금융 / 신한지주 가격 비율, 기준선 = 어제까지 60일)"),
    )
    # 상단: 누적 자산
    def eq_trace(name, equity, color, dash=None, width=2.0):
        custom = np.column_stack([hold_kr.values, z.values, equity.values * 100])
        return go.Scatter(
            x=equity.index, y=equity.values * 100, name=name, mode="lines",
            line=dict(color=color, width=width, dash=dash), customdata=custom,
            hovertemplate=("<b>%{x|%Y-%m-%d}</b><br>" + name + " 누적 자산: %{y:.1f}<br>"
                           "전략 보유 종목: %{customdata[0]}<br>Z-Score: %{customdata[1]:.2f}<extra></extra>"),
        )
    fig.add_trace(eq_trace("KB금융 단순 보유", results["KB금융 단순 보유"]["equity"], COLORS[A], "dot", 1.6), 1, 1)
    fig.add_trace(eq_trace("신한지주 단순 보유", results["신한지주 단순 보유"]["equity"], COLORS[B], "dot", 1.6), 1, 1)
    fig.add_trace(eq_trace("페어 스위칭 전략", results["페어 스위칭 전략"]["equity"], COLORS["strategy"], None, 2.4), 1, 1)
    # 스위칭 시점 마커
    strat_eq = results["페어 스위칭 전략"]["equity"]
    changed = (holdings != holdings.shift(1)) & holdings.notna()
    for code, symbol in ((A, "triangle-up"), (B, "triangle-down")):
        pts = changed & (holdings == code)
        fig.add_trace(go.Scatter(
            x=strat_eq.index[pts], y=strat_eq.values[pts] * 100, mode="markers",
            name=f"{NAMES[code]}(으)로 스위칭", marker=dict(symbol=symbol, size=10, color=COLORS[code],
                                                       line=dict(color="#fcfcfb", width=1)),
            customdata=np.column_stack([z.values[pts]]),
            hovertemplate=(f"<b>%{{x|%Y-%m-%d}}</b><br>{NAMES[code]} 매수 전환<br>"
                           "Z-Score: %{customdata[0]:.2f}<br>누적 자산: %{y:.1f}<extra></extra>"),
        ), 1, 1)
    # 하단: Z-Score + 임계선
    fig.add_trace(go.Scatter(
        x=z.index, y=z.values, name="Z-Score (60일)", mode="lines",
        line=dict(color=COLORS["z"], width=1.6),
        customdata=np.column_stack([hold_kr.values, strat_eq.values * 100]),
        hovertemplate=("<b>%{x|%Y-%m-%d}</b><br>Z-Score: %{y:.2f}<br>"
                       "전략 보유 종목: %{customdata[0]}<br>전략 누적 자산: %{customdata[1]:.1f}<extra></extra>"),
    ), 2, 1)
    for level, color, label in ((ENTRY_Z, COLORS[B], f"+{ENTRY_Z} 신한지주 매수선"),
                                (0.0, "#9a9a95", "0.0 중심선"),
                                (-ENTRY_Z, COLORS[A], f"-{ENTRY_Z} KB금융 매수선")):
        fig.add_hline(y=level, line_color=color, line_width=1.4, line_dash="dash", row=2, col=1,
                      annotation_text=label, annotation_position="top right",
                      annotation_font=dict(size=11, color=color))

    fig.update_layout(
        title=dict(text=f"KB금융(105560) vs 신한지주(055550) 롱 온리 페어 스위칭 | {START} ~ {END}, "
                        f"편도 비용 {COST:.2%}, 익일 시가 체결", x=0.01, font=dict(size=16)),
        template="plotly_white", hovermode="x unified", height=820, width=1400,
        font=dict(size=12), paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=60, r=30, t=90, b=90),
    )
    fig.update_yaxes(title_text="누적 자산 (시작 = 100)", type="log", row=1, col=1,
                     tickvals=[50, 70, 100, 150, 200, 300, 400, 500],
                     ticktext=["50", "70", "100", "150", "200", "300", "400", "500"])
    fig.update_yaxes(title_text="Z-Score", row=2, col=1)
    fig.update_xaxes(title_text="날짜", row=2, col=1, rangeslider=dict(visible=False))
    fig.write_html(OUT_HTML, include_plotlyjs=True, full_html=True)  # 단독 실행 HTML


# ---------------------------------------------------------------- 메인
def main() -> None:
    opens, closes = load_prices()
    zs = zscore(closes)
    holdings = build_holdings(zs["z"]).loc[START:END]
    opens, closes = opens.loc[holdings.index], closes.loc[holdings.index]
    n_nan = int(zs.loc[holdings.index, "z"].isna().sum())

    strat_daily, switches, first_entry = simulate_strategy(opens, closes, holdings)
    results = {
        "KB금융 단순 보유": metrics(simulate_buy_hold(opens, closes, A, holdings.index), None),
        "신한지주 단순 보유": metrics(simulate_buy_hold(opens, closes, B, holdings.index), None),
        "페어 스위칭 전략": metrics(strat_daily, switches),
    }
    first, last = holdings.index[0], holdings.index[-1]
    print(f"데이터: {first.date()} ~ {last.date()}, {len(holdings)}거래일, Z-Score 결측 {n_nan}일 | "
          f"전략 최초 진입 {first_entry.date()} ({NAMES[holdings[first_entry]]}), 그 전은 현금 대기")
    print_table(results, first, last)
    held = holdings.dropna()
    print(f"\n보유 일수: KB금융 {int((held == A).sum())}일, 신한지주 {int((held == B).sum())}일, "
          f"현금 대기 {int(holdings.isna().sum())}일")
    plot(results, zs, holdings)
    print(f"\n차트 저장 완료: {OUT_HTML}")


if __name__ == "__main__":
    main()
