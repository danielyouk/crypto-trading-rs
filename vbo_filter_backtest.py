"""
KODEX 200(069500) 래리 윌리엄스 변동성 돌파(VBO) 전략 백테스트 + 200일선 필터 검증
- 백테스트 기간: 2016-01-01 ~ 2026-08-31 (200일선 워밍업을 위해 2015-01-01부터 데이터 수신)
- 전략 1 (기본 VBO):  목표가 = 당일 시가 + 0.5 * (전일 고가 - 전일 저가)
                      당일 고가 >= 목표가 이면 목표가에 진입, 당일 장 마감 종가에 전량 청산
- 전략 2 (필터 VBO):  기본 VBO 조건 + 당일 시가 >= 200일 이동평균선(전일 종가까지로 계산)
- 벤치마크:          단순 보유 (Buy & Hold)
- 거래비용: 체결된 모든 매매마다 왕복 0.25% 차감
- 미래참조 방지: 목표가는 당일 시가와 전일 고저만 사용, 200일선은 전일 종가까지의 평균을 사용
- 차트: 2패널 대화형 Plotly 차트 (누적 자산 로그 곡선 / 언더워터 플롯) -> vbo_filter_comparison.html
"""

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

TICKER = "069500"  # KODEX 200
START, END = "2016-01-01", "2026-08-31"
WARMUP_START = "2015-01-01"  # 200일선 워밍업용
K = 0.5
MA_WINDOW = 200
COST = 0.0025  # 왕복 거래비용 0.25%
TRADING_DAYS = 252
OUT_HTML = "vbo_filter_comparison.html"

STYLE = {
    "단순 보유": ("단순 보유 (Buy & Hold)", "#7a7a7a", "dot"),
    "기본 VBO": ("기본 VBO (k=0.5)", "#eb6834", "solid"),
    "필터 VBO": ("200일선 필터 VBO", "#1f3f8f", "solid"),
}


def load_data() -> pd.DataFrame:
    df = fdr.DataReader(TICKER, WARMUP_START, END)
    df = df[["Open", "High", "Low", "Close"]].dropna().sort_index()
    return df


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["prev_range"] = (d["High"] - d["Low"]).shift(1)          # 전일 고가 - 전일 저가
    d["target"] = d["Open"] + K * d["prev_range"]              # 당일 시가 + 0.5 * 전일 변동폭
    d["ma200"] = d["Close"].rolling(MA_WINDOW).mean().shift(1)  # 전일 종가까지의 200일선
    d["breakout"] = d["High"] >= d["target"]
    d["uptrend"] = d["Open"] >= d["ma200"]
    return d


def simulate_vbo(d: pd.DataFrame, use_filter: bool) -> tuple[pd.Series, pd.Series]:
    """당일 종가 청산 기준 일별 수익률과 체결된 거래별 수익률 반환"""
    entry = d["breakout"] & d["prev_range"].notna()
    if use_filter:
        entry &= d["uptrend"]
    
    # 당일 목표가 매수 -> 당일 종가 매도 (왕복비용 차감)
    trade_ret = (d["Close"] / d["target"] - 1.0) - COST
    daily = pd.Series(0.0, index=d.index)
    daily.loc[entry] = trade_ret[entry].values
    return daily, trade_ret[entry]


def simulate_buy_hold(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    daily = d["Close"].pct_change().fillna(0.0)
    daily.iloc[0] = d["Close"].iloc[0] / d["Open"].iloc[0] - 1.0  # 첫날 시가 매수
    daily.iloc[-1] = (1 + daily.iloc[-1]) * (1 - COST) - 1        # 마지막날 청산, 왕복 비용 1회
    total = (1 + daily).prod() - 1
    return daily, pd.Series([total], index=[d.index[-1]])


def metrics(daily: pd.Series, trades: pd.Series) -> dict:
    equity = (1 + daily).cumprod()
    years = (daily.index[-1] - daily.index[0]).days / 365.25
    total_ret = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / years) - 1
    dd = equity / equity.cummax() - 1
    std = daily.std()
    sharpe = (daily.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else np.nan
    n_trades = len(trades)
    win_rate = (trades > 0).mean() if n_trades else np.nan
    return {
        "TotalReturn": total_ret, "CAGR": cagr, "MDD": dd.min(), "MDDDate": dd.idxmin(),
        "Sharpe": sharpe, "Trades": n_trades, "WinRate": win_rate,
        "Final": equity.iloc[-1], "equity": equity, "dd": dd,
    }


def print_table(results: dict, first: pd.Timestamp, last: pd.Timestamp) -> None:
    print(f"\n### KODEX 200(069500) VBO 전략 비교 ({first.date()} ~ {last.date()}, 왕복 거래비용 {COST:.2%})\n")
    print("| 전략 | 총수익률 | CAGR | MDD | 샤프 지수 | 총 거래 횟수 | 승률 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, m in results.items():
        wr = f"{m['WinRate']:.1%}" if not np.isnan(m["WinRate"]) else "-"
        print(f"| {name} | {m['TotalReturn']:+.2%} | {m['CAGR']:+.2%} | {m['MDD']:.2%} "
              f"| {m['Sharpe']:.2f} | {m['Trades']:,} | {wr} |")


def plot_interactive(results: dict) -> None:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            "<b>누적 자산 성장 곡선 (Cumulative Growth, Log Scale)</b>",
            "<b>낙폭 곡선 (Underwater Plot / Drawdown %)</b>"
        )
    )

    for name, m in results.items():
        label, color, dash = STYLE[name]
        dates = m["equity"].index
        eq_values = m["equity"].values
        cum_ret = (eq_values - 1.0) * 100
        dd_values = m["dd"].values * 100

        customdata = np.stack((
            [label] * len(dates),
            eq_values,
            cum_ret,
            dd_values
        ), axis=-1)

        # 패널 1: 누적 자산 성장 곡선
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=eq_values,
                mode="lines",
                name=label,
                line=dict(color=color, width=2, dash=dash),
                customdata=customdata,
                hovertemplate="<b>%{customdata[0]}</b><br>날짜: %{x|%Y-%m-%d}<br>자산가치: %{customdata[1]:.2f}배<br>누적 수익률: %{customdata[2]:+.2f}%<extra></extra>"
            ),
            row=1, col=1
        )

        # 패널 2: 언더워터 플롯
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=dd_values,
                mode="lines",
                name=f"{label} (MDD)",
                line=dict(color=color, width=1.5, dash=dash),
                showlegend=False,
                customdata=customdata,
                hovertemplate="<b>%{customdata[0]}</b><br>날짜: %{x|%Y-%m-%d}<br>낙폭(MDD): %{customdata[3]:.2f}%<extra></extra>"
            ),
            row=2, col=1
        )

        # MDD 최저점 Annotation
        mdd_date = m["MDDDate"]
        mdd_val = m["MDD"] * 100
        fig.add_annotation(
            x=mdd_date,
            y=mdd_val,
            xref="x2",
            yref="y2",
            text=f"<b>{name} MDD: {mdd_val:.1f}%</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            arrowcolor=color,
            ax=0,
            ay=25 if name == "단순 보유" else -25,
            bgcolor="rgba(255, 255, 255, 0.85)",
            bordercolor=color,
            borderwidth=1,
            font=dict(size=10, color="#111827")
        )

    fig.update_yaxes(type="log", title="자산 가치 (기준 = 1.0)", row=1, col=1, gridcolor="#e5e7eb")
    fig.update_yaxes(title="고점 대비 낙폭 (%)", row=2, col=1, gridcolor="#e5e7eb", zeroline=True, zerolinecolor="#9ca3af")
    fig.update_xaxes(title="날짜", row=2, col=1, gridcolor="#e5e7eb")

    fig.update_layout(
        title=dict(
            text="<b>KODEX 200 (069500) 변동성 돌파(VBO) 백테스트 & 200일선 필터 검증 (2016 ~ 2026)</b><br><span style='font-size:12px;color:#6b7280;'>왕복 거래비용 0.25% 적용 | 당일 목표가 돌파 시 매수, 당일 장 마감 종가 청산</span>",
            x=0.03, y=0.96
        ),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=40, t=100, b=50),
        height=800
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")


def main() -> None:
    raw = load_data()
    sig = build_signals(raw)
    d = sig.loc[START:END].copy()
    print(f"데이터: {d.index[0].date()} ~ {d.index[-1].date()}, {len(d)}거래일 "
          f"(워밍업 포함 원본 {len(raw)}행, 200일선 결측 {int(d['ma200'].isna().sum())}일)")

    results = {}
    daily, trades = simulate_buy_hold(d)
    results["단순 보유"] = metrics(daily, trades)
    for name, use_filter in [("기본 VBO", False), ("필터 VBO", True)]:
        daily, trades = simulate_vbo(d, use_filter)
        results[name] = metrics(daily, trades)

    print_table(results, d.index[0], d.index[-1])
    plot_interactive(results)
    print(f"\n대화형 차트 저장 완료: {OUT_HTML}")


if __name__ == "__main__":
    main()
