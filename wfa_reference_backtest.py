"""
클립 19·37 워크포워드 촬영 참고값 재현 스크립트 (강사용, 2026-10-30 규칙 정정 반영)

- 대상: KODEX 200(069500) 일봉. 신호는 오늘 종가, 체결은 다음 거래일 시가. 비용 편도 0.025%(왕복 0.05%, ETF 거래세 없음)
- 추세추종: 종가 > N일 단순이동평균이면 보유, 아니면 현금
- 워크포워드: 훈련 2년 → 검증 1년 → 적용 1년을 1년씩 밀며 8 Fold (적용 2019 ~ 2026)
- 클립 19 판정 (지금 교재):
    훈련 우위 = 후보 훈련 CAGR − 같은 기간 200일선 CAGR, 검증 우위도 같은 방식
    훈련 우위 1%p 이상인 후보만 상대 WFE = 검증 우위 ÷ 훈련 우위 × 100 계산 (작거나 0 이하면 탈락)
    상대 WFE 60% 이상 통과 → 통과한 것 중 검증 CAGR 1등을 적용, 없으면 200일선
- 비교용으로 영상 당시 교재 규칙(최소 우위 없음 + 없으면 단순 보유)과 민감도(최소 우위, 비용, 후보 수)도 출력
- 적용 구간을 이을 때 각 규칙의 전체 기간 수익률을 잘라 붙입니다. 그래서 Fold가 바뀌는 첫날의 전환 매매 비용과
  그날 밤사이 수익은 근사입니다(이어 붙인 CAGR 차이 0.1%p 안팎, MDD는 같음).
- 클립 37: 완충 밴드 가설(종가 > 200일선 x (1+밴드)일 때만 진입, 청산은 종가 < 200일선) vs 기존 200일선
  Fold별 훈련 우위·검증 우위·상대 WFE와 적격 판정(8개 중 5개 이상 통과)을 밴드 1·2·3·5%로 출력

실행:
    .venv/bin/python wfa_reference_backtest.py            # 2026-08-31까지 (클립 19와 같은 기간)
    .venv/bin/python wfa_reference_backtest.py 2026-10-29 # 클립 37처럼 최신 데이터까지
"""

import sys

import FinanceDataReader as fdr
import numpy as np
import pandas as pd

END = sys.argv[1] if len(sys.argv) > 1 else "2026-08-31"
COST_SIDE = 0.00025          # 편도 0.025% = 왕복 0.05%
RF = 0.03                    # 샤프 지수 무위험 수익률 (클립 07 정의)
MIN_EDGE = 0.01              # 훈련 우위 최소 1%p
PASS_WFE = 0.6               # 상대 WFE 60%
CANDS = [60, 100, 120, 150]  # 200일선은 잣대이자 기본값
STITCH_START = "2019-01-01"

px = fdr.DataReader("069500", "2014-01-01", END)[["Open", "Close"]].astype(float)
ON = px.Open / px.Close.shift(1) - 1   # 전일 종가 → 오늘 시가
ID = px.Close / px.Open - 1            # 오늘 시가 → 오늘 종가
BH = px.Close.pct_change().fillna(0.0)


def run_rule(signal: pd.Series, cost_side: float = COST_SIDE) -> tuple[pd.Series, pd.Series]:
    """오늘 종가 신호(1=보유) → 다음 거래일 시가부터 보유. (일별 수익률, 일별 보유 여부)"""
    pos = signal.shift(1).fillna(0.0)
    prev = pos.shift(1).fillna(0.0)
    r = (1 + prev * ON.fillna(0)) * (1 + pos * ID) - 1
    trade = (pos != prev).astype(float)
    return (1 + r) * (1 - trade * cost_side) - 1, pos


def ma_signal(n: int) -> pd.Series:
    ma = px.Close.rolling(n).mean()
    return (px.Close > ma).astype(float).where(ma.notna(), 0.0)


def band_signal(band: float) -> pd.Series:
    ma = px.Close.rolling(200).mean().values
    c = px.Close.values
    out, s = np.zeros(len(px)), 0.0
    for t in range(len(px)):
        if np.isnan(ma[t]):
            s = 0.0
        elif s == 0.0 and c[t] > ma[t] * (1 + band):
            s = 1.0
        elif s == 1.0 and c[t] < ma[t]:
            s = 0.0
        out[t] = s
    return pd.Series(out, index=px.index)


def cagr(r: pd.Series, a: str, b: str) -> float:
    seg = r.loc[a:b]
    years = ((seg.index[-1] - pd.Timestamp(a)).days + 1) / 365.25
    return (1 + seg).prod() ** (1 / years) - 1


def total(r: pd.Series, a: str, b: str) -> float:
    return (1 + r.loc[a:b]).prod() - 1


def summary(r: pd.Series) -> tuple[float, float, float]:
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    ex = r - RF / 252
    return eq.iloc[-1] ** (1 / years) - 1, (eq / eq.cummax() - 1).min(), ex.mean() / ex.std() * np.sqrt(252)


last = px.index[-1].strftime("%Y-%m-%d")
FOLDS = [
    dict(tr=(f"{y}-01-01", f"{y+1}-12-31"), va=(f"{y+2}-01-01", f"{y+2}-12-31"),
         ap=(f"{y+3}-01-01", f"{y+3}-12-31" if y + 3 < int(last[:4]) else last))
    for y in range(2016, 2024)
]


def walk_forward(cands, cost_side=COST_SIDE, min_edge=MIN_EDGE, fallback="200"):
    """클립 19 판정. min_edge=None 이면 '훈련 우위 > 0'(영상 당시), fallback='BH'면 단순 보유(영상 당시)."""
    R = {n: run_rule(ma_signal(n), cost_side) for n in set(cands) | {200}}
    rows, parts, poss = [], [], []
    for f in FOLDS:
        b_tr, b_va = cagr(R[200][0], *f["tr"]), cagr(R[200][0], *f["va"])
        info = []
        for n in cands:
            te = cagr(R[n][0], *f["tr"]) - b_tr
            ve = cagr(R[n][0], *f["va"]) - b_va
            eligible = te > 0 if min_edge is None else te >= min_edge
            wfe = ve / te if eligible else np.nan
            info.append(dict(n=n, te=te, ve=ve, wfe=wfe, ok=bool(eligible and ve >= PASS_WFE * te),
                             va=cagr(R[n][0], *f["va"])))
        passed = [i for i in info if i["ok"]]
        # 검증 CAGR이 같으면 더 긴 N (교재 규칙)
        choice = max(passed, key=lambda i: (i["va"], i["n"]))["n"] if passed else fallback
        a, b = f["ap"]
        if choice == "BH":
            parts.append(BH.loc[a:b]); poss.append(pd.Series(1.0, index=BH.loc[a:b].index))
        else:
            n = 200 if choice == "200" else choice
            parts.append(R[n][0].loc[a:b]); poss.append(R[n][1].loc[a:b])
        rows.append(dict(f=f, info=info, choice=choice))
    return rows, pd.concat(parts), pd.concat(poss), R


def pct(w: float) -> str:
    """상대 WFE 표시. 60% 기준선 근처는 반올림으로 통과처럼 보이지 않게 소수 한 자리."""
    return f"{w*100:.1f}%" if 50 <= w * 100 < 70 else f"{w*100:.0f}%"


def fmt_choice(c):
    return "단순 보유" if c == "BH" else ("200일선" if c == "200" else f"채택 {c}일")


print(f"# 데이터 끝: {last}\n")
print("## 클립 19 — 지금 교재 규칙 (훈련 우위 1%p 이상 + 상대 WFE 60% + 검증 CAGR 1등 + 없으면 200일선)\n")
rows, r, pos, R = walk_forward(CANDS)
print("| 적용 | 훈련 → 검증 | 후보별 훈련 우위 / 검증 우위 / 상대 WFE (✓ 통과) | 결정 | 적용 구간 수익 (적용한 것 / 200일선 / 단순 보유) |")
print("|---|---|---|---|---|")
for row in rows:
    f = row["f"]
    cells = []
    for i in row["info"]:
        if np.isnan(i["wfe"]):
            cells.append(f"{i['n']}일 {i['te']*100:+.1f}%p 탈락")
        else:
            cells.append(f"{i['n']}일 {i['te']*100:+.1f} / {i['ve']*100:+.1f} / {pct(i['wfe'])}{' ✓' if i['ok'] else ''}")
    a, b = f["ap"]
    n = 200 if row["choice"] == "200" else row["choice"]
    mine = R[n][0] if row["choice"] != "BH" else BH
    lab = a[:4] if b.endswith("12-31") else f"{a[:4]}-01~{b[5:7]}"
    print(f"| {lab} | {f['tr'][0][:4]}~{f['tr'][1][:4]} → {f['va'][0][:4]} | {'<br>'.join(cells)} | {fmt_choice(row['choice'])} | "
          f"{total(mine, a, b)*100:+.0f}% / {total(R[200][0], a, b)*100:+.0f}% / {total(BH, a, b)*100:+.0f}% |")

print(f"\n| 이어 붙인 적용 구간 ({STITCH_START[:7]} ~ {last[:7]}) | CAGR | MDD | 샤프 | 현금 비율 |")
print("|---|---:|---:|---:|---:|")
c, m, s = summary(r); print(f"| 지금 교재 규칙 | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | {(pos == 0).mean()*100:.0f}% |")
r200 = R[200][0].loc[STITCH_START:]; c, m, s = summary(r200)
print(f"| 고정 200일선 | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | {(R[200][1].loc[STITCH_START:] == 0).mean()*100:.0f}% |")
c, m, s = summary(BH.loc[STITCH_START:]); print(f"| (참고) 단순 보유 | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | — |")
_, r0, p0, _ = walk_forward(CANDS, min_edge=None, fallback="BH"); c, m, s = summary(r0)
print(f"| (참고) 영상 당시 교재 규칙: 최소 우위 없음 + 없으면 단순 보유 | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | {(p0 == 0).mean()*100:.0f}% |")
_, r1, p1, _ = walk_forward(CANDS, min_edge=None); c, m, s = summary(r1)
print(f"| (참고) 최소 우위 없음 + 없으면 200일선 | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | {(p1 == 0).mean()*100:.0f}% |")

print("\n### 민감도 (결과를 보고 규칙을 고르지 말 것 — 흔들림의 크기를 보려는 표)\n")
print("| 바꾼 조건 | CAGR | MDD | 샤프 | Fold별 결정 |")
print("|---|---:|---:|---:|---|")
for lab, kw in [("최소 우위 0.5%p", dict(min_edge=0.005)), ("최소 우위 2%p", dict(min_edge=0.02)),
                ("최소 우위 3%p", dict(min_edge=0.03)), ("비용 왕복 0.25%", dict(cost_side=0.00125)),
                ("후보 20~250일 (10일 간격)", dict(cands=list(range(20, 260, 10))))]:
    cands = kw.pop("cands", CANDS)
    rr, r_, _, _ = walk_forward(cands, **kw); c, m, s = summary(r_)
    print(f"| {lab} | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | {' '.join(fmt_choice(x['choice']).replace('채택 ', '') for x in rr)} |")
c, m, s = summary(run_rule(ma_signal(200), 0.00125)[0].loc[STITCH_START:])
print(f"| (비교) 고정 200일선, 비용 왕복 0.25% | {c*100:+.1f}% | {m*100:.1f}% | {s:.2f} | — |")

print("\n## 클립 37 — 완충 밴드 가설 vs 기존 200일선 (지금 교재 규칙: 훈련 우위 1%p 이상만 상대 WFE 계산)\n")
base = run_rule(ma_signal(200))[0]
print("| 밴드 | Fold별 훈련 우위 / 검증 우위 / 상대 WFE (✓ 통과) | 통과 Fold | 판정 | 영상 당시 규칙의 통과 Fold |")
print("|---|---|---:|---|---:|")
for band in [0.01, 0.02, 0.03, 0.05]:
    hyp = run_rule(band_signal(band))[0]
    cells, n_new, n_old = [], 0, 0
    for f in FOLDS:
        te = cagr(hyp, *f["tr"]) - cagr(base, *f["tr"])
        ve = cagr(hyp, *f["va"]) - cagr(base, *f["va"])
        ok_new = te >= MIN_EDGE and ve >= PASS_WFE * te
        ok_old = te > 0 and ve >= PASS_WFE * te
        n_new += ok_new; n_old += ok_old
        cells.append(f"{te*100:+.1f}/{ve*100:+.1f}/{'탈락' if te < MIN_EDGE else pct(ve / te)}{'✓' if ok_new else ''}")
    verdict = "적격 후보 (MDD·CAGR 조건도 확인)" if n_new >= 5 else "탈락"
    print(f"| {band*100:.0f}% | {' · '.join(cells)} | {n_new}/8 | {verdict} | {n_old}/8 |")
