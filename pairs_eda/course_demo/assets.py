"""Static and animated assets generated from one verified WFA bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import koreanize_matplotlib  # noqa: F401

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel, Field

from pairs_eda.course_demo.bundle import DISCLAIMER_KO, WfaResultBundle
from pairs_eda.course_demo.korea_cash import KoreaCashRegimeOutput

NAVY = "#132238"
BLUE = "#1F77B4"
TEAL = "#168C86"
ORANGE = "#E1842B"
RED = "#C7473D"
GRAY = "#667085"
LIGHT = "#EEF2F6"


class RenderSeminarAssetsInput(BaseModel):
    """Validated input for the seminar chart package."""

    bundle: WfaResultBundle
    output_dir: Path
    width_px: int = Field(default=1920, ge=960)
    height_px: int = Field(default=1080, ge=540)
    dpi: int = Field(default=160, ge=72)


class RenderSeminarAssetsOutput(BaseModel):
    """Created seminar charts and presenter notes."""

    png_files: list[Path]
    svg_files: list[Path]
    notes_file: Path


class RenderDetailAssetsInput(BaseModel):
    """Validated input for WFA replay and loss-review media."""

    bundle: WfaResultBundle
    output_dir: Path
    width_px: int = Field(default=960, ge=640)
    height_px: int = Field(default=544, ge=360)
    fps: int = Field(default=5, ge=2, le=12)
    duration_seconds: int = Field(default=7, ge=4, le=15)
    max_gif_bytes: int = Field(default=2_000_000, ge=500_000)


class RenderDetailAssetsOutput(BaseModel):
    """Created detail-page GIF and MP4 files."""

    replay_gif: Path
    replay_mp4: Path
    loss_review_gif: Path
    loss_review_mp4: Path
    paper_approval_gif: Path
    paper_approval_mp4: Path
    sizes_bytes: dict[str, int]


class RenderKoreaAssetsInput(BaseModel):
    """Validated input for the Korean ETF/cash detail-page media."""

    result: KoreaCashRegimeOutput
    output_dir: Path
    width_px: int = Field(default=960, ge=640)
    height_px: int = Field(default=544, ge=360)
    fps: int = Field(default=5, ge=2, le=12)
    duration_seconds: int = Field(default=7, ge=4, le=15)
    max_gif_bytes: int = Field(default=2_000_000, ge=500_000)

    model_config = {"arbitrary_types_allowed": True}


class RenderKoreaAssetsOutput(BaseModel):
    """Created Korean ETF/cash GIF and MP4 files."""

    gif_file: Path
    mp4_file: Path
    sizes_bytes: dict[str, int]


def _frame(bundle: WfaResultBundle) -> pd.DataFrame:
    """Return one aligned DataFrame from bundle time-series values.

    Complexity:
        Time and space O(n).

    Failure modes:
        Bundle alignment is validated before this conversion.
    """

    values = bundle.timeseries
    return pd.DataFrame(
        {
            "sp500": values.sp500_equity,
            "hybrid": values.hybrid_equity,
            "pairs": values.pairs_equity,
            "sp500_dd": values.sp500_drawdown,
            "hybrid_dd": values.hybrid_drawdown,
            "pairs_dd": values.pairs_drawdown,
            "regime": values.regime,
        },
        index=pd.to_datetime(values.dates),
    ).sort_index()


def _figure(width_px: int, height_px: int, dpi: int) -> Figure:
    """Create a consistent white 16:9 figure.

    Complexity:
        Time and space O(1).

    Failure modes:
        Matplotlib raises for unsupported backends or invalid dimensions.
    """

    return plt.figure(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
        facecolor="white",
    )


def _footer(fig: Figure, bundle: WfaResultBundle) -> None:
    """Attach auditable source and disclaimer text to a chart.

    Complexity:
        Time and space O(1).

    Failure modes:
        None for a valid Matplotlib figure and bundle.
    """

    meta = bundle.metadata
    cost = meta.transaction_costs
    text = (
        f"Data {meta.data_start}–{meta.data_end}  |  "
        f"cost {cost.get('commission_per_leg_bps', 0):.1f} + "
        f"{cost.get('slippage_per_leg_bps', 0):.1f} bps/leg  |  "
        "carry·국면 전환비용 미포함  |  "
        f"profile {meta.source}  |  generated {meta.generated_at.date()}  |  "
        f"{DISCLAIMER_KO}"
    )
    fig.text(0.5, 0.015, text, ha="center", va="bottom", fontsize=7, color=GRAY)


def _bear_bands(axis: Axes, frame: pd.DataFrame) -> None:
    """Shade contiguous pairs-regime intervals on one chart axis.

    Complexity:
        Time O(n), space O(n).

    Failure modes:
        A missing ``regime`` column raises ``KeyError``.
    """

    regime = frame["regime"].eq("pairs")
    group = regime.ne(regime.shift(fill_value=False)).cumsum()
    for _, block in frame.loc[regime].groupby(group.loc[regime]):
        axis.axvspan(block.index[0], block.index[-1], color=RED, alpha=0.10)


def _save_figure(
    fig: Figure,
    base_path: Path,
    bundle: WfaResultBundle,
    dpi: int,
) -> tuple[Path, Path]:
    """Save matching PNG and SVG versions of one figure.

    Complexity:
        Time and space scale with pixel count and vector path count.

    Failure modes:
        File permission and renderer failures propagate.
    """

    _footer(fig, bundle)
    png = base_path.with_suffix(".png")
    svg = base_path.with_suffix(".svg")
    fig.savefig(png, dpi=dpi, facecolor="white")
    fig.savefig(svg, format="svg", facecolor="white")
    plt.close(fig)
    return png, svg


def _equity_chart(
    bundle: WfaResultBundle,
    frame: pd.DataFrame,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    fig = _figure(width_px, height_px, dpi)
    axis = fig.add_subplot(111)
    axis.plot(frame.index, frame["sp500"], label="S&P 500", color=ORANGE, linewidth=1.5)
    axis.plot(frame.index, frame["hybrid"], label="하이브리드", color=TEAL, linewidth=2.3)
    axis.plot(
        frame.index,
        frame["pairs"],
        label="전 기간 페어 (저활동 진단)",
        color=BLUE,
        linewidth=1.2,
        alpha=0.8,
    )
    _bear_bands(axis, frame)
    axis.set_yscale("log")
    axis.set_title("워크포워드 누적 자산 곡선", loc="left", color=NAVY, weight="bold")
    axis.set_xlabel("날짜")
    axis.set_ylabel("가상 자산 (USD, 로그)")
    axis.grid(alpha=0.2)
    axis.legend(ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.94)
    return fig


def _drawdown_chart(
    bundle: WfaResultBundle,
    frame: pd.DataFrame,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    fig = _figure(width_px, height_px, dpi)
    axis = fig.add_subplot(111)
    axis.fill_between(
        frame.index,
        frame["sp500_dd"],
        0,
        color=ORANGE,
        alpha=0.18,
        label="S&P 500",
    )
    axis.plot(frame.index, frame["hybrid_dd"], color=TEAL, linewidth=2, label="하이브리드")
    axis.plot(
        frame.index,
        frame["pairs_dd"],
        color=BLUE,
        linewidth=1,
        label="전 기간 페어 (저활동 진단)",
    )
    axis.axhline(-0.10, color=RED, linestyle="--", linewidth=1, label="-10% 위험 신호")
    _bear_bands(axis, frame)
    axis.set_title("수익률만이 전체 성적표가 아닙니다", loc="left", color=NAVY, weight="bold")
    axis.set_xlabel("날짜")
    axis.set_ylabel("고점 대비 손실폭")
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.grid(alpha=0.2)
    axis.legend(ncol=4, loc="lower left")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.94)
    return fig


def _timeline_chart(
    bundle: WfaResultBundle,
    frame: pd.DataFrame,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    fig = _figure(width_px, height_px, dpi)
    grid = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.18)
    equity_axis = fig.add_subplot(grid[0])
    regime_axis = fig.add_subplot(grid[1], sharex=equity_axis)
    equity_axis.plot(frame.index, frame["hybrid"], color=TEAL, linewidth=2)
    equity_axis.set_yscale("log")
    equity_axis.set_ylabel("하이브리드 자산 (USD)")
    equity_axis.grid(alpha=0.2)
    regime_values = frame["regime"].map({"sp500": 0, "pairs": 1})
    regime_axis.fill_between(
        frame.index,
        0,
        regime_values,
        step="post",
        color=RED,
        alpha=0.45,
    )
    regime_axis.set_yticks([0, 1], labels=["지수", "페어"])
    regime_axis.set_xlabel("날짜")
    regime_axis.set_ylabel("시장 국면")
    fig.suptitle(
        "워크포워드: 과거로 선택하고 다음 미사용 구간에서 평가",
        x=0.05,
        ha="left",
        color=NAVY,
        weight="bold",
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.90)
    return fig


def _loss_window(frame: pd.DataFrame, radius: int = 180) -> pd.DataFrame:
    """Select a representative window around the deepest hybrid drawdown.

    Complexity:
        Time O(n), space O(radius).

    Failure modes:
        Empty frames raise ``ValueError``.
    """

    if frame.empty:
        raise ValueError("cannot select a loss window from an empty frame")
    center = frame.index.get_loc(frame["hybrid_dd"].idxmin())
    start = max(0, center - radius)
    end = min(len(frame), center + radius + 1)
    return frame.iloc[start:end]


def _recovery_trading_days(
    equity: pd.Series,
    trough_date: pd.Timestamp,
) -> int | None:
    """Return trading days from trough until equity recovers the prior peak."""

    prior_peak = float(equity.loc[:trough_date].max())
    after = equity.loc[trough_date:]
    recovered = after.loc[after.ge(prior_peak)]
    if recovered.empty:
        return None
    recovery_date = recovered.index[0]
    return int(equity.index.get_loc(recovery_date) - equity.index.get_loc(trough_date))


def _loss_chart(
    bundle: WfaResultBundle,
    frame: pd.DataFrame,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    """Connect MDD depth with post-loss recovery duration on one figure."""

    window = _loss_window(frame, radius=260)
    fig = _figure(width_px, height_px, dpi)
    grid = fig.add_gridspec(2, 1, height_ratios=(2.1, 1.0), hspace=0.12)
    equity_axis = fig.add_subplot(grid[0])
    drawdown_axis = fig.add_subplot(grid[1], sharex=equity_axis)

    normalized = window[["sp500", "hybrid"]].div(window[["sp500", "hybrid"]].iloc[0])
    equity_axis.plot(
        normalized.index,
        normalized["sp500"],
        color=ORANGE,
        linewidth=1.6,
        label="S&P 500",
    )
    equity_axis.plot(
        normalized.index,
        normalized["hybrid"],
        color=TEAL,
        linewidth=2.2,
        label="하이브리드",
    )
    _bear_bands(equity_axis, window)

    hybrid_trough = window["hybrid_dd"].idxmin()
    sp500_trough = window["sp500_dd"].idxmin()
    hybrid_mdd = float(window.loc[hybrid_trough, "hybrid_dd"])
    sp500_mdd = float(window.loc[sp500_trough, "sp500_dd"])
    hybrid_recovery = _recovery_trading_days(window["hybrid"], hybrid_trough)
    sp500_recovery = _recovery_trading_days(window["sp500"], sp500_trough)

    equity_axis.axvline(hybrid_trough, color=RED, linestyle="--", linewidth=1)
    equity_axis.annotate(
        (
            f"하이브리드 MDD {hybrid_mdd:.1%}\n"
            f"회복 {hybrid_recovery if hybrid_recovery is not None else '미회복'}일"
        ),
        xy=(hybrid_trough, normalized.loc[hybrid_trough, "hybrid"]),
        xytext=(14, 28),
        textcoords="offset points",
        color=RED,
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": RED},
    )
    equity_axis.annotate(
        (
            f"S&P MDD {sp500_mdd:.1%}\n"
            f"회복 {sp500_recovery if sp500_recovery is not None else '미회복'}일"
        ),
        xy=(sp500_trough, normalized.loc[sp500_trough, "sp500"]),
        xytext=(-90, -36),
        textcoords="offset points",
        color=ORANGE,
        fontsize=8,
        arrowprops={"arrowstyle": "->", "color": ORANGE},
    )
    equity_axis.set_title(
        "손실 깊이(MDD)와 회복에 걸린 기간을 함께 본다",
        loc="left",
        color=NAVY,
        weight="bold",
    )
    equity_axis.set_ylabel("정규화 자산")
    equity_axis.grid(alpha=0.2)
    equity_axis.legend(loc="upper left", fontsize=8)

    drawdown_axis.plot(
        window.index,
        window["sp500_dd"],
        color=ORANGE,
        linewidth=1.3,
        label=f"S&P 500 {sp500_mdd:.1%}",
    )
    drawdown_axis.plot(
        window.index,
        window["hybrid_dd"],
        color=TEAL,
        linewidth=1.8,
        label=f"하이브리드 {hybrid_mdd:.1%}",
    )
    drawdown_axis.axhline(0.0, color="#999999", linewidth=0.8)
    drawdown_axis.set_xlabel("날짜")
    drawdown_axis.set_ylabel("고점 대비 손실폭")
    drawdown_axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    drawdown_axis.grid(alpha=0.2)
    drawdown_axis.legend(loc="lower left", fontsize=8, ncol=2)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.92)
    return fig


def _before_after_chart(
    bundle: WfaResultBundle,
    frame: pd.DataFrame,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    fig = _figure(width_px, height_px, dpi)
    axis = fig.add_subplot(111)
    normalized = frame[["sp500", "hybrid"]].div(frame[["sp500", "hybrid"]].iloc[0])
    axis.plot(
        normalized.index,
        normalized["sp500"],
        color=ORANGE,
        linewidth=1.5,
        label="전: 지수 단일 기준",
    )
    axis.plot(
        normalized.index,
        normalized["hybrid"],
        color=TEAL,
        linewidth=2.4,
        label="후: 고정된 국면 전환 규칙",
    )
    _bear_bands(axis, frame)
    axis.set_yscale("log")
    axis.set_title("같은 조건의 재검증 비교 — 미래 수익 보장이 아닙니다", loc="left", color=NAVY, weight="bold")
    axis.set_xlabel("날짜")
    axis.set_ylabel("1달러의 성장 (로그)")
    axis.grid(alpha=0.2)
    axis.legend(loc="upper left")
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.10, top=0.94)
    return fig


def _korea_extension_chart(
    bundle: WfaResultBundle,
    width_px: int,
    height_px: int,
    dpi: int,
) -> Figure:
    fig = _figure(width_px, height_px, dpi)
    axis = fig.add_subplot(111)
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    boxes = [
        (0.05, 0.62, 0.19, 0.18, "KOSPI 200 지수\n(국면 온도계)"),
        (0.30, 0.62, 0.19, 0.18, "고정 규칙\nMDD + 이동평균 기울기"),
        (0.55, 0.72, 0.19, 0.16, "정상 국면\nKOSPI 200 ETF"),
        (0.55, 0.48, 0.19, 0.16, "위험 국면\n현금 대기"),
        (0.80, 0.60, 0.16, 0.18, "선택 연구\n페어 모의투자"),
    ]
    for x, y, width, height, label in boxes:
        patch = matplotlib.patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012",
            facecolor=LIGHT,
            edgecolor=NAVY,
            linewidth=1.2,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", color=NAVY)
    arrows = [
        ((0.24, 0.71), (0.30, 0.71)),
        ((0.49, 0.71), (0.55, 0.80)),
        ((0.49, 0.69), (0.55, 0.56)),
        ((0.74, 0.72), (0.80, 0.69)),
    ]
    for start, end in arrows:
        axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 2})
    axis.text(
        0.05,
        0.92,
        "한국 시장 확장: 지수는 온도계, ETF는 실제 투자 수단",
        fontsize=15,
        color=NAVY,
        weight="bold",
    )
    axis.text(
        0.05,
        0.30,
        "공개 설명의 경계",
        color=RED,
        weight="bold",
    )
    axis.text(
        0.05,
        0.24,
        "미국 WFA 결과는 미국 사례의 근거이며, 한국 성과는 별도 백테스트가 필요합니다.",
        color=GRAY,
    )
    axis.text(
        0.05,
        0.17,
        "기본 경로는 ETF/현금입니다. 한국 공매도는 대주·API 제약 확인 전까지 모의투자에만 둡니다.",
        color=GRAY,
    )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.06, top=0.99)
    return fig


def render_seminar_assets(
    inp: RenderSeminarAssetsInput,
) -> RenderSeminarAssetsOutput:
    """Render the 16:9 PNG/SVG seminar package from one bundle.

    Args:
        inp: Bundle, output directory and image dimensions.

    Returns:
        Paths to all charts and Korean presenter notes.

    Complexity:
        Time O(charts × observations), space O(observations + pixels).

    Failure modes:
        Invalid bundle data or filesystem errors propagate.
    """

    inp.output_dir.mkdir(parents=True, exist_ok=True)
    frame = _frame(inp.bundle)
    chart_builders: list[tuple[str, Any]] = [
        ("01_equity_curve", _equity_chart),
        ("02_drawdown", _drawdown_chart),
        ("03_wfa_timeline", _timeline_chart),
        ("04_loss_period", _loss_chart),
        ("05_before_after", _before_after_chart),
    ]
    png_files: list[Path] = []
    svg_files: list[Path] = []
    for filename, builder in chart_builders:
        fig = builder(
            inp.bundle,
            frame,
            inp.width_px,
            inp.height_px,
            inp.dpi,
        )
        png, svg = _save_figure(
            fig,
            inp.output_dir / filename,
            inp.bundle,
            inp.dpi,
        )
        png_files.append(png)
        svg_files.append(svg)

    korea_fig = _korea_extension_chart(
        inp.bundle,
        inp.width_px,
        inp.height_px,
        inp.dpi,
    )
    png, svg = _save_figure(
        korea_fig,
        inp.output_dir / "06_korea_extension",
        inp.bundle,
        inp.dpi,
    )
    png_files.append(png)
    svg_files.append(svg)

    notes_file = inp.output_dir / "SPEAKER_NOTES.md"
    notes_file.write_text(
        _speaker_notes(inp.bundle),
        encoding="utf-8",
    )
    return RenderSeminarAssetsOutput(
        png_files=png_files,
        svg_files=svg_files,
        notes_file=notes_file,
    )


def _speaker_notes(bundle: WfaResultBundle) -> str:
    """Build concise Korean presenter notes for the generated charts.

    Complexity:
        Time and space O(1).

    Failure modes:
        Missing summary values fall back to zero for safe rendering.
    """

    summary = bundle.summary
    return f"""# 무료 세미나 발표 메모

기준 결과: `{bundle.metadata.source}`  
데이터: {bundle.metadata.data_start}–{bundle.metadata.data_end}  
면책: {DISCLAIMER_KO}

## 01 누적 수익 곡선

- 무엇을 볼지: 지수와 하이브리드의 장기 경로, 저활동 진단용 전 기간 페어, 붉은 하락 국면
- 핵심 메시지: 결과 숫자보다 어떤 규칙으로 시장 노출이 바뀌었는지 먼저 봅니다.
- 오해 방지: 세 곡선은 공통 시작일에 같은 금액으로 정규화합니다. 전 기간 페어는 핵심 성과선이 아니라 거래 활동 저하를 확인하는 진단선입니다. 현재 S&P 500 구성 종목을 과거에 적용한 생존편향이 있으며 한국 실적이 아닙니다.

## 02 Drawdown

- 무엇을 볼지: 수익률과 함께 고점 대비 손실폭을 확인합니다.
- 핵심 메시지: 전략의 품질은 최고 수익률 하나가 아니라 감당 가능한 MDD와 회복 과정까지 포함합니다.
- 기준 hybrid MDD: {summary.get("max_drawdown", 0.0):.1%}

## 03 WFA Timeline

- 무엇을 볼지: 붉은 구간에서 페어 전략을 사용한 시점
- 핵심 메시지: 당시까지의 과거로 규칙을 고르고, 바로 다음의 미사용 과거를 모의 미래처럼 평가합니다.
- 오해 방지: WFA는 실제 미래를 예측하거나 보장하지 않습니다.

## 04 대표 손실 구간

- 무엇을 볼지: 손실이 커진 날짜, 같은 시기 지수와 hybrid의 차이
- 핵심 메시지: 손실은 숨길 실패가 아니라 원인과 다음 실험을 정하는 입력입니다.
- 다음 행동: 유지, 허용 파라미터 변경 후 재검증, 중단 중에서 사람이 선택합니다.

## 05 재검증 비교

- 무엇을 볼지: 지수 단일 baseline과 사전에 고정한 regime rule
- 핵심 메시지: 변경안은 과거 손실에 맞춰 즉시 배포하지 않고 같은 기준으로 다시 검증합니다.
- 오해 방지: “after”가 미래 수익 향상을 의미하지 않습니다.

## 06 한국 적용 구조

- 무엇을 볼지: KOSPI 200 지수는 온도계, ETF는 실제 투자 상품이라는 구분
- 핵심 메시지: 한국 입문 기본값은 ETF/현금이고 페어는 모의투자 연구 사례입니다.
- 오해 방지: 미국 결과를 한국 결과로 표현하지 않으며 대주·담보·API 제약을 확인하기 전 자동 공매도를 주장하지 않습니다.
"""


def _canvas_rgb(fig: Figure) -> np.ndarray:
    """Convert a rendered Matplotlib figure to an RGB array.

    Complexity:
        Time and space O(pixel count).

    Failure modes:
        Backend rendering failures propagate.
    """

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return np.ascontiguousarray(rgba[:, :, :3])


def _write_video_pair(
    frames: list[np.ndarray],
    gif_path: Path,
    mp4_path: Path,
    fps: int,
) -> None:
    """Write matching GIF and H.264 MP4 files with bundled codecs.

    Complexity:
        Time O(frames × pixels), space O(frames × pixels).

    Failure modes:
        Codec availability or filesystem failures propagate from ImageIO.
    """

    stacked = np.stack(frames)
    gif_frames = [
        Image.fromarray(frame).quantize(
            colors=32,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )
    iio.imwrite(
        mp4_path,
        stacked,
        extension=".mp4",
        fps=fps,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
    )


def _replay_frames(
    bundle: WfaResultBundle,
    width_px: int,
    height_px: int,
    fps: int,
    duration_seconds: int,
) -> list[np.ndarray]:
    """Render progressive WFA equity and drawdown frames.

    Complexity:
        Time O(frames × observations), space O(frames × pixels).

    Failure modes:
        Empty bundles fail during frame selection.
    """

    frame = _frame(bundle)
    frame_count = fps * duration_seconds
    stops = np.linspace(max(30, len(frame) // frame_count), len(frame), frame_count, dtype=int)
    images: list[np.ndarray] = []
    for stop in stops:
        visible = frame.iloc[:stop]
        fig = _figure(width_px, height_px, 100)
        grid = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.12)
        equity_axis = fig.add_subplot(grid[0])
        drawdown_axis = fig.add_subplot(grid[1], sharex=equity_axis)
        equity_axis.plot(visible.index, visible["sp500"], color=ORANGE, linewidth=1, label="S&P 500")
        equity_axis.plot(visible.index, visible["hybrid"], color=TEAL, linewidth=2, label="하이브리드")
        _bear_bands(equity_axis, visible)
        equity_axis.set_yscale("log")
        equity_axis.set_ylabel("가상 자산 (USD)")
        equity_axis.legend(loc="upper left", ncol=2, fontsize=8)
        equity_axis.grid(alpha=0.18)
        sp500_mdd = float(visible["sp500_dd"].min())
        hybrid_mdd = float(visible["hybrid_dd"].min())
        drawdown_axis.plot(
            visible.index,
            visible["sp500_dd"],
            color=ORANGE,
            linewidth=1.0,
            alpha=0.85,
            label=f"S&P 500 {sp500_mdd:.1%}",
        )
        drawdown_axis.plot(
            visible.index,
            visible["hybrid_dd"],
            color=TEAL,
            linewidth=1.4,
            label=f"하이브리드 {hybrid_mdd:.1%}",
        )
        drawdown_axis.axhline(-0.10, color=RED, linestyle="--", linewidth=0.8)
        _bear_bands(drawdown_axis, visible)
        drawdown_axis.set_ylabel("MDD")
        drawdown_axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        drawdown_axis.grid(alpha=0.18)
        drawdown_axis.legend(
            loc="lower left",
            ncol=2,
            fontsize=7,
        )
        fig.suptitle(
            f"검증 결과 재생  |  {visible.index[-1].date()}  |  {stop / len(frame):.0%}",
            x=0.07,
            ha="left",
            color=NAVY,
            weight="bold",
            fontsize=11,
        )
        fig.text(0.99, 0.01, "과거 시뮬레이션 · 실시간 계산 아님", ha="right", fontsize=7, color=GRAY)
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.87, hspace=0.12)
        images.append(_canvas_rgb(fig))
        plt.close(fig)
    return images


def _loss_review_frames(
    bundle: WfaResultBundle,
    width_px: int,
    height_px: int,
    fps: int,
    duration_seconds: int,
) -> list[np.ndarray]:
    """Render a closed-menu human loss-review story.

    Complexity:
        Time O(frames × loss-window observations), space O(frames × pixels).

    Failure modes:
        Empty or malformed bundle data fails during plotting.
    """

    frame = _frame(bundle)
    window = _loss_window(frame)
    normalized = window[["sp500", "hybrid"]].div(window[["sp500", "hybrid"]].iloc[0])
    worst_date = window["hybrid_dd"].idxmin()
    stages = [
        ("1  손실 구간에서 멈추기", "관찰: MDD가 확대되었습니다. 원인은 아직 단정하지 않습니다."),
        ("2  근거 검토", f"하이브리드 MDD {window['hybrid_dd'].min():.1%} | 날짜 {worst_date.date()}"),
        ("3  사용자가 결정", "유지  |  허용 변수 재검증  |  중단"),
        ("4  같은 기준으로 재검증", "배포 전 고정 규칙과 기준 지수를 다시 비교합니다."),
    ]
    total_frames = fps * duration_seconds
    images: list[np.ndarray] = []
    for frame_index in range(total_frames):
        stage_index = min(len(stages) - 1, frame_index * len(stages) // total_frames)
        title, message = stages[stage_index]
        fig = _figure(width_px, height_px, 100)
        grid = fig.add_gridspec(1, 2, width_ratios=(3, 2), wspace=0.12)
        axis = fig.add_subplot(grid[0])
        panel = fig.add_subplot(grid[1])
        axis.plot(normalized.index, normalized["sp500"], color=ORANGE, label="기준 지수")
        axis.plot(normalized.index, normalized["hybrid"], color=TEAL, linewidth=2, label="하이브리드")
        axis.axvline(worst_date, color=RED, linestyle="--")
        _bear_bands(axis, window)
        axis.set_title("대표 손실 구간", color=NAVY, loc="left", fontsize=10, weight="bold")
        axis.set_ylabel("정규화 자산")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.18)
        panel.axis("off")
        panel.add_patch(
            matplotlib.patches.FancyBboxPatch(
                (0.03, 0.12),
                0.94,
                0.76,
                boxstyle="round,pad=0.02",
                facecolor=LIGHT,
                edgecolor=NAVY,
            )
        )
        panel.text(0.08, 0.78, title, color=NAVY, weight="bold", fontsize=11, va="top", wrap=True)
        panel.text(0.08, 0.62, message, color=GRAY, fontsize=9, va="top", wrap=True)
        panel.text(
            0.08,
            0.23,
            "AI는 제안하고, 사람은 확인하고 결정합니다.",
            color=RED,
            fontsize=8,
            weight="bold",
            wrap=True,
        )
        fig.suptitle("손실 검토 — 제한된 행동 선택지", x=0.05, ha="left", color=NAVY, weight="bold")
        fig.text(0.99, 0.01, DISCLAIMER_KO, ha="right", fontsize=7, color=GRAY)
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.09, top=0.86, wspace=0.12)
        images.append(_canvas_rgb(fig))
        plt.close(fig)
    return images


def _paper_approval_frames(
    width_px: int,
    height_px: int,
    fps: int,
    duration_seconds: int,
) -> list[np.ndarray]:
    """Render a 2-Tier governance and autonomous trading flow animation.

    Complexity:
        Time O(frames × pixels), space O(frames × pixels).

    Failure modes:
        Matplotlib rendering failures propagate.
    """

    stages = [
        ("① 주간 국면 진단", "Claude Code와 시장 평가\n4대 전략 가중치 분석 (주말)"),
        ("② 정책 서명 승인", "weekly_policy.json 확정\n사람 CIO 최종 서명 (Tier 1)"),
        ("③ 무인 자동 발주", "승인 정책 기반 실시간 감시\n조건 충족 시 KIS 발주 (Tier 2)"),
        ("④ 0.1초 하드 안전망", "개별 -3% 즉각 무인 손절\n계좌 -2% 서킷브레이커 청산"),
        ("⑤ AI 일일 매매 일지", "장 마감 후 체결·손익 분석\n정기 알고리즘 튜닝 환류"),
    ]
    total_frames = fps * duration_seconds
    images: list[np.ndarray] = []
    for frame_index in range(total_frames):
        active = min(len(stages) - 1, frame_index * len(stages) // total_frames)
        fig = _figure(width_px, height_px, 100)
        axis = fig.add_subplot(111)
        axis.axis("off")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.text(
            0.03,
            0.94,
            "2-Tier 주간 정책 합의 & 무인 자동 모의매매 흐름",
            color=NAVY,
            weight="bold",
            fontsize=13,
        )
        axis.text(
            0.03,
            0.87,
            "Tier 1: 주간 거버넌스(주말 30분)  ➔  Tier 2: 일간 무인 자동 실행 & 0.1초 하드 손절(평일 장중)",
            color=TEAL,
            weight="bold",
            fontsize=9,
        )
        for index, (title, message) in enumerate(stages):
            x = 0.03 + index * 0.194
            face = "#DDEFEA" if index <= active else LIGHT
            edge = TEAL if index <= active else GRAY
            patch = matplotlib.patches.FancyBboxPatch(
                (x, 0.43),
                0.165,
                0.32,
                boxstyle="round,pad=0.014",
                facecolor=face,
                edgecolor=edge,
                linewidth=2 if index == active else 1,
            )
            axis.add_patch(patch)
            axis.text(
                x + 0.0825,
                0.66,
                title,
                ha="center",
                va="center",
                color=NAVY,
                weight="bold",
                fontsize=8.5,
            )
            axis.text(
                x + 0.0825,
                0.52,
                message,
                ha="center",
                va="center",
                color=GRAY,
                fontsize=7,
                linespacing=1.3,
            )
            if index < len(stages) - 1:
                axis.annotate(
                    "",
                    xy=(x + 0.19, 0.59),
                    xytext=(x + 0.168, 0.59),
                    arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.5},
                )
        axis.text(
            0.03,
            0.26,
            "Tier 1: 주말 정책 승인(weekly_policy.json)  |  Tier 2: 평일 무인 자동 실행  |  0.1초 하드 손절  |  AI 일일 매매 일지",
            color=NAVY,
            weight="bold",
            fontsize=9.5,
        )
        axis.text(
            0.03,
            0.15,
            "건별 승인 피로와 뇌동매매를 차단하고, 주 1회 승인된 알고리즘에 따라 기계가 100% 무인으로 규율 있게 매매합니다.",
            color=GRAY,
            fontsize=8.5,
        )
        fig.subplots_adjust(left=0.03, right=0.99, bottom=0.04, top=0.98)
        images.append(_canvas_rgb(fig))
        plt.close(fig)
    return images


def render_detail_assets(
    inp: RenderDetailAssetsInput,
) -> RenderDetailAssetsOutput:
    """Render replay and loss-review GIF/MP4 pairs.

    Args:
        inp: Bundle, output directory and animation constraints.

    Returns:
        Created paths and file sizes.

    Complexity:
        Time O(frames × observations + frames × pixels).

    Failure modes:
        GIFs exceeding the requested byte limit raise ``ValueError``.
        Image or codec failures propagate.
    """

    inp.output_dir.mkdir(parents=True, exist_ok=True)
    replay_gif = inp.output_dir / "01_wfa_replay.gif"
    replay_mp4 = inp.output_dir / "01_wfa_replay.mp4"
    loss_gif = inp.output_dir / "02_loss_review.gif"
    loss_mp4 = inp.output_dir / "02_loss_review.mp4"
    paper_gif = inp.output_dir / "03_paper_approval.gif"
    paper_mp4 = inp.output_dir / "03_paper_approval.mp4"

    replay_frames = _replay_frames(
        inp.bundle,
        inp.width_px,
        inp.height_px,
        inp.fps,
        inp.duration_seconds,
    )
    _write_video_pair(replay_frames, replay_gif, replay_mp4, inp.fps)
    loss_frames = _loss_review_frames(
        inp.bundle,
        inp.width_px,
        inp.height_px,
        inp.fps,
        inp.duration_seconds,
    )
    _write_video_pair(loss_frames, loss_gif, loss_mp4, inp.fps)
    paper_frames = _paper_approval_frames(
        inp.width_px,
        inp.height_px,
        inp.fps,
        inp.duration_seconds,
    )
    _write_video_pair(paper_frames, paper_gif, paper_mp4, inp.fps)

    sizes = {
        replay_gif.name: replay_gif.stat().st_size,
        replay_mp4.name: replay_mp4.stat().st_size,
        loss_gif.name: loss_gif.stat().st_size,
        loss_mp4.name: loss_mp4.stat().st_size,
        paper_gif.name: paper_gif.stat().st_size,
        paper_mp4.name: paper_mp4.stat().st_size,
    }
    oversized = {
        name: size
        for name, size in sizes.items()
        if name.endswith(".gif") and size > inp.max_gif_bytes
    }
    if oversized:
        formatted = ", ".join(f"{name}={size:,}" for name, size in oversized.items())
        raise ValueError(f"detail GIF exceeds {inp.max_gif_bytes:,} bytes: {formatted}")
    return RenderDetailAssetsOutput(
        replay_gif=replay_gif,
        replay_mp4=replay_mp4,
        loss_review_gif=loss_gif,
        loss_review_mp4=loss_mp4,
        paper_approval_gif=paper_gif,
        paper_approval_mp4=paper_mp4,
        sizes_bytes=sizes,
    )


def render_korea_cash_assets(
    inp: RenderKoreaAssetsInput,
) -> RenderKoreaAssetsOutput:
    """Render the Korean ETF buy-and-hold versus cash-regime animation.

    Args:
        inp: Walk-forward result and animation constraints.

    Returns:
        Matching GIF/MP4 files and their sizes.

    Complexity:
        Time O(frames × observations + frames × pixels).

    Failure modes:
        Empty results, codec failures or oversized GIFs raise errors.
    """

    inp.output_dir.mkdir(parents=True, exist_ok=True)
    equity = inp.result.equity.sort_index()
    position = inp.result.position.reindex(equity.index).fillna(0.0)
    total_frames = inp.fps * inp.duration_seconds
    stops = np.linspace(
        max(20, len(equity) // total_frames),
        len(equity),
        total_frames,
        dtype=int,
    )
    frames: list[np.ndarray] = []
    for stop in stops:
        visible = equity.iloc[:stop]
        visible_position = position.iloc[:stop]
        fig = _figure(inp.width_px, inp.height_px, 100)
        grid = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.12)
        equity_axis = fig.add_subplot(grid[0])
        position_axis = fig.add_subplot(grid[1], sharex=equity_axis)
        equity_axis.plot(
            visible.index,
            visible["etf_buy_hold"],
            color=ORANGE,
            label="ETF 계속 보유",
        )
        equity_axis.plot(
            visible.index,
            visible["etf_cash_wfa"],
            color=TEAL,
            linewidth=2,
            label="ETF / 현금 WFA",
        )
        equity_axis.set_ylabel("가상 자산 (KRW)")
        equity_axis.legend(loc="upper left", fontsize=8)
        equity_axis.grid(alpha=0.18)
        position_axis.fill_between(
            visible_position.index,
            0,
            visible_position,
            step="post",
            color=TEAL,
            alpha=0.4,
        )
        position_axis.set_ylim(-0.05, 1.05)
        position_axis.set_yticks([0, 1], labels=["현금", "ETF"])
        position_axis.set_xlabel("날짜")
        fig.suptitle(
            f"KOSPI 200 ETF / 현금 — 다음 날 체결  |  {stop / len(equity):.0%}",
            x=0.05,
            ha="left",
            color=NAVY,
            weight="bold",
            fontsize=11,
        )
        fig.text(
            0.99,
            0.01,
            "지수 = 국면 온도계  |  ETF 수정주가 = 투자 수익률  |  과거 시뮬레이션",
            ha="right",
            fontsize=7,
            color=GRAY,
        )
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.87, hspace=0.12)
        frames.append(_canvas_rgb(fig))
        plt.close(fig)

    gif_file = inp.output_dir / "04_korea_cash_regime.gif"
    mp4_file = inp.output_dir / "04_korea_cash_regime.mp4"
    _write_video_pair(frames, gif_file, mp4_file, inp.fps)
    sizes = {
        gif_file.name: gif_file.stat().st_size,
        mp4_file.name: mp4_file.stat().st_size,
    }
    if sizes[gif_file.name] > inp.max_gif_bytes:
        raise ValueError(
            f"detail GIF exceeds {inp.max_gif_bytes:,} bytes: "
            f"{gif_file.name}={sizes[gif_file.name]:,}"
        )
    return RenderKoreaAssetsOutput(
        gif_file=gif_file,
        mp4_file=mp4_file,
        sizes_bytes=sizes,
    )


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    """Return hash, byte size and path metadata for generated assets.

    Complexity:
        Time O(total file bytes), space O(number of files).

    Failure modes:
        Missing or unreadable paths raise filesystem errors.
    """

    records: list[dict[str, Any]] = []
    for path in paths:
        raw = path.read_bytes()
        records.append(
            {
                "path": str(path),
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return records


def write_asset_manifest(paths: list[Path], output_path: Path) -> Path:
    """Write a machine-readable manifest for generated media.

    Complexity:
        Time O(total file bytes), space O(number of files).

    Failure modes:
        Filesystem errors propagate.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(file_manifest(paths), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
