
from __future__ import annotations

import json
import zipfile
import io
import hashlib
import os
import shutil
import math
import re
import subprocess
import sys
import time
import threading
import queue
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
com = components

import html as _html


def _escape_html(x: object) -> str:
    """HTML-escape text for safe embedding in unsafe_allow_html blocks."""
    try:
        return _html.escape("" if x is None else str(x), quote=True)
    except Exception:
        return "" if x is None else str(x)


# Optional: used for Build-step “reality check” stats (TA filters)
try:
    from engine.features import add_features as _add_features
except Exception:  # pragma: no cover
    _add_features = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:  # pragma: no cover
    px = None
    go = None
    make_subplots = None

# Plotly rendering config (Streamlit wants config dict for Plotly options)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}

# Plotly chart helper: Streamlit has been changing its API (width vs use_container_width).
# We pick the supported signature at runtime to avoid deprecation spam.
import inspect as _inspect

try:
    _PLOTLY_CHART_SIG = _inspect.signature(st.plotly_chart)
except Exception:  # pragma: no cover
    _PLOTLY_CHART_SIG = None

_PLOTLY_HAS_WIDTH = bool(_PLOTLY_CHART_SIG and ("width" in _PLOTLY_CHART_SIG.parameters))
_PLOTLY_HAS_UCW = bool(_PLOTLY_CHART_SIG and ("use_container_width" in _PLOTLY_CHART_SIG.parameters))

# Plotly rendering: prefer Streamlit's built-in theme for consistent typography/colors.
USE_STREAMLIT_PLOTLY_THEME = True

def _sanitize_plotly_figure(fig):
    """Prevent Plotly/Streamlit front-end glitches caused by missing title/legend title text.

    In some Streamlit + Plotly.js combinations, if a title object exists but its `.text`
    is missing, Plotly can literally render the string "undefined" in the figure.
    This sanitizer ensures those text fields are always defined (even if visually hidden).
    """
    if fig is None:
        return fig

    SAFE_BLANK = "\u200b"  # zero-width space (non-empty string)
    TRANSPARENT = "rgba(0,0,0,0)"

    # Helper: safely treat plotly objects as dicts
    try:
        fig_dict = fig.to_dict() if hasattr(fig, "to_dict") else fig
    except Exception:
        return fig

    if not isinstance(fig_dict, dict):
        return fig

    layout = fig_dict.setdefault("layout", {})

    # --- Title: ensure title.text exists if title object exists (or if Streamlit template would add it) ---
    title = layout.get("title", None)
    if title is None or title is False:
        # Force a "hidden" title object so template/theme won't create a title shell with missing text.
        layout["title"] = {"text": SAFE_BLANK, "font": {"color": TRANSPARENT, "size": 1}}
    elif isinstance(title, dict):
        ttext = title.get("text", None)
        if ttext is None or str(ttext).strip() == "" or str(ttext).strip().lower() == "undefined":
            title["text"] = SAFE_BLANK
            title.setdefault("font", {})
            title["font"].setdefault("color", TRANSPARENT)
            title["font"].setdefault("size", 1)

    # --- Legend title: same idea (prevents bold "undefined" above legend items) ---
    legend = layout.get("legend", None)
    if isinstance(legend, dict):
        ltitle = legend.get("title", None)
        if ltitle is None:
            legend["title"] = {"text": SAFE_BLANK, "font": {"color": TRANSPARENT, "size": 1}}
        elif isinstance(ltitle, dict):
            lt = ltitle.get("text", None)
            if lt is None or str(lt).strip() == "" or str(lt).strip().lower() == "undefined":
                ltitle["text"] = SAFE_BLANK
                ltitle.setdefault("font", {})
                ltitle["font"].setdefault("color", TRANSPARENT)
                ltitle["font"].setdefault("size", 1)

    # --- Trace-level legend group titles (rare, but can also produce "undefined") ---
    data = fig_dict.get("data", [])
    if isinstance(data, list):
        for tr in data:
            if not isinstance(tr, dict):
                continue
            lgt = tr.get("legendgrouptitle")
            if isinstance(lgt, dict):
                gt = lgt.get("text", None)
                if gt is None or str(gt).strip() == "" or str(gt).strip().lower() == "undefined":
                    lgt["text"] = SAFE_BLANK
                    lgt.setdefault("font", {})
                    lgt["font"].setdefault("color", TRANSPARENT)
                    lgt["font"].setdefault("size", 1)

    # Push back into the Figure if we can
    try:
        if hasattr(fig, "update"):
            fig.update(fig_dict)
            return fig
    except Exception:
        pass

    return fig_dict



def _plotly(fig, *, key: Optional[str] = None) -> None:
    fig = _sanitize_plotly_figure(fig)

    # If a figure has no legendable traces, force-hide the legend. Otherwise Plotly.js may still
    # render a legend container with a broken/undefined title in some Streamlit builds.
    try:
        has_items = False
        for tr in getattr(fig, "data", []) or []:
            try:
                if getattr(tr, "showlegend", True) is False:
                    continue
                nm = getattr(tr, "name", None)
                if nm is None:
                    continue
                if str(nm).strip() == "" or str(nm).strip().lower() == "undefined":
                    continue
                has_items = True
                break
            except Exception:
                continue
        if not has_items:
            fig.update_layout(showlegend=False)
        else:
            # Ensure legend title is never missing (missing -> Plotly.js may print 'undefined')
            fig.update_layout(legend_title_text="\u00A0", legend_title_font=dict(color="rgba(0,0,0,0)", size=1))
    except Exception:
        pass

    kwargs: Dict[str, Any] = {"config": PLOTLY_CONFIG}
    if key is not None:
        kwargs["key"] = key
    if _PLOTLY_HAS_WIDTH:
        kwargs["width"] = "stretch"
    elif _PLOTLY_HAS_UCW:
        kwargs["use_container_width"] = True
    kwargs.setdefault("theme", "streamlit" if USE_STREAMLIT_PLOTLY_THEME else None)
    st.plotly_chart(fig, **kwargs)


# Optional (nice formatting + metric labels)
try:
    from lab.metrics import METRICS
except Exception:  # pragma: no cover
    METRICS = {}


# =============================================================================
# Visual system (Sprint 6)
# =============================================================================

PASS_COLOR = "#00C853"   # vibrant green
WARN_COLOR = "#FFD600"   # bright amber
FAIL_COLOR = "#FF1744"   # vivid red
NEUTRAL_COLOR = "#9E9E9E"
ACCENT_BLUE = "#2979FF"  # electric blue
ACCENT_ORANGE = "#FF6D00"  # vivid orange (used for tolerance markers)

VERDICT_COLORS = {
    "PASS": PASS_COLOR,
    "WARN": WARN_COLOR,
    "FAIL": FAIL_COLOR,
    "UNMEASURED": NEUTRAL_COLOR,
}

def _style_fig(fig, title: str | None = None):
    """Lightly standardize Plotly figures.

    If USE_STREAMLIT_PLOTLY_THEME is True, we avoid forcing a Plotly template/font so Streamlit's
    theme controls the look. We still manage margins/legend placement to prevent collisions.
    """
    if fig is None:
        return fig

    def _bold_title(t: str | None) -> str | None:
        if t is None:
            return None
        s = str(t)
        # Avoid double-wrapping bold tags
        if "<b>" in s.lower():
            return s
        return f"<b>{s}</b>"

    _title = _bold_title(title)

    if USE_STREAMLIT_PLOTLY_THEME:
        fig.update_layout(
            title=dict(text=_title, x=0.0, xanchor="left", y=0.98, yanchor="top") if _title else None,
            margin=dict(l=60, r=24, t=90 if _title else 60, b=56),
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1.0),
        )
        fig.update_xaxes(automargin=True, title_standoff=10)
        fig.update_yaxes(automargin=True, title_standoff=10)
        return fig

    # Fallback: fully self-styled Plotly (used when theme=None)
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, sans-serif", size=13, color="#1f2937"),
        title=dict(text=_title, font=dict(size=18, color="#111827"), x=0.0, xanchor="left", y=0.98),
        margin=dict(l=60, r=24, t=85, b=56),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#374151"),
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        automargin=True,
        title_standoff=10,
        tickfont=dict(size=12, color="#374151"),
        titlefont=dict(size=13, color="#374151"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.06)",
        zeroline=False,
        automargin=True,
        title_standoff=10,
        tickfont=dict(size=12, color="#374151"),
        titlefont=dict(size=13, color="#374151"),
    )
    return fig



# =============================================================================
# Stability visuals helpers
# =============================================================================

def _pct_rank(arr: np.ndarray, x: float) -> float:
    """Percentile rank of x within arr (0..100). Uses <= for ties."""
    try:
        if arr is None:
            return float("nan")
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return float("nan")
        if x is None or (not math.isfinite(float(x))):
            return float("nan")
        return float(np.mean(a <= float(x)) * 100.0)
    except Exception:
        return float("nan")


def _stability_zone_label(pct: float) -> str:
    """Human label for percentile band."""
    try:
        p = float(pct)
    except Exception:
        return ""
    if not math.isfinite(p):
        return ""
    if p >= 90.0:
        return "Elite"
    if p >= 80.0:
        return "Strong"
    if p >= 50.0:
        return "Typical"
    return "Below typical"


def _stability_percentile_bar_fig(
    you_pct: float,
    *,
    selected_pct: float | None = None,
    show_selected: bool = False,
    cutoff_pct: float | None = None,
) -> "go.Figure | None":
    """Small 'credit-score' style bar (0-100) with Typical/Strong/Elite bands and markers."""
    if go is None:
        return None

    try:
        y0, y1 = 0.0, 1.0
        fig = go.Figure()

        # Bands (0-80 typical, 80-90 strong, 90-100 elite)
        fig.add_shape(type="rect", x0=0, x1=80, y0=y0, y1=y1, line=dict(width=0), fillcolor="rgba(17,24,39,0.06)")
        fig.add_shape(type="rect", x0=80, x1=90, y0=y0, y1=y1, line=dict(width=0), fillcolor="rgba(41,121,255,0.10)")
        fig.add_shape(type="rect", x0=90, x1=100, y0=y0, y1=y1, line=dict(width=0), fillcolor="rgba(41,121,255,0.18)")

        # Segment labels
        fig.add_annotation(x=40, y=0.5, text="Typical", showarrow=False, font=dict(size=12, color="#111827"))
        fig.add_annotation(x=85, y=0.5, text="Strong", showarrow=False, font=dict(size=12, color=ACCENT_BLUE))
        fig.add_annotation(x=95, y=0.5, text="Elite", showarrow=False, font=dict(size=12, color=ACCENT_BLUE))

        def _add_marker(pct: float, label: str, color: str, *, width: int = 3, dash: str = "solid", symbol: str = "circle") -> None:
            if pct is None or (not math.isfinite(float(pct))):
                return
            x = float(min(max(float(pct), 0.0), 100.0))
            # Tick line
            fig.add_shape(type="line", x0=x, x1=x, y0=y0, y1=y1, line=dict(color=color, width=width, dash=dash))
            # Marker point (helps hover + visibility)
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[0.5],
                    mode="markers",
                    marker=dict(size=12, symbol=symbol, color=color, line=dict(width=2, color="rgba(255,255,255,0.95)")),
                    hovertemplate=f"{label}: P{float(pct):.0f}<extra></extra>",
                    showlegend=False,
                )
            )
            # Label bubble
            fig.add_annotation(
                x=x,
                y=1.12,
                yref="paper",
                text=f"{label} (P{float(pct):.0f})",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(size=12, color=color),
            )

        _add_marker(you_pct, "You", "#111827", width=4, symbol="star")

        if cutoff_pct is not None and math.isfinite(float(cutoff_pct)):
            _add_marker(float(cutoff_pct), "Cutoff", WARN_COLOR, width=2, dash="dash", symbol="x")

        if show_selected and selected_pct is not None and math.isfinite(float(selected_pct)):
            _add_marker(float(selected_pct), "Selected", ACCENT_ORANGE, width=3, dash="dot", symbol="diamond")

        fig.update_xaxes(range=[0, 100], tickvals=[0, 50, 80, 90, 100], ticktext=["0", "50", "80", "90", "100"], title=None, showgrid=False, zeroline=False)
        fig.update_yaxes(range=[0, 1], visible=False, showgrid=False, zeroline=False)
        fig.update_layout(height=140, margin=dict(l=20, r=20, t=50, b=10), showlegend=False)
        return fig
    except Exception:
        return None



def _rs_zone_width(tol: float) -> float:
    """Width of the 'near miss' band around the disappoint cutoff.

    Returns are in fractional units (0.01 = 1%). Default minimum keeps the band visible
    even when tol is close to 0.
    """
    try:
        t = float(tol)
    except Exception:
        t = 0.0
    return max(abs(t) * 0.25, 0.002)  # 0.2% minimum


def _rs_add_bands(fig, tol: float, near: float, y_min=None, y_max=None):
    """Add PASS/WARN/FAIL horizontal bands to a Plotly figure.

    IMPORTANT: Do **not** use absurd y0/y1 like +/-1e9. Shapes affect Plotly autorange,
    which can explode the axis into "1000000000%" territory. We instead bound the bands
    to the data range (+padding) and return the range so callers can lock y-axis.

    Returns:
        (y0, y1) range used for the bands.
    """
    near = abs(float(near))
    tol = float(tol)

    vals = []
    for v in (y_min, y_max, tol, tol + near):
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == fv and fv not in (float('inf'), float('-inf')):
            vals.append(fv)

    if len(vals) >= 2:
        ymin = min(vals)
        ymax = max(vals)
    else:
        # Fallback range in fractional return space
        ymin = tol - 0.01
        ymax = tol + 0.01

    span = max(1e-9, ymax - ymin)
    pad = max(span * 0.20, near * 2.0, 0.002)  # ~0.2% min pad

    y0 = min(ymin, tol) - pad
    y1 = max(ymax, tol + near) + pad

    # FAIL (below cutoff)
    fig.add_hrect(y0=y0, y1=tol, fillcolor=FAIL_COLOR, opacity=0.08, line_width=0, layer="below")
    # WARN (near cutoff)
    fig.add_hrect(y0=tol, y1=tol + near, fillcolor=WARN_COLOR, opacity=0.08, line_width=0, layer="below")
    # PASS (above cutoff)
    fig.add_hrect(y0=tol + near, y1=y1, fillcolor=PASS_COLOR, opacity=0.06, line_width=0, layer="below")

    return (y0, y1)


def _rs_violin_fig(rs_returns: pd.Series, tol: float, near: float):
    vals = pd.to_numeric(rs_returns, errors="coerce").dropna()
    fig = go.Figure()

    # Single violin = density shape + embedded box (median/IQR) for instant read.
    fig.add_trace(
        go.Violin(
            y=vals,
            box_visible=True,
            meanline_visible=False,
            points=False,
            line_color=ACCENT_BLUE,
            fillcolor="rgba(41,121,255,0.18)",
        )
    )
    y0, y1 = _rs_add_bands(fig, tol=tol, near=near, y_min=float(vals.min()) if len(vals) else None, y_max=float(vals.max()) if len(vals) else None)

    # Quantile lines (no annotations to keep it clean)
    if len(vals) > 0:
        p10 = float(vals.quantile(0.10))
        p50 = float(vals.quantile(0.50))
        p90 = float(vals.quantile(0.90))
        fig.add_hline(y=p10, line_dash="dot", line_color="#666", opacity=0.45)
        fig.add_hline(y=p50, line_dash="solid", line_color="#666", opacity=0.55)
        fig.add_hline(y=p90, line_dash="dot", line_color="#666", opacity=0.45)

    # Disappoint cutoff line + label
    fig.add_hline(
        y=tol,
        line_dash="dash",
        line_color=ACCENT_ORANGE,
        opacity=0.9,
        annotation_text="disappoint cutoff",
        annotation_position="top left",
    )

    fig.update_yaxes(range=[y0, y1], tickformat=".1%")
    _style_fig(fig, title="Rolling starts: return distribution")
    fig.update_layout(height=300)
    return fig


def _rs_timeline_fig(rs_df: pd.DataFrame, ret_col: str, tol: float, near: float):
    tmp = rs_df.copy()
    tmp["start_dt"] = pd.to_datetime(tmp["start_dt"], errors="coerce")
    tmp[ret_col] = pd.to_numeric(tmp[ret_col], errors="coerce")
    tmp = tmp.dropna(subset=["start_dt", ret_col]).sort_values("start_dt")

    r = tmp[ret_col]

    # Zone colors per point
    zones = np.where(r < tol, "FAIL", np.where(r < tol + near, "WARN", "PASS"))
    color_map = {"PASS": PASS_COLOR, "WARN": WARN_COLOR, "FAIL": FAIL_COLOR}
    colors = [color_map.get(z, "#999") for z in zones]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=tmp["start_dt"],
            y=r,
            mode="markers",
            marker=dict(color=colors, size=6, opacity=0.55),
            hovertemplate="%{x|%Y-%m-%d}<br>return: %{y:.2%}<extra></extra>",
            showlegend=False,
        )
    )

    # Rolling mean (context line)
    n = len(tmp)
    w = int(max(7, min(35, n * 0.06)))  # ~6% of points, clamped
    minp = max(3, w // 3)
    roll = r.rolling(window=w, center=True, min_periods=minp).mean()
    fig.add_trace(
        go.Scatter(
            x=tmp["start_dt"],
            y=roll,
            mode="lines",
            line=dict(color=ACCENT_BLUE, width=3),
            hovertemplate="rolling avg: %{y:.2%}<extra></extra>",
            showlegend=False,
        )
    )
    y0, y1 = _rs_add_bands(fig, tol=tol, near=near, y_min=float(r.min()) if len(r) else None, y_max=float(r.max()) if len(r) else None)
    fig.add_hline(y=tol, line_dash="dash", line_color=ACCENT_ORANGE, opacity=0.9)

    fig.update_yaxes(range=[y0, y1], tickformat=".1%")
    _style_fig(fig, title="Rolling starts: return vs start date")
    fig.update_layout(height=320)
    return fig



def _dist_boxstrip_fig(p10, p25, p50, p75, p90, title: str, *, digits: int = 1, zero_line: bool = True):
    """Compact distribution strip:
    - whiskers: p10→p90
    - box: p25→p75 (typical zone)
    - median: p50
    - optional 0% reference line (for return-style metrics)
    """
    import plotly.graph_objects as _go

    a = float(p10); b = float(p25); c = float(p50); d = float(p75); e = float(p90)

    lo = min(a, b, c, d, e, 0.0) if zero_line else min(a, b, c, d, e)
    hi = max(a, b, c, d, e, 0.0) if zero_line else max(a, b, c, d, e)
    span = max(1e-9, hi - lo)
    pad = 0.08 * span
    xmin = lo - pad
    xmax = hi + pad


    # If the distribution is essentially flat (p10≈p50≈p90), Plotly will collapse the x-range
    # and our 'Bad/Typical/Good' labels stack on top of each other. Expand the view so the
    # micro-chart stays readable without changing the overall visual style.
    degenerate = (abs(hi - lo) < 1e-6)
    if degenerate:
        mid = float(c)
        delta = max(0.01, abs(mid) * 0.05)  # at least ±1% in return units
        xmin = mid - delta
        xmax = mid + delta

    # Keep the micro-strip readable in narrow card layouts:
    # when percentiles collapse (or are extremely close), the whisker/box becomes a 1px line.
    # We enforce a minimum visible width proportional to the shown range, while keeping hover
    # values tied to the real percentiles.
    _view_w = float(xmax - xmin)
    _min_whisk_w = 0.14 * _view_w
    _min_box_w = 0.08 * _view_w

    a_draw, e_draw = a, e
    if abs(e - a) < _min_whisk_w:
        _mid_we = 0.5 * (a + e)
        a_draw = _mid_we - 0.5 * _min_whisk_w
        e_draw = _mid_we + 0.5 * _min_whisk_w

    b_draw, d_draw = b, d
    if abs(d - b) < _min_box_w:
        _mid_bd = 0.5 * (b + d)
        b_draw = _mid_bd - 0.5 * _min_box_w
        d_draw = _mid_bd + 0.5 * _min_box_w

    # Clamp to the shown range
    a_draw = max(xmin, a_draw); e_draw = min(xmax, e_draw)
    b_draw = max(xmin, b_draw); d_draw = min(xmax, d_draw)

    fig = _go.Figure()

    # 0% reference
    if zero_line and xmin <= 0.0 <= xmax:
        fig.add_shape(
            type="line",
            x0=0.0, x1=0.0,
            y0=-0.26, y1=0.26,
            line=dict(width=1, dash="dot", color="rgba(49,51,63,0.35)"),
        )
        fig.add_annotation(
            x=0.0, y=0.33,
            xref="x", yref="y",
            text="0%",
            showarrow=False,
            font=dict(size=9, color="rgba(49,51,63,0.55)"),
        )

    # whiskers
    fig.add_shape(type="line", x0=a_draw, x1=e_draw, y0=0.0, y1=0.0, line=dict(width=2, color="rgba(49,51,63,0.45)"))
    # end ticks
    for x in (a_draw, e_draw):
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.12, y1=0.12, line=dict(width=2, color="rgba(49,51,63,0.45)"))

    # typical zone (p25→p75)
    fig.add_shape(
        type="rect",
        x0=b_draw, x1=d_draw,
        y0=-0.11, y1=0.11,
        line=dict(width=1, color="rgba(49,51,63,0.35)"),
        fillcolor="rgba(49,51,63,0.12)",
    )

    # median
    fig.add_shape(type="line", x0=c, x1=c, y0=-0.16, y1=0.16, line=dict(width=2, color="rgba(49,51,63,0.65)"))

    # hover anchors at Bad/Typical/Good
    fig.add_trace(
        _go.Scatter(
            x=[a_draw, c, e_draw],
            y=[0.0, 0.0, 0.0],
            mode="markers",
            marker=dict(size=7, color="rgba(49,51,63,0.45)"),
            hovertemplate=(
                "Bad (p10): %{customdata[0]}<br>"
                "Typical (p50): %{customdata[1]}<br>"
                "Good (p90): %{customdata[2]}<extra></extra>"
            ),
            customdata=[[ _fmt_pct(a, digits), _fmt_pct(c, digits), _fmt_pct(e, digits) ]]*3,
            showlegend=False,
        )
    )

    # microcopy labels
        # microcopy labels
    if degenerate:
        # spread labels out across the expanded range so they don't overlap
        _w = float(xmax - xmin)
        _xa = float(xmin + 0.12 * _w)
        _xc = float(c)
        _xe = float(xmax - 0.12 * _w)
    else:
        _xa, _xc, _xe = float(a_draw), float(c), float(e_draw)

    fig.add_annotation(x=_xa, y=0.26, text="Bad", showarrow=False, font=dict(size=10, color="rgba(49,51,63,0.85)"))
    fig.add_annotation(x=_xc, y=0.26, text="Typical", showarrow=False, font=dict(size=10, color="rgba(49,51,63,0.85)"))
    fig.add_annotation(x=_xe, y=0.26, text="Good", showarrow=False, font=dict(size=10, color="rgba(49,51,63,0.85)"))

    fig.update_layout(
        height=95,
        margin=dict(l=6, r=6, t=26, b=8),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=12)),
        xaxis=dict(visible=False, range=[xmin, xmax]),
        yaxis=dict(visible=False, range=[-0.45, 0.45]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _dist_bar_fig(p10, p50, p90, title: str, *, digits: int = 1, zero_line: bool = True):
    """Backward-compatible: draws a box-strip, using p50 as the 'typical zone' if p25/p75 are unknown."""
    return _dist_boxstrip_fig(p10, p50, p50, p50, p90, title, digits=digits, zero_line=zero_line)

def _verdict_color(v: str) -> str:
    return VERDICT_COLORS.get(str(v or "").upper(), NEUTRAL_COLOR)

# =============================================================================
# App meta
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
TMP_DIR = REPO_ROOT / ".ui_tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

PY = sys.executable

st.set_page_config(page_title="Spot Strategy Stress Lab", layout="wide")

st.markdown(
    """
<style>
/* Hide Plotly legend titles to avoid JS showing 'undefined' */
.js-plotly-plot .legendtitletext { display: none !important; }
.js-plotly-plot .legendtitle { display: none !important; }


/* Summary strip (slider → summary strip → charts) */
.ff-summary-strip { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; margin: 6px 0 6px 0; }
.ff-summary-stat { border:1px solid rgba(49,51,63,0.18); border-radius:12px; padding:8px 10px; background: rgba(255,255,255,0.60); min-width: 140px; }
.ff-summary-stat .label { font-size:0.74rem; color: rgba(49,51,63,0.70); margin-bottom:2px; }
.ff-summary-stat .value { font-size:1.05rem; font-weight:650; color: rgba(17,24,39,0.92); line-height:1.05; }
.ff-summary-stat.big .value { font-size:1.35rem; }
.ff-summary-chip { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);
                   font-size:0.82rem; background: rgba(149,165,166,0.10); white-space:nowrap; }

/* ===== Founder's Foundry polish kit (dossier + build sheet) ===== */
.ff-badge-stack { display:flex; flex-direction:row; gap:8px; flex-wrap:wrap; justify-content:flex-end; align-items:center; }
.ff-badge { display:inline-flex; align-items:center; gap:8px; padding:4px 10px; border-radius:999px;
            border:1px solid rgba(49,51,63,0.18); background: rgba(149,165,166,0.10);
            font-size:0.80rem; font-weight:650; white-space:nowrap; }
.ff-badge.big { padding:6px 12px; font-size:0.95rem; }
.ff-badge.pass { background: rgba(46, 204, 113, 0.16); }
.ff-badge.fail { background: rgba(231, 76, 60, 0.16); }
.ff-badge.warn { background: rgba(241, 196, 15, 0.18); }
.ff-badge.neutral { background: rgba(149,165,166,0.10); }
.ff-badge .k { opacity:0.75; font-weight:650; }

.ff-chip-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:6px; }
.ff-chip { display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);
           font-size:0.82rem; background: rgba(149,165,166,0.10); white-space:nowrap; }

.ff-kpi-strip { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
.ff-kpi { flex: 1 1 180px; border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:8px 10px; background: rgba(255,255,255,0.55); }
.ff-kpi .label { font-size:0.74rem; opacity:0.70; margin-bottom:2px; }
.ff-kpi .value { font-size:1.05rem; font-weight:650; line-height:1.05; }
.ff-kpi.big .value { font-size:1.35rem; }

.ff-score-strip { display:flex; flex-wrap:wrap; gap:10px; margin: 8px 0 6px 0; }
.ff-score { flex: 1 1 260px; border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:8px 10px; background: rgba(255,255,255,0.55); }
.ff-score .top { display:flex; justify-content:space-between; gap:10px; align-items:baseline; }
.ff-score .top .label { font-size:0.74rem; opacity:0.70; }
.ff-score .top .value { font-size:0.90rem; font-weight:650; }
.ff-bar { height:6px; border-radius:999px; background: rgba(49,51,63,0.10); overflow:hidden; margin-top:6px; }
.ff-bar > div { height:100%; width: var(--pct, 0%); background: rgba(52, 152, 219, 0.70); }

.ff-idrow { display:flex; align-items:center; gap:10px; margin-top:6px; }
.ff-idrow code { padding:4px 8px; border-radius:10px; border:1px solid rgba(49,51,63,0.18); background: rgba(149,165,166,0.10); font-size:0.85rem; }
.ff-idrow button { border-radius:10px; border:1px solid rgba(49,51,63,0.18); background: rgba(255,255,255,0.60);
                  padding:4px 10px; font-size:0.82rem; cursor:pointer; }

.ff-kv { display:flex; flex-direction:column; gap:8px; }
.ff-kv-row { display:flex; gap:10px; align-items:baseline; }
.ff-kv-row .k { min-width: 140px; font-weight:650; }
.ff-kv-row .v { flex:1; }

.ff-readouts { display:flex; flex-direction:column; gap:10px; }
.ff-readout .label { font-size:0.78rem; opacity:0.72; }
.ff-readout .value { font-size:0.95rem; font-weight:650; margin-top:1px; }

.ff-grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:10px; margin-top:6px; }
.ff-mini { border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:8px 10px; background: rgba(255,255,255,0.55); }
.ff-mini .label { font-size:0.74rem; opacity:0.70; }
.ff-mini .value { font-size:1.15rem; font-weight:750; margin-top:2px; }

.ff-callout { border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:10px 12px; background: rgba(241,196,15,0.12); margin-top:10px; }
.ff-callout .label { font-size:0.78rem; opacity:0.72; margin-bottom:2px; }
.ff-callout .value { font-weight:750; }

/* Workflow (build sheet) */
.ff-workflow { display:flex; flex-direction:column; gap:10px; margin-top:6px; }
.ff-step { display:flex; gap:10px; align-items:flex-start; border:1px solid rgba(49,51,63,0.14);
          border-radius:14px; padding:10px 10px; background: rgba(255,255,255,0.55); }
.ff-step .n { width:26px; height:26px; border-radius:999px; display:flex; align-items:center; justify-content:center;
             font-weight:800; font-size:0.85rem; background: rgba(52,152,219,0.18); color: rgba(17,24,39,0.90); margin-top:2px; flex: 0 0 auto; }
.ff-step .t { font-weight:800; }
.ff-step .d { font-size:0.90rem; color: rgba(49,51,63,0.82); margin-top:2px; line-height:1.25; }
.ff-step .meta { margin-top:4px; }


/* ===== Skill Build UI (game-style) ===== */
.ff-skill-row { display:flex; flex-wrap:wrap; gap:12px; align-items:stretch; margin: 8px 0 6px 0; }
.ff-skill-card { border:1px solid rgba(49,51,63,0.18); border-radius:18px; padding:10px 12px;
                 background: rgba(255,255,255,0.55); box-shadow: 0 1px 10px rgba(0,0,0,0.04);
                 min-height: 86px; }
.ff-skill-card .t { font-weight:800; font-size:0.95rem; line-height:1.05; }
.ff-skill-card .s { font-size:0.80rem; opacity:0.78; margin-top:6px; line-height:1.2; }
.ff-skill-card .k { font-size:0.74rem; opacity:0.70; margin-top:2px; }
.ff-skill-card.off { opacity:0.55; }
.ff-skill-card.active { border-color: rgba(52,152,219,0.85); box-shadow: 0 0 0 3px rgba(52,152,219,0.18); }
.ff-skill-card.warn { border-color: rgba(241,196,15,0.85); box-shadow: 0 0 0 3px rgba(241,196,15,0.16); }

.ff-skill-econ { background: linear-gradient(135deg, rgba(46,204,113,0.20), rgba(46,204,113,0.04)); }
.ff-skill-trigger { background: linear-gradient(135deg, rgba(155,89,182,0.20), rgba(52,152,219,0.06)); }
.ff-skill-gate { background: linear-gradient(135deg, rgba(52,152,219,0.20), rgba(26,188,156,0.05)); }
.ff-skill-alloc { background: linear-gradient(135deg, rgba(241,196,15,0.22), rgba(241,196,15,0.05)); }
.ff-skill-risk { background: linear-gradient(135deg, rgba(231,76,60,0.18), rgba(231,76,60,0.04)); }

.ff-build-summary { margin-top: 6px; margin-bottom: 6px; }
.ff-build-summary code { font-size: 0.85rem; }

.ff-flow { border:1px solid rgba(49,51,63,0.12); border-radius:18px; padding:10px 12px; background: rgba(255,255,255,0.52); }
.ff-flow .hdr { display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap; }
.ff-flow .hdr .title { font-weight:850; }
.ff-flow .hdr .sub { font-size:0.80rem; opacity:0.78; }
.ff-flow-steps { margin-top: 10px; display:flex; flex-direction:column; gap:10px; }
.ff-flow-step { display:flex; gap:10px; align-items:flex-start; }
.ff-flow-dot { width:10px; height:10px; border-radius:50%; margin-top:6px; flex:0 0 auto; background: rgba(49,51,63,0.45); }
.ff-flow-node { flex:1 1 auto; padding:8px 10px; border-radius:16px; border:1px solid rgba(49,51,63,0.16);
               background: rgba(255,255,255,0.55); }
.ff-flow-node .t { font-weight:800; }
.ff-flow-node .d { font-size:0.86rem; opacity:0.80; margin-top:2px; line-height:1.25; }

.ff-module { border:1px solid rgba(49,51,63,0.14); border-radius:18px; padding:10px 12px; background: rgba(255,255,255,0.44); margin-bottom:12px; }
.ff-module.active { border-color: rgba(52,152,219,0.70); box-shadow: 0 0 0 3px rgba(52,152,219,0.12); }
.ff-module .mod-hdr { display:flex; justify-content:space-between; align-items:flex-end; gap:10px; flex-wrap:wrap; }
.ff-module .mod-hdr .left { font-weight:850; }
.ff-module .mod-hdr .right { font-weight:750; }
.ff-mini { font-size:0.80rem; opacity:0.78; }



/* ===== Gate logic tree (visual) ===== */
.ff-gate-tree-wrap { border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:10px 12px; background: rgba(255,255,255,0.55); margin-top: 8px; }
.ff-gate-meta { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin-bottom:10px; }
.ff-gate-meta .ff-gate-chip { display:inline-flex; align-items:center; gap:6px; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18);
                              font-size:0.82rem; background: rgba(149,165,166,0.10); white-space:nowrap; }
.ff-gate-chip.on { background: rgba(46, 204, 113, 0.18); }
.ff-gate-chip.off { background: rgba(231, 76, 60, 0.14); }
.ff-gate-chip.info { background: rgba(52, 152, 219, 0.14); }
.ff-gate-chip.warn { background: rgba(241, 196, 15, 0.18); }

.ff-gate-tree { display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; }
.ff-gate-join { display:flex; align-items:center; justify-content:center; font-weight:850; opacity:0.55; padding:0 4px; }
.ff-gate-box { flex: 1 1 260px; min-width: 240px; border:1px solid rgba(49,51,63,0.14); border-radius:14px; padding:10px; background: rgba(149,165,166,0.08); }
.ff-gate-box .hdr { display:flex; justify-content:space-between; gap:10px; align-items:baseline; margin-bottom:6px; }
.ff-gate-box .hdr .t { font-weight:800; }
.ff-gate-box .hdr .k { font-size:0.82rem; opacity:0.70; }
.ff-gate-box .sub { font-size:0.82rem; opacity:0.75; margin-bottom:8px; }
.ff-gate-conds { display:flex; flex-wrap:wrap; gap:6px; }
.ff-gate-cond { display:inline-block; padding:3px 8px; border-radius:999px; border:1px solid rgba(49,51,63,0.16);
                font-size:0.78rem; background: rgba(255,255,255,0.55); }
.ff-gate-cond.dim { opacity:0.72; }

.ff-gate-box.regime { background: rgba(46, 204, 113, 0.10); }
.ff-gate-box.triggers { background: rgba(155, 89, 182, 0.10); }
.ff-gate-box.result { background: rgba(52, 152, 219, 0.10); }
.ff-gate-box.result.on { background: rgba(46, 204, 113, 0.14); }
.ff-gate-box.result.off { background: rgba(231, 76, 60, 0.12); }

.ff-clause-grid { display:flex; flex-direction:column; gap:8px; }
.ff-clause { border:1px dashed rgba(49,51,63,0.22); border-radius:12px; padding:8px; background: rgba(255,255,255,0.45); }
.ff-clause.on { border-color: rgba(46, 204, 113, 0.70); box-shadow: 0 0 0 2px rgba(46, 204, 113, 0.14) inset; }
.ff-clause .ct { font-weight:800; font-size:0.82rem; margin-bottom:6px; display:flex; justify-content:space-between; gap:10px; }
.ff-clause .ct .mode { font-size:0.78rem; opacity:0.70; font-weight:650; }
.ff-gate-foot { font-size:0.80rem; opacity:0.74; margin-top:8px; }
</style>
""",
    unsafe_allow_html=True,
)
st.title("Spot Strategy Stress Lab")
st.caption("Spot-only. Batch → Rolling Starts → Walkforward → Grand verdict.")

# =============================================================================
# Small utilities
# =============================================================================

def _now_slug() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _slug(s: Any, *, max_len: int = 140) -> str:
    s = str(s or "").strip()
    if not s:
        return "run"
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "run")[: int(max_len)]


def _fmt_pct(x: Any, *, digits: int = 2) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "n/a"



def _summary_strip_html(
    stats: List[Tuple[str, str, bool]],
    *,
    chips: Optional[List[str]] = None,
) -> str:
    """Render a compact summary strip (HTML) for KPI-style readouts."""
    import html as _html

    parts: List[str] = []
    for label, value, big in stats:
        cls = "ff-summary-stat big" if big else "ff-summary-stat"
        parts.append(
            f"<div class='{cls}'>"
            f"<div class='label'>{_html.escape(str(label))}</div>"
            f"<div class='value'>{_html.escape(str(value))}</div>"
            f"</div>"
        )
    for ch in (chips or []):
        parts.append(f"<span class='ff-summary-chip'>{_html.escape(str(ch))}</span>")
    return "<div class='ff-summary-strip'>" + "".join(parts) + "</div>"


# =============================================================================
# Plan blueprint helpers (copy/paste)
# =============================================================================


def _ff_badge_html(label: str, verdict: str, *, big: bool = False) -> str:
    import html as _html
    v_raw = (verdict or "—").strip()
    v = v_raw.upper()
    cls = "neutral"
    if "PASS" in v:
        cls = "pass"
    elif "FAIL" in v:
        cls = "fail"
    elif ("WARN" in v) or ("CAUTION" in v):
        cls = "warn"
    size = " big" if big else ""
    return (
        f"<span class='ff-badge {cls}{size}'>"
        f"<span class='k'>{_html.escape(str(label))}</span>"
        f"<span>{_html.escape(v_raw or '—')}</span>"
        f"</span>"
    )

def _ff_badge_stack_html(items: List[Tuple[str, str, bool]]) -> str:
    parts = ["<div class='ff-badge-stack'>"]
    for label, verdict, big in items:
        parts.append(_ff_badge_html(label, verdict, big=bool(big)))
    parts.append("</div>")
    return "".join(parts)

def _ff_chip_row_html(items: List[str]) -> str:
    import html as _html
    if not items:
        return ""
    chips = "".join([f"<span class='ff-chip'>{_html.escape(str(x))}</span>" for x in items if str(x).strip()])
    return f"<div class='ff-chip-row'>{chips}</div>"

def _ff_kpi_strip_html(items: List[Tuple[str, str, bool]]) -> str:
    import html as _html
    parts = ["<div class='ff-kpi-strip'>"]
    for label, value, big in items:
        cls = "ff-kpi big" if big else "ff-kpi"
        parts.append(
            f"<div class='{cls}'>"
            f"<div class='label'>{_html.escape(str(label))}</div>"
            f"<div class='value'>{_html.escape(str(value))}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)

def _ff_score_strip_html(items: List[Tuple[str, str, float]]) -> str:
    import html as _html
    parts = ["<div class='ff-score-strip'>"]
    for label, value, pct in items:
        try:
            p = float(pct)
            if not math.isfinite(p):
                p = 0.0
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        parts.append(
            "<div class='ff-score' style='--pct: %.1f%%'>"
            "<div class='top'>"
            "<div class='label'>%s</div>"
            "<div class='value'>%s</div>"
            "</div>"
            "<div class='ff-bar'><div></div></div>"
            "</div>"
            % (p * 100.0, _html.escape(str(label)), _html.escape(str(value)))
        )
    parts.append("</div>")
    return "".join(parts)

def _ff_kvlist_html(items: List[Tuple[str, List[str]]]) -> str:
    import html as _html
    parts = ["<div class='ff-kv'>"]
    for k, vals in items:
        chips = "".join([f"<span class='ff-chip'>{_html.escape(str(v))}</span>" for v in (vals or []) if str(v).strip()])
        parts.append(
            f"<div class='ff-kv-row'>"
            f"<div class='k'>{_html.escape(str(k))}</div>"
            f"<div class='v'>{chips}</div>"
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)



def _ff_workflow_html(steps: List[Dict[str, Any]]) -> str:
    """Compact workflow list with numbered steps (uses existing chip styles)."""
    import html as _html
    if not steps:
        return ""
    parts = ["<div class='ff-workflow'>"]
    for i, s in enumerate(steps, 1):
        title = str(s.get("title") or "").strip()
        desc = str(s.get("desc") or "").strip()
        chips = s.get("chips") or []
        chips_html = _ff_chip_row_html([str(x) for x in chips if str(x).strip()])
        parts.append(
            "<div class='ff-step'>"
            f"<div class='n'>{i}</div>"
            "<div style='flex:1'>"
            f"<div class='t'>{_html.escape(title)}</div>"
            f"<div class='d'>{_html.escape(desc)}</div>"
            f"<div class='meta'>{chips_html}</div>"
            "</div>"
            "</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _ff_readouts_html(items: List[Tuple[str, str, float]]) -> str:
    import html as _html
    parts = ["<div class='ff-readouts'>"]
    for label, value, pct in items:
        try:
            p = float(pct)
            if not math.isfinite(p):
                p = 0.0
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        parts.append(
            "<div class='ff-readout' style='--pct: %.1f%%'>"
            "<div class='label'>%s</div>"
            "<div class='value'>%s</div>"
            "<div class='ff-bar'><div></div></div>"
            "</div>"
            % (p * 100.0, _html.escape(str(label)), _html.escape(str(value)))
        )
    parts.append("</div>")
    return "".join(parts)

def _ff_grid2_html(items: List[Tuple[str, str]]) -> str:
    import html as _html
    parts = ["<div class='ff-grid2'>"]
    for label, value in items:
        parts.append(
            "<div class='ff-mini'>"
            "<div class='label'>%s</div>"
            "<div class='value'>%s</div>"
            "</div>"
            % (_html.escape(str(label)), _html.escape(str(value)))
        )
    parts.append("</div>")
    return "".join(parts)

def _ff_callout_html(label: str, value: str) -> str:
    import html as _html
    return (
        "<div class='ff-callout'>"
        f"<div class='label'>{_html.escape(str(label))}</div>"
        f"<div class='value'>{_html.escape(str(value))}</div>"
        "</div>"
    )

def _ff_copy_id(id_str: str, *, key: str = "") -> None:
    """Compact ID + copy button.

    Uses a small HTML component so copying doesn't require extra Streamlit deps.
    Falls back to st.code if components isn't available.
    """
    try:
        import html as _html
        s = str(id_str)
        # Use a template-literal to avoid breaking HTML attribute quoting.
        s_js = s.replace("\\", "\\\\").replace("`", "\\`")
        uid = hashlib.md5((str(key) + "|" + s).encode("utf-8")).hexdigest()[:10]
        components.html(
            f"""<div class='ff-idrow'>
  <code>{_html.escape(s)}</code>
  <button id="ffcopy-{uid}">Copy</button>
</div>
<script>
  const btn = document.getElementById('ffcopy-{uid}');
  if (btn) {{
    btn.addEventListener('click', () => {{
      try {{
        navigator.clipboard.writeText(`{s_js}`);
        btn.innerText = 'Copied';
        setTimeout(() => btn.innerText = 'Copy', 900);
      }} catch (e) {{
        btn.innerText = 'Copy failed';
        setTimeout(() => btn.innerText = 'Copy', 900);
      }}
    }});
  }}
</script>""",
            height=42,
        )
    except Exception:
        st.code(str(id_str), language="text")


def _blank_dca_plan_blueprint() -> Dict[str, Any]:
    """Portable, mechanics-only JSON skeleton for the baseline plan.

    Notes
    - This is a *mechanics* blueprint (cadence, gating, sizing, exits). It is not a performance claim.
    - Unknown keys are ignored on import (forward compatible).
    - `buy_filter` expects the *internal key* (e.g. "none"), not the human label.
    - For `entry_mode="builder"`, populate `entry_logic` as:
        {
          "regime": [ {"indicator": "close", "operator": "<=", "threshold": "ema_200"}, ... ],
          "clauses": [
             [ {"indicator": "rsi_14", "operator": "<=", "threshold": 30}, ... ],
             ...
          ]
        }
      (Each clause is AND; the clause list is OR-of-AND.)
    """
    return {
        "version": 1,
        "strategy": "dca_swing",
        "market": "spot",
        "side": "long",  # spot only (no leverage)

        # Funding / buy cadence (one of: none, daily, weekly, monthly)
        "deposit_freq": "weekly",
        "deposit_amount_usd": 50.0,
        "buy_freq": "weekly",
        "buy_amount_usd": 50.0,
        "buy_mode": "scheduled",  # scheduled | signal
        "max_buys_per_gate": 0,       # 0 = unlimited (signal mode only)


        # Entry gating
        "entry_mode": "simple",          # simple | builder
        "buy_filter": "none",            # none | below_ema | rsi_oversold | macd_bullish | bb_z_low | adx_trend | donch_pullback

        # Simple filter knobs (used by some filters)
        "ema_len": 200,
        "rsi_thr": 30,
        "macd_hist_thr": 0.0,
        "bb_z_thr": -1.0,
        "adx_thr": 20.0,
        "donch_pos_thr": 0.2,

        # Builder logic (optional). Used when entry_mode=builder.
        "entry_logic": {"regime": [], "clauses": []},

        # Allocation + exits
        "max_alloc_pct": 1.0,
        "stop_loss_pct": 0.0,
        "take_profit_pct": 0.0,
        "time_stop_bars": 0,
        "trailing_stop_pct": 0.0,

        # Optional advanced exit / cash behavior
        "tp_sell_fraction": 1.0,  # fraction of position to sell on TP (1.0 = full)
        "reserve_frac_of_proceeds": 0.0,      # fraction of sell proceeds to keep reserved in cash (0.0 = none)
    }



def _clear_builder_gate_ui_state(ss: dict):
    """Reset builder-gate widget keys back to disabled."""
    # Regime slots (2)
    for i in range(1, 3):
        ss[f"new.regime{i}.type"] = "— (disabled)"
        for k in (f"new.regime{i}.op", f"new.regime{i}.thr", f"new.regime{i}.ema_len"):
            ss.pop(k, None)

    # Trigger clauses (3 clauses × 3 conditions)
    for ci in range(1, 4):
        for cj in range(1, 4):
            ss[f"new.cl{ci}.c{cj}.type"] = "— (disabled)"
            for k in (f"new.cl{ci}.c{cj}.op", f"new.cl{ci}.c{cj}.thr", f"new.cl{ci}.c{cj}.ema_len"):
                ss.pop(k, None)


def _apply_builder_entry_logic_to_ui(ss: dict, entry_logic: dict):
    """Populate the Logic Builder UI widget state from blueprint entry_logic."""
    if not isinstance(entry_logic, dict):
        entry_logic = {"regime": [], "clauses": []}

    def _apply_cond(prefix: str, cond: dict | None):
        if not isinstance(cond, dict):
            return

        ind = str(cond.get("indicator", "")).strip()
        op = str(cond.get("operator", "<=")).strip()
        thr = cond.get("threshold", None)

        # EMA reference
        if ind == "close" and isinstance(thr, str) and thr.startswith("ema_"):
            ss[f"{prefix}.type"] = "price_vs_ema"
            ss[f"{prefix}.op"] = op if op in ("<=", "<", ">=", ">") else "<="
            try:
                ema_len = int(thr.split("_", 1)[1])
                ss["new.ema_len"] = ema_len  # keep shared knob consistent
                ss[f"{prefix}.ema_len"] = ema_len
            except Exception:
                pass
            return

        # Numeric threshold conditions
        type_map = {
            "rsi_14": "rsi_14",
            "adx_14": "adx_14",
            "bb_z_20": "bb_z_20",
            "macd_hist": "macd_hist_12_26_9",
            "macd_hist_12_26_9": "macd_hist_12_26_9",
            "donch_pos_20": "donch_pos_20",
        }
        ui_type = type_map.get(ind, None)
        if ui_type is None:
            return

        ss[f"{prefix}.type"] = ui_type
        ss[f"{prefix}.op"] = op if op in ("<=", "<", ">=", ">") else "<="
        try:
            ss[f"{prefix}.thr"] = float(thr)
        except Exception:
            # leave default
            pass

    # Start from a clean slate
    _clear_builder_gate_ui_state(ss)

    regime = entry_logic.get("regime") or []
    clauses = entry_logic.get("clauses") or []

    # Regime (0..2)
    for i, cond in enumerate(regime[:2], start=1):
        _apply_cond(f"new.regime{i}", cond)

    # Clauses: list[clause], where clause is list[cond]
    for ci, clause in enumerate(clauses[:3], start=1):
        if not isinstance(clause, (list, tuple)):
            continue
        for cj, cond in enumerate(list(clause)[:3], start=1):
            _apply_cond(f"new.cl{ci}.c{cj}", cond)


def _apply_dca_plan_blueprint(bp: Dict[str, Any]) -> Tuple[bool, str]:
    """Best-effort: apply a pasted plan blueprint into Streamlit session state.

    This intentionally prioritizes *mechanics* fields and ignores unknown keys.
    """
    if not isinstance(bp, dict):
        return False, "Blueprint must be a JSON object (top-level dictionary)."

    def _pick(*keys, default=None):
        for k in keys:
            if k in bp:
                return bp[k]
        return default

    # Cadence / cash-in
    dep_freq = str(_pick("deposit_freq", "deposit_frequency", default="weekly") or "weekly").lower()
    buy_freq = str(_pick("buy_freq", "buy_frequency", default="weekly") or "weekly").lower()
    dep_freq = dep_freq if dep_freq in {"none", "daily", "weekly", "monthly"} else "weekly"
    buy_freq = buy_freq if buy_freq in {"none", "daily", "weekly", "monthly"} else "weekly"

    st.session_state["new.deposit_freq"] = dep_freq
    st.session_state["new.buy_freq"] = buy_freq

    # Buy mode (scheduled vs signal-driven)
    bm = str(_pick("buy_mode", "buy_trigger_mode", default="scheduled") or "scheduled").strip().lower()
    if bm not in {"scheduled", "signal"}:
        bm = "scheduled"
    st.session_state["new.buy_mode"] = bm
    try:
        st.session_state["new.max_buys_per_gate"] = int(float(_pick("max_buys_per_gate", default=0) or 0))
    except Exception:
        st.session_state["new.max_buys_per_gate"] = 0


    try:
        # Normalize legacy / friendly inputs (keep imports resilient).
        if isinstance(bp, dict):
            bf = bp.get("buy_filter")
            if isinstance(bf, str):
                _bf_label_to_key = {
                    "Always buy (no filter)": "none",
                    "Buy dips below EMA": "below_ema",
                    "RSI oversold": "rsi_oversold",
                    "Momentum (MACD bullish)": "macd_bullish",
                    "Bollinger z low": "bb_z_low",
                    "Trend strength (ADX)": "adx_trend",
                    "Donchian pullback": "donch_pullback",
                }
                if bf in _bf_label_to_key:
                    bp = dict(bp)
                    bp["buy_filter"] = _bf_label_to_key[bf]

            em = bp.get("entry_mode")
            if isinstance(em, str):
                _em_norm = {
                    "logic_builder": "builder",
                    "logic": "builder",
                    "builder": "builder",
                    "simple": "simple",
                }.get(em.strip().lower(), em)
                if _em_norm != em:
                    bp = dict(bp)
                    bp["entry_mode"] = _em_norm

        st.session_state["new.deposit_amount"] = float(_pick("deposit_amount_usd", "deposit_amount", default=50.0) or 0.0)
    except Exception:
        pass
    try:
        st.session_state["new.buy_amount"] = float(_pick("buy_amount_usd", "buy_amount", default=50.0) or 0.0)
    except Exception:
        pass

    # Entry mode
    em = str(_pick("entry_mode", default="simple") or "simple").lower()
    wants_builder = em.startswith("b") or ("builder" in em) or ("logic" in em)

    entry_logic = _pick("entry_logic", default=None)
    if wants_builder and isinstance(entry_logic, dict) and (entry_logic.get("regime") or entry_logic.get("clauses")):
        st.session_state["new.entry_mode"] = "Logic builder (regime + triggers)"
        st.session_state["new.blueprint_entry_logic"] = entry_logic
    else:
        st.session_state["new.entry_mode"] = "Simple (one filter)"
        st.session_state["new.blueprint_entry_logic"] = None

    # Simple filter (TradingView-style)
    bf = _pick("buy_filter", default=None)
    if isinstance(bf, str) and bf.strip():
        st.session_state["new.buy_filter"] = bf.strip()

    # Filter knobs
    for k_src, k_state in [
        ("ema_len", "new.ema_len"),
        ("rsi_thr", "new.rsi_thr"),
        ("macd_hist_thr", "new.macd_hist_thr"),
        ("bb_z_thr", "new.bb_z_thr"),
        ("adx_thr", "new.adx_thr"),
        ("donch_pos_thr", "new.donch_pos_thr"),
    ]:
        if k_src in bp:
            try:
                st.session_state[k_state] = float(bp[k_src]) if k_src != "ema_len" else int(bp[k_src])
            except Exception:
                pass

    # Allocation + exits
    try:
        st.session_state["new.max_alloc_pct"] = float(_pick("max_alloc_pct", default=1.0) or 1.0)
    except Exception:
        pass

    # Optional cash reserve + partial TP selling (if exposed/used)
    try:
        _rf = _pick("reserve_frac_of_proceeds", "reserve_frac", default=None)
        if _rf is not None:
            st.session_state["new.reserve_frac"] = max(0.0, min(1.0, float(_rf)))
    except Exception:
        pass
    try:
        _tsf = _pick("tp_sell_fraction", "take_profit_sell_fraction", default=None)
        if _tsf is not None:
            st.session_state["new.tp_sell_frac"] = max(0.0, min(1.0, float(_tsf)))
    except Exception:
        pass


    try:
        st.session_state["new.sl_pct_ui"] = float(_pick("stop_loss_pct", "sl_pct", default=0.0) or 0.0)
    except Exception:
        pass
    try:
        st.session_state["new.tp_pct_ui"] = float(_pick("take_profit_pct", "tp_pct", default=0.0) or 0.0)
    except Exception:
        pass
    try:
        st.session_state["new.max_hold_bars"] = int(_pick("time_stop_bars", "max_hold_bars", default=0) or 0)
    except Exception:
        pass
    try:
        st.session_state["new.trail_pct_ui"] = float(_pick("trailing_stop_pct", "trail_pct", default=0.0) or 0.0)
    except Exception:
        pass

    return True, "Blueprint applied."

def _blueprint_spec_text() -> str:
    """Human-readable spec for the portable plan blueprint (v1).

    This is intentionally written for copy/paste into other tools.
    Keep it aligned with _apply_dca_plan_blueprint() + the builder UI.
    """
    return """\
PLAN BLUEPRINT SPEC (v1)

Intent
- Portable JSON describing *mechanics* for a spot-only, long-only DCA/Swing plan.
- This is NOT a performance model and NOT a recommendation.

Top-level (required on import; missing values are defaulted)
- version: 1
- strategy: "dca_swing"
- market: "spot"
- side: "long"

Schedules
- deposit_freq: "none" | "daily" | "weekly" | "monthly"
- deposit_amount_usd: number (>= 0)
- buy_freq: "none" | "daily" | "weekly" | "monthly"
- buy_amount_usd: number (>= 0)
- buy_mode: "scheduled" | "signal"
- max_buys_per_gate: integer (0 = unlimited; signal mode only)
  Notes:
  - scheduled: buy attempts happen only on the buy schedule; the gate can veto.
  - signal: gate decides when we are in "accumulate mode"; while gate is true, buys can fire on any bar but are spaced by buy_freq (cooldown).

Entry gate
- entry_mode: "simple" | "builder"
- buy_filter (used when entry_mode="simple"):
    - "none"             : always allow scheduled buys
    - "below_ema"        : allow only when close <= EMA(ema_len)
    - "rsi_below"        : allow only when RSI(14) <= rsi_thr
    - "bb_z_below"       : allow only when BB z-score(20) <= bb_z_thr
    - "macd_bull"        : allow only when MACD histogram >= macd_hist_thr
    - "adx_above"        : allow only when ADX(14) >= adx_thr
    - "donch_pos_below"  : allow only when Donchian position(20) <= donch_pos_thr
  Notes:
  - Import also accepts older/human labels for buy_filter; they will be normalized.

Simple-gate knobs (used depending on buy_filter)
- ema_len: 10 | 20 | 50 | 100 | 200
- rsi_thr: number (0..100)
- macd_hist_thr: number
- bb_z_thr: number
- adx_thr: number
- donch_pos_thr: number (0..1)

Builder-gate grammar (used when entry_mode="builder")
- entry_logic:
    - regime: 0..2 Conditions (AND)
    - clauses: 0..3 Clauses (OR-of-AND)
        - each Clause is 1..3 Conditions (AND)
- Condition object:
    - indicator: string
    - operator: "<=" | "<" | ">=" | ">"
    - threshold: number OR string reference (see below)
  Supported indicator vocabulary (current UI):
    - "close"
    - "rsi_14"
    - "adx_14"
    - "bb_z_20"
    - "macd_hist"
    - "donch_pos_20"
  Supported reference thresholds:
    - "ema_<LEN>" (string), where <LEN> matches ema_len (e.g., "ema_200")
  Notes:
    - The UI currently uses "close <= ema_<LEN>" for EMA regime gates.
    - Unknown indicators may be rejected or treated as inactive depending on dataset availability.

Allocation & reserves
- max_alloc_pct: number (0..1)    # fraction of equity allowed to be invested
- reserve_frac_of_proceeds: number (0..1)  # fraction of sell proceeds reserved in cash (alias: reserve_frac)

Exit controls (0 disables)
- stop_loss_pct: number (>= 0)        # percent from entry
- take_profit_pct: number (>= 0)      # percent from entry
- tp_sell_fraction: number (0..1)     # fraction sold on take-profit
- time_stop_bars: integer (>= 0)      # max holding period in bars
- trailing_stop_pct: number (>= 0)    # percent from peak

Import behavior
- Missing keys: defaulted
- Extra keys: ignored
- Types: coerced where safe (e.g., strings to floats)
"""



def _collect_dca_plan_state():
    """Return a portable snapshot of the current DCA/Swing plan knobs (mechanics-only)."""
    ss = st.session_state

    def g(key, default=None):
        return ss.get(key, default)

    def _to_float(x, default=0.0):
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _to_int(x, default=0):
        try:
            if x is None:
                return int(default)
            return int(float(x))
        except Exception:
            return int(default)

    # Start from the blank blueprint (ensures full coverage / stable keys)
    bp = _blank_dca_plan_blueprint()

    # Entry mode (UI stores a label)
    em_ui = g("new.entry_mode", "Simple (one filter)")
    em_ui_s = (str(em_ui) if em_ui is not None else "").lower()
    entry_mode = "builder" if "builder" in em_ui_s else "simple"

    bp.update(
        {
            "deposit_freq": g("new.deposit_freq", bp.get("deposit_freq", "weekly")),
            "deposit_amount_usd": _to_float(g("new.deposit_amount", bp.get("deposit_amount_usd", 50.0)), bp.get("deposit_amount_usd", 50.0)),
            "buy_freq": g("new.buy_freq", bp.get("buy_freq", "weekly")),
            "buy_amount_usd": _to_float(g("new.buy_amount", bp.get("buy_amount_usd", 50.0)), bp.get("buy_amount_usd", 50.0)),
            "buy_mode": g("new.buy_mode", bp.get("buy_mode", "scheduled")),
            "max_buys_per_gate": _to_int(g("new.max_buys_per_gate", bp.get("max_buys_per_gate", 0)), bp.get("max_buys_per_gate", 0)),
            "entry_mode": entry_mode,
            "buy_filter": g("new.buy_filter", bp.get("buy_filter", "none")),
            "ema_len": _to_int(g("new.ema_len", bp.get("ema_len", 200)), bp.get("ema_len", 200)),
            "rsi_thr": _to_float(g("new.rsi_thr", bp.get("rsi_thr", 30.0)), bp.get("rsi_thr", 30.0)),
            "macd_hist_thr": _to_float(g("new.macd_hist_thr", bp.get("macd_hist_thr", 0.0)), bp.get("macd_hist_thr", 0.0)),
            "bb_z_thr": _to_float(g("new.bb_z_thr", bp.get("bb_z_thr", -1.0)), bp.get("bb_z_thr", -1.0)),
            "adx_thr": _to_float(g("new.adx_thr", bp.get("adx_thr", 20.0)), bp.get("adx_thr", 20.0)),
            "donch_pos_thr": _to_float(g("new.donch_pos_thr", bp.get("donch_pos_thr", 0.2)), bp.get("donch_pos_thr", 0.2)),
            "max_alloc_pct": _to_float(g("new.max_alloc_pct", bp.get("max_alloc_pct", 1.0)), bp.get("max_alloc_pct", 1.0)),
            # exits / risk controls (UI keys store % in "ui" space)
            "stop_loss_pct": _to_float(g("new.sl_pct_ui", bp.get("stop_loss_pct", 0.0)), bp.get("stop_loss_pct", 0.0)),
            "take_profit_pct": _to_float(g("new.tp_pct_ui", bp.get("take_profit_pct", 0.0)), bp.get("take_profit_pct", 0.0)),
            "time_stop_bars": _to_int(g("new.max_hold_bars", bp.get("time_stop_bars", 0)), bp.get("time_stop_bars", 0)),
            "trailing_stop_pct": _to_float(g("new.trail_pct_ui", bp.get("trailing_stop_pct", 0.0)), bp.get("trailing_stop_pct", 0.0)),
            # misc knobs
            "tp_sell_fraction": _to_float(g("new.tp_sell_frac", bp.get("tp_sell_fraction", 1.0)), bp.get("tp_sell_fraction", 1.0)),
            "reserve_frac_of_proceeds": _to_float(g("new.reserve_frac", bp.get("reserve_frac_of_proceeds", bp.get("reserve_frac", 0.0))), bp.get("reserve_frac_of_proceeds", bp.get("reserve_frac", 0.0))),
        }
    )

    def _read_cond(prefix: str):
        t = g(f"{prefix}.type", "—")
        if t in (None, "—", "(disabled)", "(none)", ""):
            return None

        op = g(f"{prefix}.op", "<=")

        if t in ("price_vs_ema", "Price vs EMA"):
            ema_len = _to_int(g(f"{prefix}.ema_len", 200), 200)
            return {"indicator": "close", "operator": op, "threshold": f"ema_{ema_len}"}

        # indicator id is stored directly for non-EMA conditions (e.g. rsi_14, adx_14, bb_z_20, donch_pos_20)
        thr = g(f"{prefix}.thr", None)
        if thr is None:
            thr = 0.0
        # keep ints as ints if they were entered as ints
        try:
            thr_f = float(thr)
            thr_out = int(thr_f) if abs(thr_f - int(thr_f)) < 1e-12 else thr_f
        except Exception:
            thr_out = thr

        return {"indicator": t, "operator": op, "threshold": thr_out}

    if entry_mode == "builder":
        regime = []
        for i in (1, 2):
            c = _read_cond(f"new.regime{i}")
            if c:
                regime.append(c)

        clauses = []
        for ci in (1, 2, 3):
            conds = []
            for cj in (1, 2, 3):
                c = _read_cond(f"new.clause{ci}.cond{cj}")
                if c:
                    conds.append(c)
            if conds:
                clauses.append(conds)

        bp["entry_logic"] = {"regime": regime, "clauses": clauses}
    else:
        bp["entry_logic"] = {"regime": [], "clauses": []}

    return bp


def _apply_dca_plan_blueprint(bp: dict) -> None:
    """Apply a plan blueprint into Streamlit session state (mechanics-only)."""
    if not isinstance(bp, dict):
        return

    ss = st.session_state

    def _to_float(x, default=0.0):
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    def _to_int(x, default=0):
        try:
            if x is None:
                return int(default)
            return int(float(x))
        except Exception:
            return int(default)

    # Core
    ss["new.deposit_freq"] = bp.get("deposit_freq", ss.get("new.deposit_freq", "weekly"))
    ss["new.deposit_amount"] = _to_float(bp.get("deposit_amount_usd", ss.get("new.deposit_amount", 50.0)), ss.get("new.deposit_amount", 50.0))
    ss["new.buy_freq"] = bp.get("buy_freq", ss.get("new.buy_freq", "weekly"))
    ss["new.buy_amount"] = _to_float(bp.get("buy_amount_usd", ss.get("new.buy_amount", 50.0)), ss.get("new.buy_amount", 50.0))
    ss["new.buy_mode"] = str(bp.get("buy_mode", ss.get("new.buy_mode", "scheduled")) or "scheduled").lower().strip()
    ss["new.max_buys_per_gate"] = _to_int(bp.get("max_buys_per_gate", ss.get("new.max_buys_per_gate", 0)), ss.get("new.max_buys_per_gate", 0))

    # Entry mode: store the UI label (the radio uses labels)
    entry_mode = str(bp.get("entry_mode", "simple")).lower().strip()
    ss["new.entry_mode"] = "Logic builder (regime + triggers)" if entry_mode == "builder" else "Simple (one filter)"

    # Simple filter id (stored directly)
    ss["new.buy_filter"] = bp.get("buy_filter", ss.get("new.buy_filter", "none"))

    # Shared thresholds
    ss["new.ema_len"] = _to_int(bp.get("ema_len", ss.get("new.ema_len", 200)), ss.get("new.ema_len", 200))
    ss["new.rsi_thr"] = _to_float(bp.get("rsi_thr", ss.get("new.rsi_thr", 30.0)), ss.get("new.rsi_thr", 30.0))
    ss["new.macd_hist_thr"] = _to_float(bp.get("macd_hist_thr", ss.get("new.macd_hist_thr", 0.0)), ss.get("new.macd_hist_thr", 0.0))
    ss["new.bb_z_thr"] = _to_float(bp.get("bb_z_thr", ss.get("new.bb_z_thr", -1.0)), ss.get("new.bb_z_thr", -1.0))
    ss["new.adx_thr"] = _to_float(bp.get("adx_thr", ss.get("new.adx_thr", 20.0)), ss.get("new.adx_thr", 20.0))
    ss["new.donch_pos_thr"] = _to_float(bp.get("donch_pos_thr", ss.get("new.donch_pos_thr", 0.2)), ss.get("new.donch_pos_thr", 0.2))

    # Allocation / misc
    ss["new.max_alloc_pct"] = _to_float(bp.get("max_alloc_pct", ss.get("new.max_alloc_pct", 1.0)), ss.get("new.max_alloc_pct", 1.0))
    ss["new.reserve_frac"] = _to_float(bp.get("reserve_frac", ss.get("new.reserve_frac", 0.0)), ss.get("new.reserve_frac", 0.0))
    ss["new.tp_sell_frac"] = _to_float(bp.get("tp_sell_fraction", ss.get("new.tp_sell_frac", 1.0)), ss.get("new.tp_sell_frac", 1.0))

    # Exits (UI space)
    ss["new.sl_pct_ui"] = _to_float(bp.get("stop_loss_pct", ss.get("new.sl_pct_ui", 0.0)), ss.get("new.sl_pct_ui", 0.0))
    ss["new.tp_pct_ui"] = _to_float(bp.get("take_profit_pct", ss.get("new.tp_pct_ui", 0.0)), ss.get("new.tp_pct_ui", 0.0))
    ss["new.max_hold_bars"] = _to_int(bp.get("time_stop_bars", ss.get("new.max_hold_bars", 0)), ss.get("new.max_hold_bars", 0))
    ss["new.trail_pct_ui"] = _to_float(bp.get("trailing_stop_pct", ss.get("new.trail_pct_ui", 0.0)), ss.get("new.trail_pct_ui", 0.0))
    # Builder logic: reflect into the Logic Builder UI widget keys (so checklist + preview match)
    entry_logic = bp.get("entry_logic") or {"regime": [], "clauses": []}
    ss["new.blueprint_entry_logic"] = entry_logic
    try:
        if entry_mode == "builder":
            _apply_builder_entry_logic_to_ui(ss, entry_logic)
        else:
            _clear_builder_gate_ui_state(ss)
    except Exception:
        pass
def _render_plan_blueprint_import_ui() -> None:
    with st.expander("Plan blueprint (copy/paste)", expanded=False):
        st.caption("Portable JSON you can copy, edit, and paste back in. Mechanics-only (not recommendations).")

        tab_bp, tab_spec = st.tabs(["Blueprint", "Blueprint spec"])

        with tab_bp:
            st.markdown("**Blank blueprint**")
            st.code(json.dumps(_blank_dca_plan_blueprint(), indent=2), language="json")

            with st.expander("Builder example (copy/paste)", expanded=False):
                ex = _blank_dca_plan_blueprint()
                ex["entry_mode"] = "builder"
                ex["entry_logic"] = {
                    "regime": [{"indicator": "close", "operator": "<=", "threshold": "ema_200"}],
                    "clauses": [
                        [
                            {"indicator": "rsi_14", "operator": "<=", "threshold": 30},
                            {"indicator": "bb_z_20", "operator": "<=", "threshold": -1.0},
                        ],
                        [
                            {"indicator": "adx_14", "operator": ">=", "threshold": 20},
                        ],
                    ],
                }
                st.code(json.dumps(ex, indent=2), language="json")

            st.markdown("**Paste blueprint JSON to load**")
            # Text area state: initialize once, mutate via callbacks (Streamlit forbids post-widget mutation in the same run)
            if "plan_blueprint_paste" not in st.session_state:
                st.session_state["plan_blueprint_paste"] = ""

            def _clear_plan_blueprint_paste():
                st.session_state["plan_blueprint_paste"] = ""
            bp_text = st.text_area(
                "Blueprint JSON",
                height=220,
                placeholder='Paste JSON here (must be an object). Example: {"deposit_freq":"weekly", ... }',
                key="plan_blueprint_paste",
            )


            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button("Load blueprint", type="primary", use_container_width=True):
                                        # Parse (lenient) and merge with defaults, then apply to widget state
                    _s = (bp_text or "").strip()
                    if _s.startswith("```"):
                        _s = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", _s.strip(), flags=re.S)
                    # Allow pasting a JSON fragment without surrounding braces
                    if _s and (not _s.startswith("{")) and (not _s.startswith("[")):
                        _s = "{" + _s
                    if _s and _s.startswith("{") and (not _s.endswith("}")):
                        _s = _s + "}"
                    try:
                        _data = json.loads(_s) if _s else None
                    except Exception as _e:
                        _data = None
                        st.error(f"Invalid blueprint JSON: {_e}")
                    if isinstance(_data, dict):
                        _bp = _blank_dca_plan_blueprint()
                        _bp.update(_data)
                        _apply_dca_plan_blueprint(_bp)
                        st.success("Loaded blueprint into the build.")
                        # No explicit rerun needed; the button click already triggers a rerun.
                    else:
                        st.error("Blueprint must be a JSON object (e.g., { ... }).")
            with cols[1]:
                st.button("Clear", use_container_width=True, on_click=_clear_plan_blueprint_paste)

            st.divider()
            st.markdown("**Export current page state as blueprint**")
            cur = _collect_dca_plan_state()
            st.code(json.dumps(cur, indent=2), language="json")

        with tab_spec:
            st.markdown("**Blueprint spec (v1)**")
            st.caption("Field definitions + allowed values for power users and external tooling. Mechanics-only.")
            spec = _blueprint_spec_text()
            st.code(spec, language="text")
            st.download_button(
                "Download spec",
                data=spec.encode("utf-8"),
                file_name="plan_blueprint_spec_v1.txt",
                mime="text/plain",
                use_container_width=False,
            )



def _render_current_plan_blueprint(params: Dict[str, Any]) -> None:
    with st.expander("Current plan blueprint (copy)", expanded=False):
        st.code(json.dumps(params, indent=2), language="json")

def _fmt_num(x: Any, *, digits: int = 4) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:.{digits}f}"
    except Exception:
        return "n/a"



def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _drawdown_to_frac(dd: pd.Series) -> pd.Series:
    """Best-effort: convert drawdown series to a 0..1 fraction."""
    x = _to_float_series(dd).copy()
    # Heuristic: if values look like 20..80 (percent) convert to fraction
    finite = x.dropna()
    if len(finite) > 0:
        q95 = float(finite.quantile(0.95))
        if q95 > 2.0:  # likely percent points
            x = x / 100.0
    return x


def _pareto_frontier(df: pd.DataFrame, x: str, y: str, *, x_round: int = 6, y_eps: float = 1e-12) -> pd.DataFrame:
    """Return Pareto frontier for maximize y, minimize x.

    Notes:
    - Collapses near-duplicate x values (rounding) to avoid ugly vertical segments.
    - Uses a strict-improvement threshold on y to avoid float jitter.
    """
    if df.empty:
        return df.copy()

    tmp = df[[x, y]].dropna().copy()
    if tmp.empty:
        return tmp

    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return tmp

    # Collapse duplicate/near-duplicate x values so the frontier doesn't look like a glitchy barcode.
    tmp["_xbin"] = tmp[x].round(x_round)
    tmp = tmp.groupby("_xbin", as_index=False).agg({x: "min", y: "max"}).sort_values(x, ascending=True)

    best = -1e100
    keep_rows = []
    for _, row in tmp.iterrows():
        val = float(row[y])
        if val > best + y_eps:
            best = val
            keep_rows.append(row)

    out = pd.DataFrame(keep_rows)
    # Ensure x,y columns exist for downstream plotting.
    return out[[x, y]].sort_values(x, ascending=True)


def _pareto_frontier_rows(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    x_round: int = 6,
    y_eps: float = 1e-12,
) -> pd.DataFrame:
    """Return Pareto frontier *rows* from the original dataframe.

    We maximize y and minimize x. This version keeps the original row payload (config_id, label, trades, etc.)
    so we can show a frontier table and better hover text.

    Implementation:
    1) Round x into bins to collapse near-duplicates (avoids ugly vertical barcode segments).
    2) Within each x-bin, pick the row with max y.
    3) Sweep increasing x and keep only rows that strictly improve y (epsilon to avoid float jitter).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[x] = pd.to_numeric(tmp[x], errors="coerce")
    tmp[y] = pd.to_numeric(tmp[y], errors="coerce")
    tmp = tmp.dropna(subset=[x, y])
    if tmp.empty:
        return pd.DataFrame()

    tmp["_xbin"] = tmp[x].round(x_round)

    # Pick best-y row per bin (if ties, idxmax returns first occurrence).
    try:
        idx = tmp.groupby("_xbin")[y].idxmax()
        cand = tmp.loc[idx].sort_values(x, ascending=True)
    except Exception:
        # Fallback: no groupby for weird data
        cand = tmp.sort_values([x, y], ascending=[True, False]).drop_duplicates("_xbin", keep="first")

    best = -1e100
    keep_rows = []
    for _, row in cand.iterrows():
        val = float(row[y])
        if val > best + y_eps:
            best = val
            keep_rows.append(row)

    out = pd.DataFrame(keep_rows).drop(columns=["_xbin"], errors="ignore")
    return out.sort_values(x, ascending=True)



def _goodness_percentile(s: pd.Series, *, low_is_good: bool) -> pd.Series:
    """0..1 where 1 is best."""
    x = _to_float_series(s)
    pct = x.rank(pct=True, ascending=True)
    if low_is_good:
        n = max(int(pct.notna().sum()), 1)
        return (1.0 - pct) + (1.0 / n)
    return pct


def _fmt_money(x: Any) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "n/a"
        return f"{v:,.2f}"
    except Exception:
        return "n/a"


def _metric_label(metric_id: str) -> str:
    spec = METRICS.get(metric_id)
    return spec.label if spec else metric_id


def _metric_fmt(metric_id: str, x: Any) -> str:
    spec = METRICS.get(metric_id)
    if spec and hasattr(spec, "fmt"):
        try:
            return spec.fmt(float(x))
        except Exception:
            return str(x)
    # Fallback
    if "dd" in metric_id or "drawdown" in metric_id or metric_id.endswith("_return"):
        return _fmt_pct(x)
    return _fmt_num(x)


def _read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str, mtime: float) -> pd.DataFrame:
    _ = mtime  # cache buster
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _add_features_cached(path: str, mtime: float) -> Optional[pd.DataFrame]:
    """Load dataset + compute features once per dataset version (used only for UI context)."""
    _ = mtime  # cache buster
    if _add_features is None:
        return None
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    try:
        df.columns = [str(c).strip().lower() for c in df.columns]
    except Exception:
        pass
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return None
    return _add_features(df)


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return _read_csv_cached(str(path), path.stat().st_mtime)
    except Exception:
        return None



@st.cache_data(show_spinner=False)
def _read_parquet_cached(path: str, mtime: float) -> pd.DataFrame:
    """Cached parquet reader (mtime busts cache)."""
    _ = mtime
    return pd.read_parquet(path)


def _load_any_df(path: Path) -> Optional[pd.DataFrame]:
    """Load CSV or Parquet (best-effort) for UI preview/diagnostics."""
    try:
        if not path or not Path(path).exists():
            return None
        p = Path(path)
        suf = p.suffix.lower()
        if suf in {".parquet", ".pq"}:
            return _read_parquet_cached(str(p), p.stat().st_mtime)
        return _read_csv_cached(str(p), p.stat().st_mtime)
    except Exception:
        return None


def _load_any_df_tail(path: Path, n: int = 2500) -> Optional[pd.DataFrame]:
    """Load a tail window for preview (avoids huge loads for large CSVs)."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        suf = p.suffix.lower()
        if suf in {".parquet", ".pq"}:
            df = _read_parquet_cached(str(p), p.stat().st_mtime)
            return df.tail(n) if df is not None else None

        # CSV: stream in chunks and keep tail
        keep = None
        for chunk in pd.read_csv(str(p), chunksize=50_000):
            if keep is None:
                keep = chunk
            else:
                keep = pd.concat([keep, chunk], ignore_index=True)
            if len(keep) > n * 3:
                keep = keep.tail(n * 2).reset_index(drop=True)
        if keep is None:
            return None
        return keep.tail(n).reset_index(drop=True)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _read_catalog_cached(path: str, mtime: float) -> List[Dict[str, Any]]:
    _ = mtime  # cache buster
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(obj, list):
        out = [x for x in obj if isinstance(x, dict)]
        return out
    if isinstance(obj, dict) and isinstance(obj.get("datasets"), list):
        return [x for x in obj.get("datasets") if isinstance(x, dict)]
    return []


def _resolve_dataset_path(entry: Dict[str, Any]) -> Optional[Path]:
    try:
        raw = (
            entry.get("file_path")
            or entry.get("path")
            or entry.get("filepath")
            or entry.get("file")
            or ""
        )
        if not raw:
            return None
        p = Path(str(raw))
        if not p.is_absolute():
            # allow paths relative to repo or data dir
            cand = REPO_ROOT / p
            if cand.exists():
                return cand
            cand2 = DATA_DIR / p
            if cand2.exists():
                return cand2
            return cand  # default repo-relative
        return p
    except Exception:
        return None


def _infer_symbol_from_filename(name: str) -> str:
    try:
        base = Path(name).stem
        # common patterns: eth_daily_..., BTCUSDT_1d_..., btc_1d
        tok = re.split(r"[_\-\s]+", base)[0]
        tok = tok.upper()
        tok = tok.replace("USDT", "").replace("USD", "")
        if 2 <= len(tok) <= 12:
            return tok
        return base[:12].upper()
    except Exception:
        return str(name or "").upper()


def _safe_dt_str(x: Any) -> str:
    try:
        if x is None:
            return ""
        s = str(x).strip()
        if not s:
            return ""
        # keep YYYY-MM-DD if present
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            return s[:10]
        return s
    except Exception:
        return ""


def _dataset_option_label(d: Dict[str, Any]) -> str:
    sym = str(d.get("symbol") or d.get("ticker") or d.get("id") or "").upper()
    name = str(d.get("name") or "").strip()
    tf = str(d.get("timeframe") or d.get("tf") or "1D").upper()
    start = _safe_dt_str(d.get("start_dt") or d.get("start") or d.get("start_date"))
    end = _safe_dt_str(d.get("end_dt") or d.get("end") or d.get("end_date"))
    rows = d.get("rows") or d.get("n_rows") or d.get("bars") or None
    rng = ""
    if start and end:
        rng = f"{start} → {end}"
    elif start:
        rng = f"{start} → …"
    elif end:
        rng = f"… → {end}"
    rtxt = ""
    try:
        if rows is not None:
            rtxt = f" · {int(rows):,} bars"
    except Exception:
        rtxt = ""
    if name and name.lower() != sym.lower():
        return f"{sym} — {name} · {tf}" + (f" · {rng}" if rng else "") + rtxt
    return f"{sym} · {tf}" + (f" · {rng}" if rng else "") + rtxt


def _catalog_paths() -> List[Path]:
    return [
        DATA_DIR / "datasets" / "catalog.json",
        DATA_DIR / "catalog.json",
        DATA_DIR / "datasets_catalog.json",
    ]


def _build_fallback_catalog() -> List[Dict[str, Any]]:
    """Fallback: scan ./data for csv/parquet datasets when no catalog is present."""
    out: List[Dict[str, Any]] = []
    seen: set = set()

    # Prefer a dedicated datasets folder if it exists
    roots = []
    if (DATA_DIR / "datasets").exists():
        roots.append(DATA_DIR / "datasets")
    roots.append(DATA_DIR)

    for root in roots:
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if p.name.lower() in {"catalog.json", "datasets_catalog.json"}:
                    continue
                if p.suffix.lower() not in {".csv", ".parquet", ".pq"}:
                    continue
                rel = str(p.relative_to(REPO_ROOT)) if REPO_ROOT in p.parents else str(p)
                if rel in seen:
                    continue
                seen.add(rel)

                sym = _infer_symbol_from_filename(p.name)
                entry = {
                    "id": f"{sym}_{p.suffix.lower().lstrip('.')}:{rel}",
                    "symbol": sym,
                    "name": "",
                    "timeframe": "1D",
                    "file_path": rel,
                }
                out.append(entry)
        except Exception:
            continue

    # Ensure sample dataset still shows up if present
    sample_csv = DATA_DIR / "eth_daily_2023_to_now.csv"
    if sample_csv.exists():
        rel = str(sample_csv.relative_to(REPO_ROOT)) if REPO_ROOT in sample_csv.parents else str(sample_csv)
        entry = {
            "id": "ETH_csv:sample",
            "symbol": "ETH",
            "name": "Sample",
            "timeframe": "1D",
            "file_path": rel,
        }
        out.insert(0, entry)

    return out


def _get_dataset_catalog() -> List[Dict[str, Any]]:
    # Try explicit catalog files first
    for cp in _catalog_paths():
        try:
            if cp.exists():
                return _read_catalog_cached(str(cp), cp.stat().st_mtime)
        except Exception:
            pass
    return _build_fallback_catalog()


def _sort_filter_catalog(
    catalog: List[Dict[str, Any]],
    query: str,
    sort_by: str,
    *,
    use_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    q = str(query or "").strip().lower()
    items = []
    for d in catalog or []:
        if not isinstance(d, dict):
            continue
        sym = str(d.get("symbol") or "").lower()
        name = str(d.get("name") or "").lower()
        if q and (q not in sym and q not in name):
            continue
        items.append(d)

    def _hist_len(d: Dict[str, Any]) -> float:
        try:
            s = d.get("start_dt") or d.get("start") or d.get("start_date")
            e = d.get("end_dt") or d.get("end") or d.get("end_date")
            if not s or not e:
                return 0.0
            ds = pd.to_datetime(s, errors="coerce")
            de = pd.to_datetime(e, errors="coerce")
            if pd.isna(ds) or pd.isna(de):
                return 0.0
            return float((de - ds).days)
        except Exception:
            return 0.0

    def _updated_ts(d: Dict[str, Any]) -> float:
        try:
            u = d.get("updated_at") or d.get("updated") or d.get("last_updated") or ""
            du = pd.to_datetime(u, errors="coerce")
            if pd.isna(du):
                return 0.0
            return float(du.timestamp())
        except Exception:
            return 0.0

    sort_by = str(sort_by or "")
    if sort_by.startswith("Most") and use_counts:
        items.sort(key=lambda d: int(use_counts.get(str(d.get("id") or ""), 0)), reverse=True)
    elif sort_by.startswith("Alphabet"):
        items.sort(key=lambda d: str(d.get("symbol") or ""))
    elif sort_by.startswith("Longest"):
        items.sort(key=_hist_len, reverse=True)
    elif sort_by.startswith("Newest"):
        items.sort(key=_updated_ts, reverse=True)
    else:
        items.sort(key=lambda d: str(d.get("symbol") or ""))

    return items


def _infer_bar_ms_from_csv(path: Path) -> Optional[int]:
    """Infer median bar size in milliseconds from the first ~5000 rows of a dataset (CSV/Parquet).

    Tries 'ts' (ms epoch) first; falls back to parsing 'dt'/'date' as datetimes.
    Returns None if it can't infer a stable interval.
    """
    try:
        if not path.exists():
            return None
        p = Path(path)
        if p.suffix.lower() in {'.parquet', '.pq'}:
            sample = pd.read_parquet(str(p)).head(5000)
        else:
            sample = pd.read_csv(str(p), nrows=5000)
        if sample is None or len(sample) < 3:
            return None

        cols = {c.lower(): c for c in sample.columns}

        if "ts" in cols:
            ts = pd.to_numeric(sample[cols["ts"]], errors="coerce").dropna().astype("int64")
            if len(ts) < 3:
                return None
            diffs = ts.diff().dropna()
            med = float(diffs.median())
            if med <= 0:
                return None
            return int(med)

        for key in ["dt", "date", "datetime", "time", "timestamp"]:
            if key in cols:
                dt = pd.to_datetime(sample[cols[key]], errors="coerce", utc=True)
                dt = dt.dropna()
                if len(dt) < 3:
                    continue
                diffs = dt.diff().dropna().dt.total_seconds() * 1000.0
                med = float(diffs.median())
                if med <= 0:
                    continue
                return int(med)

        return None
    except Exception:
        return None


def _bars_per_day_from_run_meta(run_dir: Path) -> int:
    """Estimate bars/day based on the run's batch_meta.json and dataset."""
    try:
        meta = _read_json(run_dir / "batch_meta.json")
        data = meta.get("data") if isinstance(meta, dict) else None
        if not data:
            return 1
        bar_ms = _infer_bar_ms_from_csv(Path(str(data)))
        if not bar_ms or bar_ms <= 0:
            return 1
        bpd = int(round(86_400_000 / float(bar_ms)))
        return int(max(1, min(86_400, bpd)))
    except Exception:
        return 1


def _human_bar_interval_from_run(run_dir: Path) -> str:
    try:
        meta = _read_json(run_dir / "batch_meta.json")
        data = meta.get("data") if isinstance(meta, dict) else None
        if not data:
            return "unknown"
        bar_ms = _infer_bar_ms_from_csv(Path(str(data)))
        if not bar_ms:
            return "unknown"
        sec = bar_ms / 1000.0
        if sec >= 86_400:
            return f"~{sec/86_400:.1f} days/bar"
        if sec >= 3600:
            return f"~{sec/3600:.1f} hours/bar"
        if sec >= 60:
            return f"~{sec/60:.1f} minutes/bar"
        return f"~{sec:.0f} seconds/bar"
    except Exception:
        return "unknown"

@st.cache_data(show_spinner=False)
def _read_jsonl_cached(path: str, mtime: float) -> List[Dict[str, Any]]:
    _ = mtime
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                out.append(json.loads(s))
            except Exception:
                continue
    return out


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return _read_jsonl_cached(str(path), path.stat().st_mtime)




def _truthy(x: Any) -> bool:
    try:
        if x is True:
            return True
        s = str(x).strip().lower()
        return s in {"1", "true", "yes", "y", "t"}
    except Exception:
        return False


def _find_baseline_config_id(run_dir: Path) -> Optional[str]:
    """Best-effort: find the user's original (baseline) config_id for this run.

    Priority:
      1) batch_meta.json["baseline_config_id"] (newer runs)
      2) configs_resolved.jsonl entries with normalized.params.__baseline__ truthy
    """
    # 1) batch_meta.json
    try:
        meta = _read_json(Path(run_dir) / "batch_meta.json")
        cid = meta.get("baseline_config_id") if isinstance(meta, dict) else None
        if cid:
            return str(cid)
    except Exception:
        pass

    # 2) configs_resolved.jsonl
    try:
        rows = _load_jsonl(Path(run_dir) / "configs_resolved.jsonl")
        for r in rows:
            if not isinstance(r, dict):
                continue
            cid = r.get("config_id")
            norm = r.get("normalized") or {}
            params = norm.get("params") if isinstance(norm, dict) else None
            if not isinstance(params, dict):
                params = {}
            if _truthy(params.get("__baseline__")):
                if cid:
                    return str(cid)
    except Exception:
        pass

    return None



def _lookup_row_by_config_id(cid: str, *dfs: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """Return the first matching row across candidate DataFrames (config_id match), else None."""
    if cid is None:
        return None
    key = _canon_cfg_id(str(cid).strip())
    for df in dfs:
        try:
            if df is None or df.empty or "config_id" not in df.columns:
                continue
            s = df["config_id"].astype(str).str.strip().map(_canon_cfg_id)
            m = df[s == key]
            if not m.empty:
                return m.iloc[0]
        except Exception:
            continue
    return None


def _resolve_selected_ctx(run_dir, pick, df2=None, top_map=None, rs_dir_effective=None, wf_dir_effective=None):
    """Resolve a single source-of-truth context for the selected config.
    Keeps Evidence UI consistent and prevents scattered lookups / NameErrors.
    Returns a dict with: row, cfg_norm, art_dir, trades_df, equity_df, trades_n, and simple availability flags.
    """
    from pathlib import Path as _Path
    import pandas as _pd

    ctx = {
        "pick": str(pick),
        "run_dir": _Path(run_dir) if run_dir is not None else None,
        "row": {},
        "cfg_norm": {},
        "art_dir": None,
        "trades_df": None,
        "equity_df": None,
        "trades_n": None,
        "has_artifacts": False,
        "has_receipts": True,
    }

    # Selected row (shortlist table)
    try:
        if df2 is not None and hasattr(df2, "columns") and "config_id" in df2.columns:
            sel = df2[df2["config_id"].astype(str) == str(pick)]
            if sel is not None and not sel.empty:
                ctx["row"] = dict(sel.iloc[0])
    except Exception:
        ctx["row"] = {}

    # Normalized config (optional)
    try:
        cfg_norm = {}
        if ctx["run_dir"] is not None:
            p = ctx["run_dir"] / "configs_resolved.jsonl"
            if p.exists():
                # _load_jsonl exists in this app; fallback safe if not.
                try:
                    rows = _load_jsonl(p)  # noqa: F821
                except Exception:
                    rows = []
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line=line.strip()
                            if not line: 
                                continue
                            import json as _json
                            rows.append(_json.loads(line))
                cfg_map = {str(r.get("config_id")): (r.get("normalized") or {}) for r in rows if r}
                cfg_norm = cfg_map.get(str(pick), {}) or {}
        ctx["cfg_norm"] = cfg_norm
        ctx["has_receipts"] = True  # receipts can still render without cfg_norm
    except Exception:
        ctx["cfg_norm"] = {}
        ctx["has_receipts"] = True

    # Artifact dir resolution: prefer replay_cache/<pick>, else top_map
    try:
        run_dir_p = ctx["run_dir"]
        if run_dir_p is not None:
            replay_dir = run_dir_p / "replay_cache" / str(pick)
            art_dir = replay_dir
            if top_map is not None:
                try:
                    art_dir = top_map.get(str(pick), replay_dir)
                except Exception:
                    art_dir = replay_dir
            # Prefer replay dir when it has core artifacts
            if (replay_dir / "equity_curve.csv").exists():
                art_dir = replay_dir
            ctx["art_dir"] = art_dir
            ctx["has_artifacts"] = bool(art_dir is not None and _Path(art_dir).exists())
    except Exception:
        ctx["art_dir"] = None
        ctx["has_artifacts"] = False

    # Load trades/equity if present (best source of truth for counts)
    try:
        art_dir = ctx["art_dir"]
        if art_dir is not None and _Path(art_dir).exists():
            tpath = _Path(art_dir) / "trades.csv"
            if tpath.exists():
                tdf = _pd.read_csv(tpath)
                ctx["trades_df"] = tdf
                ctx["trades_n"] = int(len(tdf))
            epath = _Path(art_dir) / "equity_curve.csv"
            if epath.exists():
                ctx["equity_df"] = _pd.read_csv(epath)
    except Exception:
        pass

    # Fallback trade count from row if trades.csv missing
    if ctx["trades_n"] is None:
        for k in ("trades", "n_trades", "trade_count", "trades_count"):
            try:
                v = ctx["row"].get(k)
                if v is not None:
                    ctx["trades_n"] = int(v)
                    break
            except Exception:
                continue
        if ctx["trades_n"] is None:
            ctx["trades_n"] = 0

    # RS/WF availability for this config (summary rows if present)
    try:
        ctx["has_rs"] = False
        ctx["has_wf"] = False
        ctx["rs_sum_row"] = None
        ctx["wf_sum_row"] = None

        pick_canon = _canon_cfg_id(pick)

        if rs_dir_effective:
            try:
                rs_sum = load_rs_summary(run_dir, rs_dir_effective)  # noqa: F821
            except Exception:
                rs_sum = None
            if rs_sum is not None and hasattr(rs_sum, "columns") and ("config_id" in rs_sum.columns):
                _sel = rs_sum[rs_sum["config_id"] == pick_canon]
                if _sel is not None and not _sel.empty:
                    ctx["rs_sum_row"] = dict(_sel.iloc[0])
                    ctx["has_rs"] = True

        if wf_dir_effective:
            try:
                wf_sum = load_wf_summary(wf_dir_effective)  # noqa: F821
            except Exception:
                wf_sum = None
            if wf_sum is not None and hasattr(wf_sum, "columns") and ("config_id" in wf_sum.columns):
                wf_sum = wf_sum.copy()
                wf_sum["config_id"] = wf_sum["config_id"].astype(str).map(_canon_cfg_id)
                _sel = wf_sum[wf_sum["config_id"] == pick_canon]
                if _sel is not None and not _sel.empty:
                    ctx["wf_sum_row"] = dict(_sel.iloc[0])
                    ctx["has_wf"] = True
                else:
                    # Fallback: wf_summary exists but does not include this config_id.
                    # Try wf_results.csv to decide whether the test ran for this config.
                    try:
                        wf_rows = load_wf_results(wf_dir_effective)  # noqa: F821
                        if wf_rows is not None and hasattr(wf_rows, "columns") and ("config_id" in wf_rows.columns):
                            wf_rows = wf_rows.copy()
                            wf_rows["config_id"] = wf_rows["config_id"].astype(str).map(_canon_cfg_id)
                            _selr = wf_rows[wf_rows["config_id"] == pick_canon]
                            if _selr is not None and not _selr.empty:
                                # Synthesize a minimal summary row from the per-window rows.
                                ret_c = _pick_col(_selr, ["window_return", "test_return", "return", "performance.window_return"])
                                if ret_c:
                                    arr = pd.to_numeric(_selr[ret_c], errors="coerce").dropna().to_numpy(dtype=float)
                                    if arr.size:
                                        p10 = float(np.quantile(arr, 0.10))
                                        p50 = float(np.quantile(arr, 0.50))
                                        p90 = float(np.quantile(arr, 0.90))
                                        ctx["wf_sum_row"] = {
                                            "return_p10": p10,
                                            "return_p50": p50,
                                            "return_p90": p90,
                                            "p10": p10,
                                            "p50": p50,
                                            "p90": p90,
                                            "neg_rate": float((arr < 0.0).mean()),
                                            "pct_profitable_windows": float((arr > 0.0).mean()),
                                        }
                                ctx["has_wf"] = True
                    except Exception:
                        pass
    except Exception:
        ctx["has_rs"] = False
        ctx["has_wf"] = False
        ctx["rs_sum_row"] = None
        ctx["wf_sum_row"] = None

    return ctx
def _summarize_grid_composition(grid_path: Path) -> Dict[str, Any]:
    """Summarize what kind of population we generated (for the run-story 'dopamine loop').

    This is NOT advice; it's just a factual breakdown of the variant set.
    """
    from datetime import datetime as _dt

    total = 0
    logic = 0
    always = 0
    filter_counts = defaultdict(int)

    trail = 0
    time_stop = 0
    sl = 0
    tp = 0

    # Extra: characterize logic complexity (bounded + readable)
    logic_clauses_sum = 0
    logic_conds_sum = 0

    if not grid_path.exists():
        return {"total": 0, "generated_at": _dt.utcnow().isoformat()}

    with open(grid_path, "r", encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            total += 1
            params = row.get("params") or {}
            bf = str(params.get("buy_filter", "none") or "none").strip().lower()
            el = params.get("entry_logic") if isinstance(params.get("entry_logic"), dict) else None
            clauses = []
            if isinstance(el, dict):
                clauses = el.get("clauses") or []
            # Heuristic:
            # - "logic-builder" = buy_filter is none AND entry_logic has at least one non-empty clause
            # - "always"       = buy_filter none AND entry_logic clauses empty
            # - otherwise      = simple buy_filter mode
            if bf in {"none", ""}:
                if clauses and any(isinstance(c, list) and len(c) > 0 for c in clauses):
                    logic += 1
                    logic_clauses_sum += int(len(clauses))
                    logic_conds_sum += int(sum(len(c) for c in clauses if isinstance(c, list)))
                else:
                    always += 1
            else:
                filter_counts[bf] += 1

            try:
                if float(params.get("trail_pct", 0.0) or 0.0) > 0.0:
                    trail += 1
            except Exception:
                pass
            try:
                if int(params.get("max_hold_bars", 0) or 0) > 0:
                    time_stop += 1
            except Exception:
                pass
            try:
                if float(params.get("sl_pct", 0.0) or 0.0) > 0.0:
                    sl += 1
            except Exception:
                pass
            try:
                if float(params.get("tp_pct", 0.0) or 0.0) > 0.0:
                    tp += 1
            except Exception:
                pass

    comp: Dict[str, Any] = {
        "generated_at": _dt.utcnow().isoformat(),
        "total": int(total),
        "entry": {
            "logic_builder": int(logic),
            "always": int(always),
            "simple_filters": {k: int(v) for k, v in sorted(filter_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
            "logic_avg_clauses": (logic_clauses_sum / logic) if logic else 0.0,
            "logic_avg_conditions": (logic_conds_sum / logic) if logic else 0.0,
        },
        "exits": {
            "trailing_stop_enabled": int(trail),
            "time_stop_enabled": int(time_stop),
            "stop_loss_enabled": int(sl),
            "take_profit_enabled": int(tp),
        },
    }
    return comp




def _render_grid_composition(comp: Dict[str, Any]) -> None:
    """Compact UI block: what did we generate? (collapsed by default)."""
    total = int(comp.get("total", 0) or 0)
    if total <= 0:
        st.info("No run composition available.")
        return

    entry = comp.get("entry") or {}
    exits = comp.get("exits") or {}

    def _pct(x: int) -> str:
        return f"{(100.0 * float(x) / float(total)):.0f}%"

    logic_n = int(entry.get("logic_builder", 0) or 0)
    always_n = int(entry.get("always", 0) or 0)

    trail_n = int(exits.get("trailing_stop_enabled", 0) or 0)
    time_n = int(exits.get("time_stop_enabled", 0) or 0)
    sl_n = int(exits.get("stop_loss_enabled", 0) or 0)
    tp_n = int(exits.get("take_profit_enabled", 0) or 0)

    with st.expander("Run composition (what we're testing)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total variants", f"{total:,}")
        c2.metric("Always-buy", _pct(always_n), f"{always_n:,}")
        c3.metric("Logic-builder", _pct(logic_n), f"{logic_n:,}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Trailing stop", _pct(trail_n), f"{trail_n:,}")
        c5.metric("Time stop", _pct(time_n), f"{time_n:,}")
        c6.metric("SL / TP enabled", f"{_pct(sl_n)} / {_pct(tp_n)}", f"{sl_n:,} / {tp_n:,}")

        sf = entry.get("simple_filters") or {}
        if isinstance(sf, dict) and len(sf) > 0:
            nice = {
                "below_ema": "Dip below EMA",
                "rsi_below": "RSI low",
                "bb_z_below": "BB z (oversold)",
                "macd_bull": "MACD bullish",
                "adx_above": "ADX trending",
                "donch_pos_below": "Donchian bottom",
            }
            parts = []
            for k, v in sf.items():
                try:
                    v = int(v)
                except Exception:
                    continue
                parts.append((str(k), v))
            parts = sorted(parts, key=lambda kv: (-kv[1], kv[0]))
            if parts:
                shown = []
                for k, v in parts[:4]:
                    shown.append(f"{nice.get(k, k)}: {v} ({_pct(v)})")
                extra = sum(v for _, v in parts[4:])
                if extra > 0:
                    shown.append(f"Other: {extra} ({_pct(extra)})")
                st.caption("Entry filters: " + " · ".join(shown))
def _tail_jsonl(path: Path, *, max_lines: int = 400) -> List[Dict[str, Any]]:
    """Read the last N JSONL rows (best-effort)."""
    if not path or (not path.exists()):
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        lines = lines[-int(max_lines) :]
        out: List[Dict[str, Any]] = []
        for s in lines:
            s = s.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out
    except Exception:
        return []





def _render_run_monitor(progress_path: Optional[Path]) -> None:
    """Render run progress from JSONL telemetry.

    Goal: calm, casual-user-friendly view.
    """

    if progress_path is None:
        st.info("Waiting for progress telemetry…")
        return

    paths: List[Path] = []
    if progress_path.is_dir():
        paths = sorted(progress_path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not paths:
            st.info("Waiting for progress telemetry…")
            return
    else:
        if not progress_path.exists():
            st.info("Waiting for progress telemetry…")
            return
        paths = [progress_path]

    events: List[Dict[str, Any]] = []
    for p in paths:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                obj.setdefault("_src", p.name)
                events.append(obj)

    if not events:
        st.info("Waiting for progress telemetry…")
        return

    def _t(e: Dict[str, Any]) -> float:
        try:
            return float(e.get("t"))
        except Exception:
            return float("nan")

    if any(isinstance(e.get("t"), (int, float)) for e in events):
        events = sorted(events, key=lambda e: (_t(e) if _t(e) == _t(e) else float("inf")))

    last = events[-1]
    df = pd.DataFrame(events)

    stage = str(last.get("stage", ""))
    _stage_key = stage.split(":")[0].strip().lower()
    _phase_key = str(last.get("phase", "")).split(":")[0].strip().lower()

    _stage_map = {
        "batch": "Batch sweep",
        "rolling_starts": "Rolling Starts",
        "walkforward": "Walkforward",
        "postprocess": "Postprocess",
        "grid": "Variants",
    }
    _phase_map = {
        "run": "running",
        "artifacts": "finalizing",
        "done": "done",
        "rerun": "rerun",
        "rank": "ranking",
        "sweep": "sweep",
    }

    stage_disp = _stage_map.get(_stage_key, stage or "(unknown)")
    phase_disp = _phase_map.get(_phase_key, _phase_key) if _phase_key else ""

    done = last.get("i", last.get("done", last.get("n_done", 0)))
    total = last.get("n", last.get("total", last.get("n_total", 0)))
    rate = last.get("rate", last.get("throughput"))

    def _to_int(x: Any) -> int:
        try:
            return int(float(x))
        except Exception:
            return 0

    def _to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return float("nan")

    done_i = _to_int(done)
    total_i = _to_int(total)
    rate_f = _to_float(rate)

    def _fmt_eta(seconds: float) -> str:
        try:
            s = int(max(0, round(float(seconds))))
        except Exception:
            return "—"
        if s < 60:
            return f"{s}s"
        m, s = divmod(s, 60)
        if m < 60:
            return f"{m}m {s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"

    eta = None
    if total_i > 0 and done_i >= 0 and rate_f == rate_f and rate_f > 0:
        rem = max(0, total_i - done_i)
        eta = _fmt_eta(rem / rate_f)

    # Human speed formatting (avoid twitchy decimals)
    speed_disp = "—"
    if rate_f == rate_f and rate_f > 0:
        if rate_f >= 10:
            speed_disp = f"{rate_f:.0f}/s"
        else:
            speed_disp = f"{rate_f:.1f}/s"

    # One-line status summary
    parts = [stage_disp]
    if phase_disp and phase_disp not in {"running", "sweep"}:
        parts.append(phase_disp)
    if total_i > 0:
        parts.append(f"{done_i:,}/{total_i:,}")
    if eta is not None:
        parts.append(f"ETA {eta}")
    if speed_disp != "—":
        parts.append(f"Speed {speed_disp}")

    # Stage explanation (calm)
    now_map = {
        "grid": "Generating your variation set (baseline ± drift).",
        "batch": "Sweeping variations and rejecting obvious junk early.",
        "postprocess": "Scoring, ranking, and saving the most useful artifacts.",
        "rolling_starts": "Re-testing survivors from many start dates to check timing fragility.",
        "walkforward": "Re-testing survivors across time windows to check period-to-period stability.",
    }
    now_text = now_map.get(_stage_key, "Running…")

    # Main status card
    with st.container():
        st.markdown("### Running stress test")
        st.caption(" · ".join(parts))

        # Progress bar
        if total_i > 0:
            st.progress(min(1.0, max(0.0, float(done_i) / float(total_i))))
        else:
            st.progress(0.0)

        st.markdown(f"**Now:** {now_text}")

        # Clean progress chart: variants tested over elapsed minutes
        progress_col = None
        for cand in ("i", "done", "n_done"):
            if cand in df.columns:
                progress_col = cand
                break

        if progress_col is not None and "t" in df.columns:
            tt = pd.to_numeric(df["t"], errors="coerce")
            pp = pd.to_numeric(df[progress_col], errors="coerce")
            mask = tt.notna() & pp.notna()
            if mask.sum() >= 3:
                t0 = float(tt[mask].min())
                mins = (tt[mask] - t0) / 60.0
                tmp = pd.DataFrame({"minutes": mins, "tested": pp[mask]}).sort_values("minutes")
                # De-dupe to keep charts calm
                tmp = tmp.drop_duplicates(subset=["minutes"], keep="last")
                tmp = tmp.set_index("minutes")
                st.line_chart(tmp["tested"], height=160)

        # Reject reasons (collapsed by default)
        fails = last.get("fails")
        if not (isinstance(fails, dict) and fails):
            ft = last.get("fail_top")
            if isinstance(ft, list) and ft:
                try:
                    fails = {str(k): int(v) for k, v in ft}
                except Exception:
                    fails = None

        def _nice_reason(k: str) -> str:
            k = str(k)
            k = k.replace("_", " ")
            k = k.replace("pct", "%")
            k = k.replace("best trade", "one-trade")
            k = k.replace("domin", "dominance")
            k = k.strip()
            return k[:28] + ("…" if len(k) > 28 else "")

        if isinstance(fails, dict) and fails:
            s = pd.Series({str(k): int(v) for k, v in fails.items()}).sort_values(ascending=False)
            if len(s) > 0:
                top = s.head(3)
                rest = int(s.iloc[3:].sum()) if len(s) > 3 else 0
                if rest > 0:
                    top["Other"] = rest
                top.index = [_nice_reason(x) for x in top.index]

                with st.expander("Reject reasons (so far)", expanded=False):
                    st.bar_chart(top, height=180)

        # Advanced telemetry (optional)
        with st.expander("Telemetry (advanced)", expanded=False):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Stage", stage_disp)
            with cols[1]:
                if total_i > 0:
                    st.metric("Progress", f"{done_i:,}/{total_i:,}")
                else:
                    st.metric("Progress", f"{done_i:,}")
            with cols[2]:
                st.metric("Throughput", speed_disp)

            if "t" in df.columns and "rate" in df.columns:
                rr = pd.to_numeric(df["rate"], errors="coerce")
                tt = pd.to_numeric(df["t"], errors="coerce")
                mask = rr.notna() & tt.notna()
                if mask.sum() >= 3:
                    t0 = float(tt[mask].min())
                    mins = (tt[mask] - t0) / 60.0
                    tmp = pd.DataFrame({"minutes": mins, "rate": rr[mask]}).sort_values("minutes")
                    tmp = tmp.drop_duplicates(subset=["minutes"], keep="last").set_index("minutes")
                    st.line_chart(tmp["rate"], height=140)

def _tail_text(lines: Iterable[str], n: int = 40) -> str:
    xs = list(lines)[-max(0, int(n)) :]
    return "".join(xs)


def _run_cmd(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    progress_path: Optional[Path] = None,
    refresh_hz: float = 4.0,
) -> None:
    """Run a command and stream output + telemetry into the UI.

    Sprint 3 polish:
    - Logs are hidden by default (toggle Debug in the sidebar)
    - On failure, show a short summary + last output lines
    """
    if not cmd:
        raise ValueError("Empty command")

    # Make "python" consistent across platforms
    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    debug = bool(st.session_state.get("ui.debug", False))

    with st.expander(label, expanded=True):
        details = st.expander("Details & troubleshooting", expanded=debug)
        with details:
            st.code(" ".join(cmd), language="bash")
            if not debug:
                st.caption("Debug is off — raw logs are hidden. Toggle Debug in the sidebar to view streaming output.")
            log_ph = st.empty()

        mon_ph = st.empty()

        # Spawn process
        t0 = time.time()
        p = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        q: "queue.Queue[str]" = queue.Queue()

        def _reader():
            try:
                assert p.stdout is not None
                for line in p.stdout:
                    q.put(line)
            except Exception:
                pass

        th = threading.Thread(target=_reader, daemon=True)
        th.start()

        lines = deque(maxlen=800)
        sleep_s = max(0.05, 1.0 / float(max(1.0, refresh_hz)))

        while p.poll() is None:
            # Drain output queue
            for _ in range(400):
                try:
                    lines.append(q.get_nowait())
                except Exception:
                    break

            with mon_ph.container():
                _render_run_monitor(progress_path)

            if debug and lines:
                log_ph.code("".join(list(lines)[-140:]), language="text")

            time.sleep(sleep_s)

        # Final drain
        for _ in range(8000):
            try:
                lines.append(q.get_nowait())
            except Exception:
                break

        dt = time.time() - t0
        rc = int(p.returncode or 0)

        # Always show logs on failure (even if debug is off)
        if rc != 0:
            tail = _tail_text(lines, n=60)
            with details:
                st.error(f"Failed (code={rc}) after {dt:.1f}s")
                st.code(tail or "(no output captured)", language="text")
            raise RuntimeError(f"{label} failed (code={rc}) after {dt:.1f}s")

        # On success, only show full logs in debug mode
        if debug and lines:
            with details:
                st.caption(f"Completed in {dt:.1f}s")
                st.code("".join(lines), language="text")


def _render_replay_artifacts_controls(
    *,
    run_dir: Path,
    pick: str,
    replay_dir: Path,
    has_core_artifacts: bool,
    can_replay: bool,
    key_prefix: str = "replay.controls",
    show_when_ready: bool = True,
) -> None:
    """Render a single, canonical replay-artifacts control surface.

    Why this exists:
    - Users were seeing multiple buttons that effectively do the same thing (generate/regen artifacts).
    - This helper unifies the behavior (same command, same options, same progress handling).
    - Other sections can *reference* this control rather than duplicating buttons.

    Behavior:
    - If artifacts are missing: show an info callout + primary button.
    - If artifacts exist: optionally show a small expander with a regen button + refresh-cache toggle.
    """
    replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
    if not replay_script.exists():
        st.warning("Replay artifact generator script missing (tools/generate_replay_artifacts.py).")
        return

    def _defaults_from_meta() -> Tuple[float, int]:
        try:
            meta = _read_json(run_dir / "batch_meta.json")
        except Exception:
            meta = {}
        try:
            starting_equity = float((meta or {}).get("starting_equity", 10000.0) or 10000.0)
        except Exception:
            starting_equity = 10000.0
        try:
            seed = int((meta or {}).get("seed", 1) or 1)
        except Exception:
            seed = 1
        return starting_equity, seed

    starting_equity, seed = _defaults_from_meta()

    # If the user requests a refresh, we delete the replay_dir and rebuild from scratch.
    # This is useful when new artifact types (e.g., events.csv) were added after the original run.
    def _run(refresh_cache: bool) -> None:
        if refresh_cache:
            try:
                if replay_dir.exists():
                    shutil.rmtree(replay_dir)
            except Exception:
                pass

        progress_path = replay_dir / "progress.jsonl"
        cmd = [
            PY,
            str(replay_script),
            "--from-run",
            str(run_dir),
            "--config-id",
            str(pick),
            "--progress-file",
            str(progress_path),
            "--starting-equity",
            str(starting_equity),
            "--seed",
            str(seed),
        ]
        _run_cmd(cmd, cwd=REPO_ROOT, label="Replay: generate artifacts", progress_path=progress_path)
        st.rerun()

    # Missing artifacts: show primary CTA immediately.
    if not has_core_artifacts:
        st.info(
            "Replay artifacts for this candidate haven't been generated yet. "
            "Generate them to unlock the price + event timeline and detailed receipts."
        )
        if st.button(
            "Generate replay artifacts",
            key=f"{key_prefix}.gen.{pick}",
            disabled=(not can_replay),
            type="primary",
        ):
            _run(refresh_cache=False)
        return

    # Artifacts exist: keep controls available but tucked away.
    if show_when_ready:
        with st.expander("Replay artifacts", expanded=False):
            st.caption("Use this if artifacts look incomplete or you added new artifact types (e.g., events.csv).")
            refresh_cache = st.checkbox(
                "Refresh cache (rebuild from scratch)",
                value=False,
                key=f"{key_prefix}.refresh.{pick}",
            )
            if st.button(
                "Regenerate replay artifacts",
                key=f"{key_prefix}.regen.{pick}",
                disabled=(not can_replay),
            ):
                _run(refresh_cache=bool(refresh_cache))



def _run_subprocess_stream(
    cmd: List[str],
    *,
    cwd: Path,
    label: str,
    ui_ph: "st.delta_generator.DeltaGenerator",
    progress_path: Optional[Path] = None,
    refresh_hz: float = 4.0,
    debug: Optional[bool] = None,
) -> Tuple[int, float, str]:
    """Run a command and stream progress/logs into a *single* UI panel.

    Returns: (returncode, seconds, tail_text)
    """
    if not cmd:
        raise ValueError("Empty command")

    if str(cmd[0]).lower() in {"python", "py", "py.exe", "python3"}:
        cmd = [PY, *cmd[1:]]

    if debug is None:
        debug = bool(st.session_state.get("ui.debug", False))

    with ui_ph.container():
        st.caption(f"Stage: {label}")
        mon_ph = st.empty()
        details = st.expander("Details & troubleshooting", expanded=bool(debug))
        with details:
            st.code(" ".join(cmd), language="bash")
            if not debug:
                st.caption("Debug is off — streaming logs are hidden. Toggle Debug in the sidebar to view them.")
            log_ph = st.empty()

    # Spawn process
    t0 = time.time()
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    q: "queue.Queue[str]" = queue.Queue()

    def _reader():
        try:
            assert p.stdout is not None
            for line in p.stdout:
                q.put(line)
        except Exception:
            pass

    th = threading.Thread(target=_reader, daemon=True)
    th.start()

    lines = deque(maxlen=800)
    sleep_s = max(0.05, 1.0 / float(max(1.0, refresh_hz)))

    while p.poll() is None:
        for _ in range(400):
            try:
                lines.append(q.get_nowait())
            except Exception:
                break

        with mon_ph.container():
            if progress_path is not None:
                _render_run_monitor(progress_path)
            else:
                st.info("Running…")

        if bool(debug) and lines:
            with details:
                log_ph.code("".join(list(lines)[-140:]), language="text")

        time.sleep(sleep_s)

    # Final drain
    for _ in range(8000):
        try:
            lines.append(q.get_nowait())
        except Exception:
            break

    dt = time.time() - t0
    rc = int(p.returncode or 0)
    tail = _tail_text(lines, n=80)

    if rc != 0:
        with ui_ph.container():
            st.error(f"{label} failed (code={rc}) after {dt:.1f}s")
            with details:
                st.code(tail or "(no output captured)", language="text")

    return rc, dt, tail


@dataclass
class _PipelineStage:
    key: str
    label: str



class _PipelineUI:
    """A tiny 'lab monitor' that runs stages sequentially and keeps the UI clean.

    - One stepper strip
    - One active-stage panel (only one stage expanded at a time)
    - Logs hidden by default (Debug toggle)
    """

    def __init__(self, stages: List[_PipelineStage], *, debug: Optional[bool] = None):
        self.stages = stages
        self.debug = bool(st.session_state.get("ui.debug", False)) if debug is None else bool(debug)
        self.status: Dict[str, str] = {s.key: "pending" for s in stages}  # pending|running|done|fail
        self.durations: Dict[str, float] = {}
        self.stepper_ph = st.empty()
        self.active_ph = st.empty()
        self.render()

    def _icon(self, stt: str) -> str:
        return {
            "pending": "⬜",
            "running": "⏳",
            "done": "✅",
            "fail": "❌",
        }.get(stt, "⬜")

    def render(self) -> None:
        """Pill-style pipeline status strip."""
        with self.stepper_ph.container():
            cols = st.columns(len(self.stages))

            styles = {
                "pending": "background:#f3f4f6; border:1px solid #e5e7eb; color:#374151;",
                "running": "background:#dbeafe; border:1px solid #bfdbfe; color:#1d4ed8;",
                "done": "background:#dcfce7; border:1px solid #bbf7d0; color:#166534;",
                "fail": "background:#fee2e2; border:1px solid #fecaca; color:#991b1b;",
            }

            for col, s in zip(cols, self.stages):
                stt = self.status.get(s.key, "pending")
                style = styles.get(stt, styles["pending"])
                html = (
                    f"<div style='display:inline-block; padding:6px 10px; border-radius:999px; "
                    f"font-weight:600; font-size:14px; {style}'>"
                    f"{s.label}</div>"
                )
                col.markdown(html, unsafe_allow_html=True)
                dur = self.durations.get(s.key)
                if dur is not None:
                    col.caption(f"{dur:.1f}s")

    def run(self, key: str, cmd: List[str], *, cwd: Path, progress_path: Optional[Path] = None) -> None:
        label = next((s.label for s in self.stages if s.key == key), key)
        self.status[key] = "running"
        self.render()

        rc, dt, tail = _run_subprocess_stream(
            cmd,
            cwd=cwd,
            label=label,
            ui_ph=self.active_ph,
            progress_path=progress_path,
            debug=self.debug,
        )
        self.durations[key] = float(dt)

        if rc != 0:
            self.status[key] = "fail"
            self.render()
            raise RuntimeError(f"{label} failed (code={rc})")

        self.status[key] = "done"
        self.render()


def _list_runs() -> List[Path]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    xs = [p for p in RUNS_DIR.glob("batch_*") if p.is_dir()]
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)

def _has_any_results(run_dir: Path) -> bool:
    """True if the run folder contains any batch result CSV."""
    for fn in (
        "results_full_passed.csv",
        "results_passed.csv",
        "results_full.csv",
        "results.csv",
    ):
        if (run_dir / fn).exists():
            return True
    return False



def _pick_latest_dir(base: Path, glob_pat: str) -> Optional[Path]:
    if not base.exists():
        return None
    xs = [p for p in base.glob(glob_pat) if p.is_dir()]
    if not xs:
        return None
    return sorted(xs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _pick_latest_rs_dir(run_dir: Path) -> Optional[Path]:
    """Return the most appropriate Rolling Starts output folder for this run.

    Supports:
      - CLI default: {run_dir}/rolling_starts/{rolling_starts_summary.csv,...}
      - UI outputs: {run_dir}/rolling_starts/rs_*/...
    If multiple candidates exist, picks the most recently modified directory.
    """
    rs_root = run_dir / "rolling_starts"
    if not rs_root.exists():
        return None

    cands: List[Path] = []
    if (rs_root / "rolling_starts_summary.csv").exists() or (rs_root / "rolling_starts_detail.csv").exists():
        cands.append(rs_root)

    for d in rs_root.glob("rs_*"):
        if d.is_dir() and (d / "rolling_starts_summary.csv").exists():
            cands.append(d)

    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _pick_latest_wf_dir(run_dir: Path) -> Optional[Path]:
    """Return the most appropriate Walkforward output folder for this run.

    Supports:
      - CLI default: {run_dir}/walkforward_*/{wf_summary.csv,...}
      - UI outputs: {run_dir}/walkforward/wf_*/{wf_summary.csv,...}
      - (Legacy) direct files in {run_dir}/walkforward/{wf_summary.csv,...}
    If multiple candidates exist, picks the most recently modified directory.
    """
    cands: List[Path] = []
    # CLI default folders at run root
    for p in run_dir.glob("walkforward_*"):
        if p.is_dir() and (p / "wf_summary.csv").exists():
            cands.append(p)

    wf_root = run_dir / "walkforward"
    if wf_root.exists():
        # Legacy direct files
        if (wf_root / "wf_summary.csv").exists() or (wf_root / "wf_results.csv").exists():
            cands.append(wf_root)
        # UI subfolders
        for d in wf_root.glob("wf_*"):
            if d.is_dir() and (d / "wf_summary.csv").exists():
                cands.append(d)

    if not cands:
        return None
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _parse_top_artifact_dirs(run_dir: Path) -> Dict[str, Path]:
    """
    Map config_id -> top artifact dir, when present.
    Folder format: top/0001_<config_id>_<label>/
    """
    top_dir = run_dir / "top"
    out: Dict[str, Path] = {}
    if not top_dir.exists():
        return out
    for d in top_dir.iterdir():
        if not d.is_dir():
            continue
        parts = d.name.split("_", 2)
        if len(parts) < 2:
            continue
        cid = parts[1]
        out[str(cid)] = d
    return out


# =============================================================================
# "Questions" (stage filters)
# =============================================================================

_SEVERITY_ORDER = {"info": 0, "warn": 1, "warning": 1, "critical": 2, "crit": 2}


@dataclass(frozen=True)
class ConstraintSpec:
    metric_id: str
    op: str  # ">=" or "<="
    threshold: float
    severity: str  # "info" | "warn" | "critical"
    note: str = ""


@dataclass(frozen=True)
class ChoiceSpec:
    label: str
    constraints: List[ConstraintSpec]


@dataclass(frozen=True)
class QuestionSpec:
    id: str
    title: str
    explanation: str
    choices: List[ChoiceSpec]
    default_index: int = 0


@dataclass
class EvalOutcome:
    verdict: str  # PASS | WARN | FAIL
    crits: int
    warns: int
    infos: int
    missing: int
    violations: List[Dict[str, Any]]
    missing_metrics: List[str]


def _to_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def _passes(op: str, value: float, thr: float) -> bool:
    if op == ">=":
        return value >= thr
    if op == "<=":
        return value <= thr
    return True


def evaluate_row_with_questions(
    row: Dict[str, Any],
    questions: List[QuestionSpec],
    answers: Dict[str, int],
) -> EvalOutcome:
    violations: List[Dict[str, Any]] = []
    missing_metrics: List[str] = []

    crits = warns = infos = 0
    missing = 0

    for q in questions:
        pick = int(answers.get(q.id, q.default_index))
        pick = max(0, min(pick, len(q.choices) - 1))
        choice = q.choices[pick]

        for c in choice.constraints:
            v = _to_float(row.get(c.metric_id, float("nan")))
            if v != v:  # NaN
                missing += 1
                missing_metrics.append(c.metric_id)
                continue

            if _passes(c.op, v, float(c.threshold)):
                continue

            sev = str(c.severity).strip().lower()
            sev_rank = _SEVERITY_ORDER.get(sev, 0)

            if sev_rank >= 2:
                crits += 1
            elif sev_rank == 1:
                warns += 1
            else:
                infos += 1

            violations.append(
                {
                    "question_id": q.id,
                    "question": q.title,
                    "metric_id": c.metric_id,
                    "metric": _metric_label(c.metric_id),
                    "value": v,
                    "op": c.op,
                    "threshold": float(c.threshold),
                    "severity": sev,
                    "message": f"{_metric_label(c.metric_id)} is {_metric_fmt(c.metric_id, v)} but your limit is {c.op} {_metric_fmt(c.metric_id, float(c.threshold))}.",
                    "note": c.note,
                }
            )

    verdict = "PASS"
    if crits > 0:
        verdict = "FAIL"
    elif warns > 0:
        verdict = "WARN"

    # Make missing list stable unique
    missing_metrics2 = sorted({str(x) for x in missing_metrics})

    return EvalOutcome(
        verdict=verdict,
        crits=int(crits),
        warns=int(warns),
        infos=int(infos),
        missing=int(missing),
        violations=violations,
        missing_metrics=missing_metrics2,
    )


def apply_stage_eval(
    df: pd.DataFrame,
    *,
    stage_key: str,
    questions: List[QuestionSpec],
    answers: Dict[str, int],
) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    rows = df.to_dict(orient="records")
    verdicts: List[str] = []
    crits: List[int] = []
    warns: List[int] = []
    missing: List[int] = []

    for r in rows:
        out = evaluate_row_with_questions(r, questions, answers)
        verdicts.append(out.verdict)
        crits.append(out.crits)
        warns.append(out.warns)
        missing.append(out.missing)

    out_df = df.copy()
    out_df[f"{stage_key}.verdict"] = verdicts
    out_df[f"{stage_key}.crits"] = crits
    out_df[f"{stage_key}.warns"] = warns
    out_df[f"{stage_key}.missing"] = missing
    return out_df


def _question_ui(questions: List[QuestionSpec], *, key_prefix: str) -> Dict[str, int]:
    answers: Dict[str, int] = {}
    for q in questions:
        opts = [c.label for c in q.choices]
        idx = st.radio(
            q.title,
            options=list(range(len(opts))),
            format_func=lambda i: opts[int(i)],
            index=int(q.default_index),
            key=f"{key_prefix}.{q.id}",
        )
        st.caption(q.explanation)
        answers[q.id] = int(idx)
    return answers


def batch_questions() -> List[QuestionSpec]:
    return [
        QuestionSpec(
            id="batch_drawdown",
            title="How big of a drop from a previous high can you tolerate?",
            explanation="Single-run max drawdown on equity curve.",
            choices=[
                ChoiceSpec("Max 20% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.20, "critical")]),
                ChoiceSpec("Max 35% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.35, "warn")]),
                ChoiceSpec("Max 50% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.50, "warn")]),
                ChoiceSpec("Max 70% drop", [ConstraintSpec("performance.max_drawdown_equity", "<=", 0.70, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="batch_profit",
            title="Do you require net profit (excluding deposits) to be positive?",
            explanation="If deposits are part of the plan, this isolates whether the strategy actually made money beyond what you put in.",
            choices=[
                ChoiceSpec("Yes (must be ≥ $0)", [ConstraintSpec("equity.net_profit_ex_cashflows", ">=", 0.0, "warn")]),
                ChoiceSpec("No (let losers through)", []),
            ],
            default_index=0,
        ),
        QuestionSpec(
            id="batch_fees",
            title="How sensitive are you to fees and churn?",
            explanation="Higher turnover usually means more slippage/fees in real life.",
            choices=[
                ChoiceSpec(
                    "Very fee-sensitive",
                    [
                        ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 0.5, "warn"),
                        ConstraintSpec("efficiency.fee_impact_pct", "<=", 10.0, "warn"),
                    ],
                ),
                ChoiceSpec(
                    "Moderately fee-sensitive",
                    [
                        ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 1.5, "warn"),
                        ConstraintSpec("efficiency.fee_impact_pct", "<=", 25.0, "warn"),
                    ],
                ),
                ChoiceSpec("Not very fee-sensitive", [ConstraintSpec("exposure.turnover_notional_over_avg_equity", "<=", 3.0, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
    ]


def rolling_questions() -> List[QuestionSpec]:
    return [
        QuestionSpec(
            id="rs_worst_return",
            title="If you started at a bad time, how much loss can you tolerate?",
            explanation="We simulate many start dates. This checks the worst 10% outcome (p10) time-weighted return.",
            choices=[
                ChoiceSpec("Worst 10% must not lose money", [ConstraintSpec("twr_p10", ">=", 0.0, "critical")]),
                ChoiceSpec("I can tolerate up to -10%", [ConstraintSpec("twr_p10", ">=", -0.10, "warn")]),
                ChoiceSpec("I can tolerate up to -25%", [ConstraintSpec("twr_p10", ">=", -0.25, "warn")]),
                ChoiceSpec("I can tolerate up to -50%", [ConstraintSpec("twr_p10", ">=", -0.50, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="rs_drawdown",
            title="In a bad-but-common scenario, how deep a drawdown can you tolerate?",
            explanation="This uses the p90 drawdown across rolling starts.",
            choices=[
                ChoiceSpec("Max 20% drop", [ConstraintSpec("dd_p90", "<=", 0.20, "critical")]),
                ChoiceSpec("Max 35% drop", [ConstraintSpec("dd_p90", "<=", 0.35, "warn")]),
                ChoiceSpec("Max 50% drop", [ConstraintSpec("dd_p90", "<=", 0.50, "warn")]),
                ChoiceSpec("Max 70% drop", [ConstraintSpec("dd_p90", "<=", 0.70, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="rs_underwater",
            title="How long can you tolerate being underwater (below a prior high)?",
            explanation="This uses p90 underwater duration (days) across rolling starts.",
            choices=[
                ChoiceSpec("About 1 month", [ConstraintSpec("uw_p90_days", "<=", 30.0, "critical")]),
                ChoiceSpec("About 3 months", [ConstraintSpec("uw_p90_days", "<=", 90.0, "warn")]),
                ChoiceSpec("About 6 months", [ConstraintSpec("uw_p90_days", "<=", 180.0, "warn")]),
                ChoiceSpec("About 1 year", [ConstraintSpec("uw_p90_days", "<=", 365.0, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=2,
        ),
        QuestionSpec(
            id="rs_util",
            title="Do you want this plan invested most of the time?",
            explanation="Median invested fraction across rolling starts.",
            choices=[
                ChoiceSpec("Mostly invested", [ConstraintSpec("util_p50", ">=", 0.75, "warn")]),
                ChoiceSpec("Balanced", [ConstraintSpec("util_p50", ">=", 0.50, "info")]),
                ChoiceSpec("Mostly in cash is fine", [ConstraintSpec("util_p50", ">=", 0.25, "info")]),
                ChoiceSpec("I don't care", []),
            ],
            default_index=1,
        ),
    ]


def walkforward_questions() -> List[QuestionSpec]:
    # Walkforward metrics are produced by engine.walkforward -> wf_summary.csv
    #
    # Philosophy: WF exists to punish "in-sample optimism".
    # Defaults below are intentionally mild (WARN/INFO) so users can explore,
    # but the questions steer attention toward worst-case and stability.
    return [
        QuestionSpec(
            id="wf_typical",
            title="Do you require typical walk-forward performance to be positive?",
            explanation="Median (p50) return across walk-forward windows.",
            choices=[
                ChoiceSpec("Yes (p50 ≥ 0)", [ConstraintSpec("return_p50", ">=", 0.0, "warn")]),
                ChoiceSpec("No", []),
            ],
            default_index=0,
        ),
        QuestionSpec(
            id="wf_worst_typical",
            title="How negative can the 'worst typical' window be?",
            explanation="10th percentile (p10) return across windows. This is a good 'stability' anchor.",
            choices=[
                ChoiceSpec("p10 ≥ -5%", [ConstraintSpec("return_p10", ">=", -0.05, "warn")]),
                ChoiceSpec("p10 ≥ -10%", [ConstraintSpec("return_p10", ">=", -0.10, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_min",
            title="How bad can the single worst window be?",
            explanation="Minimum window return across all walk-forward windows (the absolute faceplant).",
            choices=[
                ChoiceSpec("Worst ≥ -10%", [ConstraintSpec("min_window_return", ">=", -0.10, "warn")]),
                ChoiceSpec("Worst ≥ -25%", [ConstraintSpec("min_window_return", ">=", -0.25, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_dd",
            title="How much drawdown pain can you tolerate in walk-forward?",
            explanation="90th percentile max drawdown (dd_p90) across windows. Lower is better.",
            choices=[
                ChoiceSpec("dd_p90 ≤ 20%", [ConstraintSpec("dd_p90", "<=", 0.20, "warn")]),
                ChoiceSpec("dd_p90 ≤ 35%", [ConstraintSpec("dd_p90", "<=", 0.35, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_consistency",
            title="How consistent should it be across windows?",
            explanation="Percent of windows with positive return.",
            choices=[
                ChoiceSpec("≥ 60% profitable windows", [ConstraintSpec("pct_profitable_windows", ">=", 0.60, "warn")]),
                ChoiceSpec("≥ 50% profitable windows", [ConstraintSpec("pct_profitable_windows", ">=", 0.50, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
        QuestionSpec(
            id="wf_trading",
            title="Should it actually trade in most windows?",
            explanation="Percent of windows with at least 1 trade. Avoids strategies that only 'wake up' rarely.",
            choices=[
                ChoiceSpec("≥ 80% windows traded", [ConstraintSpec("pct_windows_traded", ">=", 0.80, "warn")]),
                ChoiceSpec("≥ 60% windows traded", [ConstraintSpec("pct_windows_traded", ">=", 0.60, "info")]),
                ChoiceSpec("Don't filter on this", []),
            ],
            default_index=1,
        ),
    ]


# =============================================================================
# Run data loaders / mergers
# =============================================================================


_CFG_HEX_RE = re.compile(r"^[0-9a-f]{8,}$", re.IGNORECASE)

def _canon_cfg_id(x: Any) -> str:
    """Canonicalize config IDs across stages.

    Some legacy artifacts omit the 'cfg_' prefix (or mix cases). We normalize to:
      - '' for null/NaN
      - preserve digit-only ids (line numbers)
      - 'cfg_' + lower(hex) for hex-like ids without the prefix
      - otherwise keep the original string (trimmed)
    """
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    if s.isdigit():
        return s
    if s.startswith("cfg_"):
        return s
    if _CFG_HEX_RE.fullmatch(s):
        return "cfg_" + s.lower()
    return s

def _ensure_config_id(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee a canonical string config_id column.

    We strongly prefer the flattened column name from the batch runner (config.id),
    because some artifacts may include a legacy/config_id column that is not stable
    across stages. This keeps RS/WF joins and evidence lookups consistent.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    if "config.id" in out.columns:
        out["config_id"] = out["config.id"].astype(str).str.strip()
    elif "config_id" in out.columns:
        out["config_id"] = out["config_id"].astype(str).str.strip()
    else:
        # last-resort fallbacks (rare)
        for alt in ["config.id", "config_id", "id", "cfg_id"]:
            if alt in out.columns:
                out["config_id"] = out[alt].astype(str).str.strip()
                break

    if "config.label" in out.columns:
        out["config.label"] = out["config.label"].astype(str)
    if "config_id" in out.columns:
        out["config_id"] = out["config_id"].map(_canon_cfg_id)
    return out



def load_batch_frames(run_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Returns:
      - sweep_all: results.csv
      - sweep_passed: results_passed.csv
      - full_all: results_full.csv
      - full_passed: results_full_passed.csv
      - ranked: post/ranked.csv
    """
    frames: Dict[str, Optional[pd.DataFrame]] = {}
    frames["sweep_all"] = _load_csv(run_dir / "results.csv")
    frames["sweep_passed"] = _load_csv(run_dir / "results_passed.csv")
    frames["full_all"] = _load_csv(run_dir / "results_full.csv")
    frames["full_passed"] = _load_csv(run_dir / "results_full_passed.csv")
    frames["ranked"] = _load_csv(run_dir / "post" / "ranked.csv")

    for k, v in list(frames.items()):
        if v is not None:
            frames[k] = _ensure_config_id(v)

    return frames


def pick_survivors(frames: Dict[str, Optional[pd.DataFrame]]) -> Tuple[pd.DataFrame, str]:
    """
    Pick the 'survivor' set we treat as the batch output.
    Preference order:
      1) full_passed (rerun-passed configs) if exists and non-empty
      2) sweep_passed (sweep-passed configs)
      3) full_all
      4) sweep_all
    """
    for key in ["full_passed", "sweep_passed", "full_all", "sweep_all"]:
        df = frames.get(key)
        if df is not None and not df.empty:
            return df.copy(), key
    return pd.DataFrame([]), "none"


def load_rs_summary(run_dir: Path, rs_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if rs_dir is None:
        return None
    p = rs_dir / "rolling_starts_summary.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip().map(_canon_cfg_id)
    return df


def load_rs_detail(run_dir: Path, rs_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if rs_dir is None:
        return None
    p = rs_dir / "rolling_starts_detail.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip().map(_canon_cfg_id)
    return df


def load_wf_summary(wf_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if wf_dir is None:
        return None
    p = wf_dir / "wf_summary.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip().map(_canon_cfg_id)
    # normalize pct column name for our questions (keep original too)
    if "pct_profitable_windows" in df.columns:
        df["pct_profitable_windows"] = pd.to_numeric(df["pct_profitable_windows"], errors="coerce")
    return df


def load_wf_results(wf_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    if wf_dir is None:
        return None
    p = wf_dir / "wf_results.csv"
    df = _load_csv(p)
    if df is None or df.empty:
        return df
    df = df.copy()
    if "config_id" in df.columns:
        df["config_id"] = df["config_id"].astype(str).str.strip().map(_canon_cfg_id)
    return df


def merge_stage(
    base: pd.DataFrame,
    add: Optional[pd.DataFrame],
    *,
    on: str = "config_id",
    suffix: str = "",
) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    if add is None or add.empty or on not in add.columns:
        out = base.copy()
        out[f"{suffix}.measured" if suffix else "measured"] = False
        return out

    out = base.merge(add, how="left", on=on, suffixes=("", f".{suffix}"))
    out[f"{suffix}.measured" if suffix else "measured"] = out[on].isin(add[on].astype(str))
    return out


# =============================================================================
# New run wizard: DCA/Swing baseline builder
# =============================================================================

def _filter_true_pct(
    df_feat: pd.DataFrame,
    buy_filter: str,
    *,
    ema_len: int = 200,
    rsi_thr: float = 40.0,
    macd_hist_thr: float = 0.0,
    bb_z_thr: float = -1.0,
    adx_thr: float = 20.0,
    donch_pos_thr: float = 0.20,
) -> Optional[float]:
    """Percent of bars where the entry filter is true (for Build-step context)."""
    if df_feat is None or df_feat.empty:
        return None
    f = str(buy_filter or "none").strip().lower()

    try:
        close = df_feat["close"].astype(float)
    except Exception:
        return None

    mask = None
    if f in {"none", ""}:
        mask = pd.Series(True, index=df_feat.index)
    elif f == "below_ema":
        col = f"ema_{int(ema_len)}"
        if col in df_feat.columns:
            mask = close <= pd.to_numeric(df_feat[col], errors="coerce")
    elif f == "rsi_below":
        col = "rsi_14"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(rsi_thr)
    elif f == "macd_bull":
        col = "macd_hist_12_26_9"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") >= float(macd_hist_thr)
    elif f == "bb_z_below":
        col = "bb_z_20"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(bb_z_thr)
    elif f == "adx_above":
        col = "adx_14"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") >= float(adx_thr)
    elif f == "donch_pos_below":
        col = "donch_pos_20"
        if col in df_feat.columns:
            mask = pd.to_numeric(df_feat[col], errors="coerce") <= float(donch_pos_thr)

    if mask is None:
        return None
    try:
        return float(mask.mean() * 100.0)
    except Exception:
        return None



def _simple_filter_to_entry_logic(
    buy_filter: str,
    *,
    ema_len: int = 200,
    rsi_thr: float = 40.0,
    macd_hist_thr: float = 0.0,
    bb_z_thr: float = -1.0,
    adx_thr: float = 20.0,
    donch_pos_thr: float = 0.20,
) -> Dict[str, Any]:
    """
    Translate legacy single buy_filter into entry_logic (regime=[], clauses=[...]).
    entry_logic condition schema matches dca_swing_strategy_overhaul_v1.py:
      - indicator (lhs)
      - operator
      - threshold (rhs literal OR offset when ref_indicator used)
      - ref_indicator (optional rhs indicator)
    """
    f = str(buy_filter or "none").strip().lower()
    regime: List[Dict[str, Any]] = []
    clauses: List[List[Dict[str, Any]]] = []

    if f in {"none", ""}:
        return {"regime": regime, "clauses": clauses}  # no triggers => always allowed

    if f == "below_ema":
        clauses = [[{"indicator": "close", "operator": "<=", "ref_indicator": f"ema_{int(ema_len)}", "threshold": 0.0}]]
    elif f == "rsi_below":
        clauses = [[{"indicator": "rsi_14", "operator": "<=", "threshold": float(rsi_thr)}]]
    elif f == "bb_z_below":
        clauses = [[{"indicator": "bb_z_20", "operator": "<=", "threshold": float(bb_z_thr)}]]
    elif f == "macd_bull":
        clauses = [[{"indicator": "macd_hist_12_26_9", "operator": ">=", "threshold": float(macd_hist_thr)}]]
    elif f == "adx_above":
        clauses = [[{"indicator": "adx_14", "operator": ">=", "threshold": float(adx_thr)}]]
    elif f == "donch_pos_below":
        clauses = [[{"indicator": "donch_pos_20", "operator": "<=", "threshold": float(donch_pos_thr)}]]

    return {"regime": regime, "clauses": clauses}


def _cond_mask(df_feat: pd.DataFrame, cond: Dict[str, Any]) -> Optional[pd.Series]:
    """Vectorized mask for one condition against df_feat. Returns None if missing columns."""
    if df_feat is None or df_feat.empty or not isinstance(cond, dict):
        return None

    ind = cond.get("indicator") or cond.get("feature") or cond.get("lhs")
    op = str(cond.get("operator") or cond.get("op") or "").strip()
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator")

    if op not in {"<", "<=", ">", ">="}:
        return None

    name = str(ind or "").strip()
    if not name:
        return None

    # lhs
    if name.lower() in {"open", "high", "low", "close", "volume", "vol"}:
        if name.lower() == "open":
            lhs = pd.to_numeric(df_feat.get("open"), errors="coerce")
        elif name.lower() == "high":
            lhs = pd.to_numeric(df_feat.get("high"), errors="coerce")
        elif name.lower() == "low":
            lhs = pd.to_numeric(df_feat.get("low"), errors="coerce")
        elif name.lower() == "close":
            lhs = pd.to_numeric(df_feat.get("close"), errors="coerce")
        else:
            lhs = pd.to_numeric(df_feat.get("volume"), errors="coerce")
    else:
        if name not in df_feat.columns:
            return None
        lhs = pd.to_numeric(df_feat[name], errors="coerce")

    # rhs: literal OR ref indicator (+ offset)
    if ref is not None and str(ref).strip():
        r = str(ref).strip()
        if r not in df_feat.columns and r.lower() not in {"open", "high", "low", "close", "volume", "vol"}:
            return None
        if r.lower() == "open":
            rhs0 = pd.to_numeric(df_feat.get("open"), errors="coerce")
        elif r.lower() == "high":
            rhs0 = pd.to_numeric(df_feat.get("high"), errors="coerce")
        elif r.lower() == "low":
            rhs0 = pd.to_numeric(df_feat.get("low"), errors="coerce")
        elif r.lower() == "close":
            rhs0 = pd.to_numeric(df_feat.get("close"), errors="coerce")
        elif r.lower() in {"volume", "vol"}:
            rhs0 = pd.to_numeric(df_feat.get("volume"), errors="coerce")
        else:
            rhs0 = pd.to_numeric(df_feat.get(r), errors="coerce")
        off = float(thr or 0.0)
        rhs = rhs0 + off
    else:
        rhs = float(thr)

    if op == "<":
        m = lhs < rhs
    elif op == "<=":
        m = lhs <= rhs
    elif op == ">":
        m = lhs > rhs
    else:
        m = lhs >= rhs

    return m.fillna(False)


def _entry_logic_masks(df_feat: pd.DataFrame, entry_logic: Dict[str, Any]) -> Optional[Tuple[pd.Series, pd.Series, pd.Series]]:
    """Return (regime_mask, entry_mask, combined_mask)."""
    if df_feat is None or df_feat.empty or not isinstance(entry_logic, dict):
        return None

    idx = df_feat.index
    regime_mask = pd.Series(True, index=idx)
    for c in entry_logic.get("regime", []) or []:
        cm = _cond_mask(df_feat, c)
        if cm is None:
            return None
        regime_mask &= cm

    clauses = entry_logic.get("clauses", []) or []
    if not clauses:
        entry_mask = pd.Series(True, index=idx)
    else:
        entry_mask = pd.Series(False, index=idx)
        for clause in clauses:
            if not clause:
                entry_mask |= True
                continue
            cm_all = pd.Series(True, index=idx)
            for c in clause:
                cm = _cond_mask(df_feat, c)
                if cm is None:
                    return None
                cm_all &= cm
            entry_mask |= cm_all

    combined = regime_mask & entry_mask
    return regime_mask, entry_mask, combined


def _entry_logic_true_pcts(df_feat: pd.DataFrame, entry_logic: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """(regime%, entry%, combined%)"""
    masks = _entry_logic_masks(df_feat, entry_logic)
    if masks is None:
        return None
    r, e, c = masks
    return float(r.mean() * 100.0), float(e.mean() * 100.0), float(c.mean() * 100.0)


def _human_condition(cond: Dict[str, Any]) -> str:
    if not isinstance(cond, dict):
        return ""
    ind = cond.get("indicator") or cond.get("feature") or cond.get("lhs")
    op = cond.get("operator") or cond.get("op")
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator")

    ind = str(ind or "").strip()
    op = str(op or "").strip()
    if ref is not None and str(ref).strip():
        r = str(ref).strip()
        off = float(thr or 0.0)
        if abs(off) < 1e-12:
            return f"{ind} {op} {r}"
        sign = "+" if off >= 0 else "-"
        return f"{ind} {op} {r} {sign} {abs(off):g}"
    else:
        try:
            v = float(thr)
            # prettier ints
            if abs(v - round(v)) < 1e-9:
                return f"{ind} {op} {int(round(v))}"
            return f"{ind} {op} {v:g}"
        except Exception:
            return f"{ind} {op} {thr}"


def _human_entry_logic(entry_logic: Dict[str, Any]) -> str:
    if not isinstance(entry_logic, dict):
        return ""
    parts: List[str] = []
    regime = entry_logic.get("regime", []) or []
    clauses = entry_logic.get("clauses", []) or []

    if regime:
        parts.append("Regime: " + " AND ".join([_human_condition(c) for c in regime if isinstance(c, dict)]))

    if not clauses:
        parts.append("Entry: (always allowed)")
    else:
        clause_strs = []
        for cl in clauses:
            if not cl:
                clause_strs.append("(always)")
            else:
                clause_strs.append("(" + " AND ".join([_human_condition(c) for c in cl if isinstance(c, dict)]) + ")")
        parts.append("Entry: " + " OR ".join(clause_strs))

    return " · ".join([p for p in parts if p.strip()])

# =============================================================================
# Entry gate UX helpers (plain-English + sanity widgets)
# =============================================================================

def _entry_logic_required_columns(entry_logic: Dict[str, Any]) -> List[str]:
    """Return required df_feat columns for this entry_logic (best-effort)."""
    need: set[str] = set()

    def _add(name: Any) -> None:
        nm = str(name or "").strip()
        if not nm:
            return
        if nm.lower() in {"open", "high", "low", "close", "volume", "vol"}:
            return
        need.add(nm)

    if not isinstance(entry_logic, dict):
        return []

    for c in (entry_logic.get("regime") or []):
        if isinstance(c, dict):
            _add(c.get("indicator") or c.get("feature") or c.get("lhs"))
            _add(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))
    for cl in (entry_logic.get("clauses") or []):
        for c in (cl or []):
            if isinstance(c, dict):
                _add(c.get("indicator") or c.get("feature") or c.get("lhs"))
                _add(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))

    return sorted(list(need))


def _cond_signature(cond: Dict[str, Any]) -> Optional[Tuple[str, str, str, Any]]:
    if not isinstance(cond, dict):
        return None
    ind = str(cond.get("indicator") or cond.get("feature") or cond.get("lhs") or "").strip()
    op = str(cond.get("operator") or cond.get("op") or "").strip()
    ref = str(cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator") or "").strip()
    thr = cond.get("threshold", cond.get("value", 0.0))
    if not ind or not op:
        return None
    try:
        thr_v = float(thr)
        # stable-ish signature
        thr_v = round(thr_v, 6)
        thr_key: Any = thr_v
    except Exception:
        thr_key = str(thr)
    return (ind, op, ref, thr_key)


def _find_duplicate_conditions(conds: List[Dict[str, Any]]) -> List[str]:
    """Return human-readable duplicate condition strings within a list."""
    seen: Dict[Tuple[str, str, str, Any], Tuple[int, str]] = {}
    dups: List[str] = []
    for c in (conds or []):
        sig = _cond_signature(c)
        if sig is None:
            continue
        txt = _human_condition(c) or "condition"
        if sig not in seen:
            seen[sig] = (1, txt)
        else:
            cnt, txt0 = seen[sig]
            seen[sig] = (cnt + 1, txt0)
    for sig, (cnt, txt0) in seen.items():
        if cnt > 1:
            dups.append(f"{txt0} (×{cnt})")
    return dups


def _plain_condition(cond: Dict[str, Any]) -> str:
    """Translate a condition into a more human phrase (best-effort)."""
    if not isinstance(cond, dict):
        return ""
    ind = str(cond.get("indicator") or cond.get("feature") or cond.get("lhs") or "").strip()
    op = str(cond.get("operator") or cond.get("op") or "").strip()
    thr = cond.get("threshold", cond.get("value", 0.0))
    ref = str(cond.get("ref_indicator") or cond.get("rhs") or cond.get("rhs_indicator") or "").strip()

    def _fmt_thr(x: Any) -> str:
        try:
            v = float(x)
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            # common compact formatting
            if abs(v) >= 10:
                return f"{v:.0f}"
            if abs(v) >= 1:
                return f"{v:.2f}"
            return f"{v:.3f}"
        except Exception:
            return str(x)

    # Special cases we actually offer in the UI
    if ind.lower() == "close" and ref.startswith("ema_") and op in {"<", "<=", ">", ">="}:
        try:
            ema_len = int(ref.split("_")[1])
        except Exception:
            ema_len = ref.replace("ema_", "")
        direction = "below" if op in {"<", "<="} else "above"
        return f"Price is {direction} EMA{ema_len}"

    if ind == "rsi_14" and op in {"<", "<="}:
        return f"RSI(14) is oversold (≤ {_fmt_thr(thr)})"
    if ind == "rsi_14" and op in {">", ">="}:
        return f"RSI(14) is strong (≥ {_fmt_thr(thr)})"

    if ind == "bb_z_20" and op in {"<", "<="}:
        return f"Price is stretched low (BB z-score ≤ {_fmt_thr(thr)})"
    if ind == "bb_z_20" and op in {">", ">="}:
        return f"Price is stretched high (BB z-score ≥ {_fmt_thr(thr)})"

    if ind.startswith("macd_hist") and op in {">", ">="}:
        return f"Momentum is bullish (MACD hist ≥ {_fmt_thr(thr)})"
    if ind.startswith("macd_hist") and op in {"<", "<="}:
        return f"Momentum is bearish (MACD hist ≤ {_fmt_thr(thr)})"

    if ind == "adx_14" and op in {">", ">="}:
        return f"Trend strength is high (ADX ≥ {_fmt_thr(thr)})"
    if ind == "adx_14" and op in {"<", "<="}:
        return f"Trend strength is low (ADX ≤ {_fmt_thr(thr)})"

    if ind == "donch_pos_20" and op in {"<", "<="}:
        return f"Price is near range low (Donchian pos ≤ {_fmt_thr(thr)})"
    if ind == "donch_pos_20" and op in {">", ">="}:
        return f"Price is near range high (Donchian pos ≥ {_fmt_thr(thr)})"

    # Fallback
    return _human_condition(cond) or "condition"


def _entry_logic_plain_english(entry_logic: Dict[str, Any]) -> str:
    """Return a short, plain-English explanation of the gate."""
    if not isinstance(entry_logic, dict):
        return ""

    reg = [c for c in (entry_logic.get("regime") or []) if isinstance(c, dict)]
    clauses = [cl for cl in (entry_logic.get("clauses") or []) if isinstance(cl, list) and len([c for c in cl if isinstance(c, dict)]) > 0]

    parts: List[str] = []
    if reg:
        reg_txt = " and ".join([_plain_condition(c) for c in reg])
        parts.append(f"Regime: only trade when {reg_txt}.")
    else:
        parts.append("Regime: none (always allowed).")

    if not clauses:
        parts.append("Triggers: none (buys are allowed whenever the regime is true).")  # consistent with engine semantics
    else:
        clause_bits = []
        for i, cl in enumerate(clauses):
            name = chr(65 + i)
            ct = [c for c in cl if isinstance(c, dict)]
            clause_txt = " and ".join([_plain_condition(c) for c in ct])
            clause_bits.append(f"Clause {name}: {clause_txt}")
        parts.append("Triggers: the buy window opens when ANY clause is true (OR). ")
        parts.extend([f"- {x}" for x in clause_bits])

    return "\n".join(parts).strip()


def _gate_strip_html(bits: List[bool]) -> str:
    """Render a tiny truth strip (left→right = older→newer)."""
    import html as _html
    if not bits:
        return ""
    sqs = []
    for b in bits:
        col = "rgba(46, 204, 113, 0.65)" if bool(b) else "rgba(49,51,63,0.18)"
        sqs.append(f"<span style='display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:2px;background:{col};'></span>")
    return "<div style='display:flex; align-items:center; gap:8px; flex-wrap:wrap;'>" + "<span>" + "".join(sqs) + "</span></div>"

def _gate_logic_tree_html(entry_logic: Dict[str, Any], snap: Optional[Dict[str, Any]] = None) -> str:
    """
    Lightweight visual "tree" for the gate: Regime (AND) + Trigger clauses (OR-of-AND).

    This is intentionally non-graphical (no SVG). It's a structured card layout that makes
    the logic readable at a glance.
    """
    snap = snap or {}
    regime = [c for c in (entry_logic.get("regime") or []) if isinstance(c, dict)]
    clauses_raw = entry_logic.get("clauses") or []
    clauses: List[List[Dict[str, Any]]] = []
    for cl in clauses_raw:
        if isinstance(cl, list):
            clauses.append([c for c in cl if isinstance(c, dict)])

    gate_now = snap.get("gate_now", None)
    blocker = str(snap.get("blocker_now") or "")
    cov = snap.get("coverage_pct", None)
    bits = snap.get("bits") or []

    # Which clause(s) are currently triggering (best-effort parse of snapshot text).
    triggered: set[str] = set()
    try:
        m = re.search(r"Gate is true\s*\(clause\s*([A-Z,\s]+)\)", blocker)
        if m:
            for tok in m.group(1).replace(",", " ").split():
                if len(tok) == 1 and tok.isalpha():
                    triggered.add(tok.upper())
        if not triggered:
            m2 = re.search(r"Triggered\s*\(clause\s*([A-Z])\)", blocker)
            if m2:
                triggered.add(m2.group(1).upper())
    except Exception:
        triggered = set()

    def chip(text: str, cls: str = "") -> str:
        return f"<span class='ff-gate-chip {cls}'>{_escape_html(text)}</span>"

    def cond_chip(text: str) -> str:
        return f"<span class='ff-gate-cond'>{_escape_html(text)}</span>"

    meta_bits: List[str] = []
    if gate_now is True:
        meta_bits.append(chip("Now: TRUE", "on"))
    elif gate_now is False:
        meta_bits.append(chip("Now: FALSE", "off"))
    else:
        meta_bits.append(chip("Now: —", "info"))

    if isinstance(cov, (int, float)) and math.isfinite(float(cov)):
        meta_bits.append(chip(f"Coverage: {float(cov):.1f}%", "info"))

    # Regime box
    if regime:
        reg_items = "".join([cond_chip(_plain_condition(c)) for c in regime])
        reg_sub = "All must be true (AND)"
    else:
        reg_items = cond_chip("No regime conditions (always in regime)")
        reg_sub = "Always-on regime"

    reg_box = (
        "<div class='ff-gate-box regime'>"
        "<div class='hdr'><div class='t'>1) Regime</div><div class='k'>AND</div></div>"
        f"<div class='sub'>{_escape_html(reg_sub)}</div>"
        f"<div class='ff-gate-conds'>{reg_items}</div>"
        "</div>"
    )

    # Triggers box (clauses)
    if clauses:
        clause_boxes = []
        for i, cl in enumerate(clauses):
            letter = chr(ord("A") + i)
            on = "on" if letter in triggered else ""
            if cl:
                items = "".join([cond_chip(_plain_condition(c)) for c in cl])
            else:
                items = cond_chip("Empty clause (will never trigger)")
            clause_boxes.append(
                "<div class='ff-clause {on}'>"
                "<div class='ct'><span>Clause {letter}</span><span class='mode'>AND</span></div>"
                "<div class='ff-gate-conds'>{items}</div>"
                "</div>".format(on=on, letter=letter, items=items)
            )
        trig_box_body = "<div class='ff-clause-grid'>" + "".join(clause_boxes) + "</div>"
        trig_sub = "Any clause can trigger (OR). Inside a clause, all conditions must be true (AND)."
    else:
        trig_box_body = "<div class='ff-gate-conds'>" + cond_chip("No trigger clauses (gate depends only on regime)") + "</div>"
        trig_sub = "No triggers configured"

    trig_box = (
        "<div class='ff-gate-box triggers'>"
        "<div class='hdr'><div class='t'>2) Triggers</div><div class='k'>OR</div></div>"
        f"<div class='sub'>{_escape_html(trig_sub)}</div>"
        f"{trig_box_body}"
        "</div>"
    )

    # Result
    res_cls = "on" if gate_now is True else ("off" if gate_now is False else "")
    res_text = "TRUE" if gate_now is True else ("FALSE" if gate_now is False else "—")
    why = blocker.strip()
    why_line = f"<div class='sub'>{_escape_html(why)}</div>" if why else "<div class='sub'>—</div>"
    res_box = (
        f"<div class='ff-gate-box result {res_cls}'>"
        "<div class='hdr'><div class='t'>3) Gate</div><div class='k'>Now</div></div>"
        f"<div class='sub'>Now: <b>{_escape_html(res_text)}</b></div>"
        f"{why_line}"
        "</div>"
    )

    foot = (
        "<div class='ff-gate-foot'>"
        "<b>Rule:</b> Buy allowed when <b>(Regime is true)</b> AND <b>(any Trigger clause is true)</b>. "
        "This is a mechanics preview, not a performance estimate."
        "</div>"
    )
    # Coverage snapshots (Early / Mid / Late), percent-based window (sanity check, not performance)
    snapshots_html = ""
    try:
        snaps = snap.get("snapshots") or []
        if snaps:
            rows = []
            for s in snaps:
                nm = str(s.get("name") or "")
                bits_s = s.get("bits") or []
                tn = int(s.get("true") or 0)
                nn = int(s.get("n") or 0)
                pct = s.get("pct")
                stat = f"{tn}/{nn}"
                try:
                    if isinstance(pct, (int, float)) and math.isfinite(float(pct)):
                        stat = f"{tn}/{nn} ({float(pct):.0f}%)"
                except Exception:
                    pass
                strip = _gate_strip_html([bool(b) for b in bits_s]) if bits_s else ""
                rows.append(
                    "<div style='display:flex; align-items:center; gap:8px; margin:4px 0;'>"
                    f"<div style='width:52px; font-size:0.74rem; opacity:0.65;'>{_escape_html(nm)}</div>"
                    + strip
                    + f"<div style='width:88px; text-align:right; font-size:0.74rem; opacity:0.65; white-space:nowrap; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;'>{_escape_html(stat)}</div>"
                    + "</div>"
                )
            title = "<div style='font-size:0.74rem; opacity:0.60; margin-top:6px; margin-bottom:2px;'><b>How often the gate opens (train)</b> — Early / Mid / Late</div>"
            note_txt = str(snap.get("snap_note") or "").strip()
            if not note_txt:
                note_txt = "Frequency only (not a performance estimate)."
            note = "<div style='font-size:0.70rem; opacity:0.55; margin-bottom:4px;'>" + _escape_html(note_txt) + "</div>"
            snapshots_html = "<div class='ff-gate-snapshots'>" + title + note + "<div>" + "".join(rows) + "</div></div>"
    except Exception:
        snapshots_html = ""



    return (
        "<div class='ff-gate-tree-wrap'>"
        "<div class='ff-gate-meta'>" + "".join(meta_bits) + "</div>"
        + snapshots_html
        + "<div class='ff-gate-tree'>"
        f"{reg_box}<div class='ff-gate-join'>AND</div>{trig_box}<div class='ff-gate-join'>→</div>{res_box}"
        "</div>"
        f"{foot}"
        "</div>"
    )



def _snap_window_len(n_train: int, window_pct: float, *, min_bars: int, max_bars: int) -> int:
    """Compute snapshot window size as % of train slice, clamped to [min_bars, max_bars]."""
    try:
        n_train = int(n_train)
    except Exception:
        n_train = 0
    if n_train <= 0:
        return 0
    try:
        w = int(round(float(n_train) * float(window_pct)))
    except Exception:
        w = int(round(float(n_train) * 0.02))
    if n_train < int(min_bars):
        return max(1, n_train)
    w = max(int(min_bars), w)
    w = min(int(max_bars), w)
    w = min(n_train, w)
    return max(1, int(w))


def _downsample_bool_bits(bits: List[bool], max_dots: int = 120) -> List[bool]:
    """Downsample boolean bits to at most max_dots for dot-strip display (majority vote per bin)."""
    try:
        max_dots = int(max_dots)
    except Exception:
        max_dots = 120
    if max_dots <= 0:
        return []
    if bits is None:
        return []
    try:
        bits = [bool(x) for x in bits]
    except Exception:
        return []
    n = len(bits)
    if n <= max_dots:
        return bits
    # Bin to max_dots; dot is True if >=50% of bin is True.
    out: List[bool] = []
    for i in range(max_dots):
        a = int((i * n) / max_dots)
        b = int(((i + 1) * n) / max_dots)
        if b <= a:
            b = min(n, a + 1)
        seg = bits[a:b]
        if not seg:
            out.append(False)
        else:
            t = sum(1 for x in seg if x)
            out.append(t >= (len(seg) / 2.0))
    return out


def _entry_logic_snapshot(
    df_feat: Optional[pd.DataFrame],
    entry_logic: Dict[str, Any],
    *,
    train_frac: float = 0.70,
    snap_window_pct: float = 0.02,
    snap_min_bars: int = 1,
    snap_max_bars: int = 200,
    snap_max_dots: int = 120,
) -> Dict[str, Any]:
    """Compute a non-performance sanity snapshot for the gate (train slice only).

    Notes:
    - "train" = the slice used for in-sample diagnostics (default: first 70% of rows).
    - Snapshots are *frequency* checks (how often gate is true), not performance estimates.
    """
    out: Dict[str, Any] = {
        "ok": False,
        "missing": [],
        "gate_now": None,
        "blocker_now": "",
        "bits": [],
        "true_count": 0,
        "lookback": 0,
        "coverage_pct": None,
        "snapshots": [],
        "snap_window_pct": float(snap_window_pct),
        "snap_min_bars": int(snap_min_bars),
        "snap_max_bars": int(snap_max_bars),
        "snap_max_dots": int(snap_max_dots),
        "snap_window_bars": 0,
        "snap_note": "",
        "note": "",
    }

    if df_feat is None or df_feat.empty:
        out["note"] = "No dataset loaded."
        return out

    try:
        n = int(len(df_feat))
        n_train = int(max(1, round(n * float(train_frac))))
        df_cov = df_feat.iloc[:n_train].copy()
    except Exception:
        out["note"] = "Could not slice dataset."
        return out

    # Required columns (best-effort)
    need: set = set()

    def _add_need(name: Any) -> None:
        nm = str(name or "").strip()
        if not nm:
            return
        if nm.lower() in {"open", "high", "low", "close", "volume", "vol"}:
            return
        need.add(nm)

    try:
        for c in (entry_logic.get("regime") or []):
            if isinstance(c, dict):
                _add_need(c.get("indicator") or c.get("feature") or c.get("lhs"))
                _add_need(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))
        for cl in (entry_logic.get("clauses") or []):
            for c in (cl or []):
                if isinstance(c, dict):
                    _add_need(c.get("indicator") or c.get("feature") or c.get("lhs"))
                    _add_need(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))
    except Exception:
        pass

    missing = sorted([c for c in need if c not in df_cov.columns])
    out["missing"] = missing
    if missing:
        out["note"] = "Missing required fields for coverage: " + ", ".join(missing)
        return out

    masks = _entry_logic_masks(df_cov, entry_logic)
    if masks is None:
        out["note"] = "Could not compute gate masks (missing columns or invalid condition)."
        return out

    r_mask, e_mask, c_mask = masks

    try:
        cov = float(c_mask.mean() * 100.0)
        out["coverage_pct"] = cov
    except Exception:
        out["coverage_pct"] = None

    # Gate now (last train bar)
    try:
        out["gate_now"] = bool(c_mask.iloc[-1])
    except Exception:
        out["gate_now"] = None

    # Blocker now (best-effort explanation)
    try:
        if out["gate_now"] is True:
            out["blocker_now"] = "Gate is true"
        else:
            # Find first failing regime / clause for a minimal hint.
            bl = ""
            try:
                reg = entry_logic.get("regime") or []
                for c in reg:
                    cm = _cond_mask(df_cov, c)
                    if cm is None:
                        continue
                    if not bool(cm.iloc[-1]):
                        bl = _human_condition(c)
                        break
            except Exception:
                pass
            if not bl:
                # Clauses: identify which clauses are true/false at the last bar.
                try:
                    cls = entry_logic.get("clauses") or []
                    ok_letters = []
                    for i, cl in enumerate(cls):
                        cm_all = pd.Series(True, index=df_cov.index)
                        ok = True
                        for c in (cl or []):
                            cm = _cond_mask(df_cov, c)
                            if cm is None:
                                ok = False
                                break
                            cm_all &= cm
                        if ok and bool(cm_all.iloc[-1]):
                            ok_letters.append(chr(65 + i))
                    if ok_letters:
                        bl = f"Triggered (clause {', '.join(ok_letters)})"
                    else:
                        bl = "No trigger clause is true"
                except Exception:
                    pass
            out["blocker_now"] = bl
    except Exception:
        pass

    # Snapshot windows (Early/Mid/Late) on train slice
    try:
        c_arr = [bool(x) for x in np.asarray(c_mask.values, dtype=bool)]
        n_tr = int(len(c_arr))
        w = _snap_window_len(n_tr, float(snap_window_pct), min_bars=int(snap_min_bars), max_bars=int(snap_max_bars))
        out["snap_window_bars"] = int(w)

        if w > 0:
            # Early = last w bars of the first third (avoids warmup bias)
            third_end = int(max(w, n_tr // 3))
            third_end = int(min(n_tr, third_end))
            early_raw = c_arr[int(max(0, third_end - w)) : third_end]

            mid_center = n_tr // 2
            mid_start = int(max(0, min(max(0, n_tr - w), mid_center - (w // 2))))
            mid_raw = c_arr[mid_start : mid_start + w]

            late_raw = c_arr[-w:]

            snaps_raw = [("Early", early_raw), ("Mid", mid_raw), ("Late", late_raw)]
            snaps = []
            for nm, arr in snaps_raw:
                nn = int(len(arr))
                tn = int(sum(1 for x in arr if x))
                pct = (tn / float(nn)) * 100.0 if nn else 0.0
                snaps.append(
                    {
                        "name": nm,
                        "bits": _downsample_bool_bits(arr, max_dots=int(snap_max_dots)),
                        "true": tn,
                        "n": nn,
                        "pct": pct,
                        "n_raw": nn,
                    }
                )
            out["snapshots"] = snaps

            # Keep a compact strip in out["bits"] for legacy callers: use the Late window.
            out["bits"] = _downsample_bool_bits(late_raw, max_dots=int(snap_max_dots))
            out["true_count"] = int(sum(1 for x in late_raw if x))
            out["lookback"] = int(w)

            pct_ui = float(snap_window_pct) * 100.0
            out["snap_note"] = f"Window = {pct_ui:.1f}% of train (capped at {int(snap_max_bars)} bars) → {int(w)} bars. Frequency only."
        else:
            out["snap_note"] = "Not enough data for snapshots."
    except Exception:
        out["snap_note"] = "Could not compute snapshots."

    out["ok"] = True
    return out
def _cond_from_state(prefix: str, ss: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Reconstruct a builder condition from st.session_state (no UI).
    try:
        cond_type = str(ss.get(f"{prefix}.type", "— (disabled)") or "— (disabled)")
    except Exception:
        cond_type = "— (disabled)"
    if not cond_type or cond_type.startswith("—"):
        return None

    def _op(default: str) -> str:
        v = ss.get(f"{prefix}.op", default)
        return str(v or default)

    def _thr(default: float) -> float:
        try:
            return float(ss.get(f"{prefix}.thr", default))
        except Exception:
            return float(default)

    if cond_type == "price_vs_ema":
        op = _op("<=")
        try:
            ln = int(ss.get(f"{prefix}.ema_len", 200))
        except Exception:
            ln = 200
        return {"indicator": "close", "operator": op, "ref_indicator": f"ema_{ln}", "threshold": 0.0}

    if cond_type == "rsi_14":
        return {"indicator": "rsi_14", "operator": _op("<="), "threshold": _thr(40.0)}

    if cond_type == "bb_z_20":
        return {"indicator": "bb_z_20", "operator": _op("<="), "threshold": _thr(-1.0)}

    if cond_type == "macd_hist_12_26_9":
        op = _op(">=")
        return {"indicator": "macd_hist_12_26_9", "operator": op, "threshold": _thr(0.0)}

    if cond_type == "adx_14":
        op = _op(">=")
        return {"indicator": "adx_14", "operator": op, "threshold": _thr(20.0)}

    if cond_type == "donch_pos_20":
        return {"indicator": "donch_pos_20", "operator": _op("<="), "threshold": _thr(0.20)}

    return None


def _entry_logic_from_builder_state(ss: Dict[str, Any]) -> Dict[str, Any]:
    # Best-effort reconstruction of entry_logic from the builder's session_state keys.
    reg_conds: List[Dict[str, Any]] = []
    clauses: List[List[Dict[str, Any]]] = []

    reg_ids = list(ss.get("new.logic.regime_ids") or [])
    for rid in reg_ids:
        c = _cond_from_state(f"new.regime.{rid}", ss)
        if c:
            reg_conds.append(c)

    clause_ids = list(ss.get("new.logic.clause_ids") or [])
    for cid in clause_ids:
        cond_ids = list(ss.get(f"new.logic.clause.{cid}.cond_ids") or [])
        cl: List[Dict[str, Any]] = []
        for cond_id in cond_ids:
            c = _cond_from_state(f"new.clause.{cid}.{cond_id}", ss)
            if c:
                cl.append(c)
        if cl:
            clauses.append(cl)

    return {"regime": reg_conds, "clauses": clauses}


def _render_skill_build_header(slot, *, df_feat: Optional[pd.DataFrame]) -> None:
    # Game-style 'skill build' header: cards + flow + stepper. Mechanics-only (no perf claims).
    ss = st.session_state
    ss.setdefault("build.active_module", "funding")
    ss.setdefault("build.active_modifier", "")

    # Pull current UI values (best-effort defaults)
    deposit_freq = str(ss.get("new.deposit_freq", "none") or "none").lower()
    deposit_amt = float(ss.get("new.deposit_amount", 0.0) or 0.0)

    buy_mode = str(ss.get("new.buy_mode", "scheduled") or "scheduled").strip().lower()
    buy_freq = str(ss.get("new.buy_freq", "weekly") or "weekly").strip().lower()
    buy_amt = float(ss.get("new.buy_amount", 0.0) or 0.0)
    max_buys_per_gate = int(float(ss.get("new.max_buys_per_gate", 0) or 0))

    entry_mode = str(ss.get("new.entry_mode", "Simple (one filter)") or "Simple (one filter)")
    buy_filter = str(ss.get("new.buy_filter", "none") or "none").strip().lower()

    max_alloc_pct = float(ss.get("new.max_alloc_pct", 1.0) or 1.0)

    sl_ui = float(ss.get("new.sl_pct_ui", 0.0) or 0.0)
    tp_ui = float(ss.get("new.tp_pct_ui", 0.0) or 0.0)
    trail_ui = float(ss.get("new.trail_pct_ui", 0.0) or 0.0)
    max_hold = int(ss.get("new.max_hold_bars", 0) or 0)

    # Effective entry_logic for diagnostics
    if str(entry_mode).startswith("Simple"):
        eff_logic = _simple_filter_to_entry_logic(buy_filter)
    else:
        eff_logic = _entry_logic_from_builder_state(ss)

    snap_pct_ui = float(ss.get('new.gate_snap_pct', 2.0) or 2.0)
    snap_max_bars = int(ss.get('new.gate_snap_max_bars', 200) or 200)
    snap_max_bars = int(max(30, min(500, snap_max_bars)))
    snap = _entry_logic_snapshot(df_feat, eff_logic, train_frac=0.70, snap_window_pct=float(snap_pct_ui)/100.0, snap_min_bars=1, snap_max_bars=snap_max_bars, snap_max_dots=120) if df_feat is not None else {'ok': False}
    # Cache for downstream sections (keeps Gate logic visual + config snapshots in sync)
    try:
        ss["cache.gate.snap"] = snap
        ss["cache.gate.eff_logic"] = eff_logic
        ss["cache.gate.snap_pct_ui"] = float(snap_pct_ui)
        ss["cache.gate.snap_max_bars"] = int(snap_max_bars)
    except Exception:
        pass
    cov = snap.get("coverage_pct") if isinstance(snap, dict) else None
    gate_now = snap.get("gate_now") if isinstance(snap, dict) else None
    blocker_now = str(snap.get("blocker_now") or "") if isinstance(snap, dict) else ""

    band = ""
    try:
        if cov is not None and float(cov) == float(cov):
            _c = float(cov)
            if _c < 5:
                band = "Rare"
            elif _c < 40:
                band = "Moderate"
            elif _c < 90:
                band = "Frequent"
            else:
                band = "Always-on"
    except Exception:
        band = ""

    # Labels
    fund_label = "Off" if (deposit_freq == "none" or deposit_amt <= 0) else f"{deposit_freq} · ${int(round(deposit_amt))}"
    if buy_amt <= 0:
        buy_label = "Off"
    else:
        core = f"{buy_freq} · ${int(round(buy_amt))}"
        if buy_mode == "signal":
            core = f"≤ {core}"
            if max_buys_per_gate > 0:
                core = f"{core} · max {max_buys_per_gate}"
        buy_label = core

    FILTER_LABELS = {
        "none": "Always buy",
        "below_ema": "Below EMA",
        "rsi_below": "RSI low",
        "bb_z_below": "BB stretch",
        "macd_bull": "MACD bullish",
        "adx_above": "ADX strong",
        "donch_pos_below": "Donch bottom",
    }
    if str(entry_mode).startswith("Simple"):
        gate_label = FILTER_LABELS.get(buy_filter, str(buy_filter))
    else:
        r_n = len((eff_logic or {}).get("regime") or [])
        c_n = len((eff_logic or {}).get("clauses") or [])
        gate_label = (f"{r_n} regime · {c_n} clause" + ("s" if c_n != 1 else "")) if (r_n or c_n) else "Always buy"

    cov_label = ""
    try:
        if cov is not None and float(cov) == float(cov):
            cov_label = f"{float(cov):.1f}% ({band})"
    except Exception:
        cov_label = ""

    alloc_label = "No cap" if max_alloc_pct >= 0.999 else f"Max {int(round(max_alloc_pct*100))}%"

    sl_label = "Off" if sl_ui <= 0 else f"{sl_ui:.1f}%"
    tp_label = "Off" if tp_ui <= 0 else f"{tp_ui:.1f}%"
    tr_label = "Off" if trail_ui <= 0 else f"{trail_ui:.1f}%"
    tm_label = "Off" if max_hold <= 0 else f"{max_hold} bars"

    # Warnings (purely mechanical)
    warn_gate = False
    warn_text = ""
    try:
        always_gate = bool(str(entry_mode).startswith("Simple") and buy_filter in {"none", ""}) or (
            (not str(entry_mode).startswith("Simple"))
            and len((eff_logic or {}).get("regime") or []) == 0
            and len((eff_logic or {}).get("clauses") or []) == 0
        )
        if buy_mode == "signal" and always_gate and buy_amt > 0 and max_buys_per_gate == 0:
            warn_gate = True
            warn_text = "Behaves like continuous DCA (gate always true + unlimited buys/window)."
    except Exception:
        pass

    active_module = str(ss.get("build.active_module") or "funding")
    active_mod = str(ss.get("build.active_modifier") or "")

    with slot.container():
        st.markdown("### Strategy build")
        st.caption("Game-style loadout view (read-only). Mechanics-only (not advice).")

        cards = [
            ("Funding", fund_label, "Adds cash to the pile.", "econ", "funding", ""),
            ("Buy trigger", buy_label, "When buys can attempt.", "trigger", "buys", ""),
            ("Gate", (gate_label + (f" · {cov_label}" if cov_label else "")), "Permission to buy.", "gate", "entry", ""),
            ("Allocation cap", alloc_label, "Max exposure.", "alloc", "allocation", ""),
            ("Stop loss", sl_label, "Cuts losses.", "risk", "risk", "stop_loss"),
            ("Take profit", tp_label, "Harvests gains.", "risk", "risk", "take_profit"),
            ("Trailing", tr_label, "Protects peak.", "risk", "risk", "trailing"),
            ("Time stop", tm_label, "Exits after time.", "risk", "risk", "time_stop"),
        ]

        for r in range(2):
            cols = st.columns(4, gap="small")
            for j in range(4):
                i = r * 4 + j
                t, s, k, style, mod, sub = cards[i]
                is_active = False  # header is read-only
                cls = f"ff-skill-card ff-skill-{style}"
                if str(s).strip().lower() in {"off", "no cap"} and not is_active:
                    cls += " off"
                if t == "Gate" and warn_gate:
                    cls += " warn"
                cols[j].markdown(
                    f"<div class='{cls}'>"
                    f"<div class='t'>{_escape_html(str(t))}</div>"
                    f"<div class='s'>{_escape_html(str(s))}</div>"
                    f"<div class='k'>{_escape_html(str(k))}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if warn_text:
            st.warning(warn_text)

        gate_now_txt = "Unknown"
        if gate_now is True:
            gate_now_txt = "TRUE"
        elif gate_now is False:
            gate_now_txt = "FALSE"

        exits = []
        if sl_ui > 0:
            exits.append(f"SL {sl_ui:.1f}%")
        if tp_ui > 0:
            exits.append(f"TP {tp_ui:.1f}%")
        if trail_ui > 0:
            exits.append(f"Trail {trail_ui:.1f}%")
        if max_hold > 0:
            exits.append(f"Time {max_hold}b")

        pieces = []
        pieces.append(f"Mode: {'Gate-driven' if buy_mode == 'signal' else 'Scheduled'}")
        pieces.append(f"Buy: {buy_label}")
        if cov_label:
            pieces.append(f"Gate coverage: {cov_label}")
        pieces.append(f"Cap: {alloc_label}")
        pieces.append("Exits: " + (", ".join(exits) if exits else "Off"))
        pieces.append(f"Gate now: {gate_now_txt}")

        st.markdown(
            "<div class='ff-build-summary'>"
            + " ".join([f"<span class='ff-summary-chip'>{_escape_html(p)}</span>" for p in pieces])
            + "</div>",
            unsafe_allow_html=True,
        )

        # Gate logic (visual): shown whenever the Logic Builder is selected.
        if not str(entry_mode).startswith("Simple"):
            st.markdown("##### Gate logic (visual)")
            st.markdown(_gate_logic_tree_html(eff_logic, snap), unsafe_allow_html=True)

    



def build_dca_baseline_params() -> Dict[str, Any]:
    st.subheader("Baseline Plan")

    # Render a "build card" header placeholder (filled after controls are read).
    header_slot = st.empty()

    # Build-step context (optional): compute indicator columns to show “how often will this trigger?”
    df_feat = None
    data_path_str = st.session_state.get("new.data_path")
    if data_path_str:
        try:
            p = Path(str(data_path_str))
            if p.exists():
                df_feat = _add_features_cached(str(p), p.stat().st_mtime)
        except Exception:
            df_feat = None

    # Skill build header (cards + flow)
    _render_skill_build_header(header_slot, df_feat=df_feat)

    build_atr_med = None

    # -------------------------
    # Loadout (single-column)
    # -------------------------
    colL = st.container()

    # Defaults for legacy knobs (used in simple mode and as defaults in the builder)
    buy_filter = "none"
    ema_len = 200
    rsi_thr = 40.0
    macd_hist_thr = 0.0
    bb_z_thr = -1.0
    adx_thr = 20.0
    donch_pos_thr = 0.20

    entry_logic: Dict[str, Any] = {"regime": [], "clauses": []}

    # Defaults (controls disabled by default)
    sl_pct = 0.0
    tp_pct = 0.0
    tp_sell_fraction = 1.0
    reserve_frac = 0.0
    max_hold_bars = 0
    trail_pct = 0.0

    with colL:
        # ------------------------------------------------------------------
        # Skill strip navigation (Phase 3)
        # ------------------------------------------------------------------
        if "build.active_module" not in st.session_state:
            st.session_state["build.active_module"] = "funding"
        if "build.active_modifier" not in st.session_state:
            st.session_state["build.active_modifier"] = ""

        active_module = str(st.session_state.get("build.active_module") or "funding")
        active_mod = str(st.session_state.get("build.active_modifier") or "")

        def _builder_has_any_clause() -> bool:
            for i in range(1, 4):
                for j in range(1, 4):
                    v = st.session_state.get(f"new.cl{i}.c{j}.type")
                    if v and (not str(v).startswith("—")):
                        return True
            return False

        # Determine "configured" states from session_state (best-effort, purely UI)
        dep_freq_ss = str(st.session_state.get("new.deposit_freq", "none")).lower()
        dep_amt_ss = float(st.session_state.get("new.deposit_amount", 0.0) or 0.0)
        buy_amt_ss = float(st.session_state.get("new.buy_amount", 0.0) or 0.0)
        entry_mode_ss = str(st.session_state.get("new.entry_mode", "Simple (one filter)"))
        buy_filter_ss = str(st.session_state.get("new.buy_filter", "none"))
        max_alloc_ss = float(st.session_state.get("new.max_alloc_pct", 1.0) or 1.0)

        sl_ui_ss = float(st.session_state.get("new.sl_pct_ui", 0.0) or 0.0)
        tp_ui_ss = float(st.session_state.get("new.tp_pct_ui", 0.0) or 0.0)
        max_hold_ss = int(st.session_state.get("new.max_hold_bars", 0) or 0)
        trail_ui_ss = float(st.session_state.get("new.trail_pct_ui", 0.0) or 0.0)

        funding_on = (dep_freq_ss != "none") and (dep_amt_ss > 0)
        buys_on = buy_amt_ss > 0
        entry_on = (entry_mode_ss.startswith("Simple") and buy_filter_ss != "none") or ((not entry_mode_ss.startswith("Simple")) and _builder_has_any_clause())
        alloc_on = max_alloc_ss < 1.0
        exits_on = (sl_ui_ss > 0) or (tp_ui_ss > 0) or (max_hold_ss > 0) or (trail_ui_ss > 0)

        def _node_icon(target: str, *, sub: str = "", configured: bool = False) -> str:
            if active_module == target and (not sub or active_mod == sub):
                return "🔷"
            return "🟩" if configured else "⬜"

        def _skill_btn(col, label: str, target: str, *, sub: str = "", configured: bool = False):
            icon = _node_icon(target, sub=sub, configured=configured)
            key = f"build.skill.{target}.{sub or 'main'}"
            if col.button(f"{icon} {label}", key=key):
                st.session_state["build.active_module"] = target
                st.session_state["build.active_modifier"] = sub
                st.rerun()


        # -------------------------
        # Funding module
        # -------------------------
        st.markdown("<div class='ff-module ff-module-econ " + ("active" if active_module == "funding" else "") + "'>", unsafe_allow_html=True)
        with st.container():
            top_a, top_b = st.columns([0.62, 0.38])
            with top_a:
                st.markdown("**Funding**")
                st.caption("Cash in schedule (optional).")
            with top_b:
                fund_sum = st.empty()

            with st.expander("Configure", expanded=(active_module == "funding")):
                deposit_freq = st.selectbox(
                    "Deposit frequency",
                    options=["none", "daily", "weekly", "monthly"],
                    index=2,
                    key="new.deposit_freq",
                )
                deposit_amount = st.number_input(
                    "Deposit amount (USD)",
                    min_value=0.0,
                    value=50.0,
                    step=10.0,
                    key="new.deposit_amount",
                )
                if str(deposit_freq).lower() == "none":
                    deposit_amount = 0.0

            fund_label = "Off" if (str(deposit_freq).lower() == "none" or float(deposit_amount) <= 0) else f"{deposit_freq} · ${int(round(float(deposit_amount)))}"
            fund_sum.markdown(f"<div style='text-align:right; font-weight:700'>{fund_label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # Buy cadence module
        # -------------------------
        st.markdown("<div class='ff-module ff-module-trigger " + ("active" if active_module == "buys" else "") + "'>", unsafe_allow_html=True)
        with st.container():
            top_a, top_b = st.columns([0.62, 0.38])
            with top_a:
                st.markdown("**Buy cadence**")
                _bm = str(st.session_state.get("new.buy_mode", "scheduled") or "scheduled").strip().lower()
                if _bm == "signal":
                    st.caption("Buys are driven by the entry gate; this sets the spacing between buys.")
                else:
                    st.caption("Buys attempt on a schedule (subject to allocation + entry gate).")
            with top_b:
                buy_sum = st.empty()

            with st.expander("Configure", expanded=(active_module == "buys")):
                buy_mode = st.radio(
                    "Buy trigger",
                    options=["scheduled", "signal"],
                    format_func=lambda x: "Buy on schedule" if x == "scheduled" else "Buy while gate is true",
                    horizontal=True,
                    key="new.buy_mode",
                )

                if str(buy_mode).strip().lower() == "signal":
                    freq_label = "Max buy frequency (cooldown)"
                    st.caption("While the gate is true, buys can happen on any day — but not more often than this.")
                else:
                    freq_label = "Buy frequency"
                    st.caption("The strategy only attempts buys on this schedule; the gate can veto the attempt.")

                buy_freq = st.selectbox(
                    freq_label,
                    options=["daily", "weekly", "monthly"],
                    index=1,
                    key="new.buy_freq",
                )
                buy_amount = st.number_input(
                    "Buy amount (USD)",
                    min_value=0.0,
                    value=50.0,
                    step=10.0,
                    key="new.buy_amount",
                )

                if str(buy_mode).strip().lower() == "signal":
                    st.number_input(
                        "Max buys per signal window (0 = unlimited)",
                        min_value=0,
                        value=int(st.session_state.get("new.max_buys_per_gate", 0) or 0),
                        step=1,
                        key="new.max_buys_per_gate",
                    )

            # Canonical values (used later to build params + summary)
            buy_mode = str(st.session_state.get("new.buy_mode", "scheduled") or "scheduled").strip().lower()
            max_buys_per_gate = int(float(st.session_state.get("new.max_buys_per_gate", 0) or 0))

            if float(buy_amount) <= 0:
                buy_label = "Off"
            else:
                core = f"{buy_freq} · ${int(round(float(buy_amount)))}"
                if buy_mode == "signal":
                    core = f"≤ {core}"
                    if max_buys_per_gate > 0:
                        core = f"{core} · max {max_buys_per_gate}"
                buy_label = core

            buy_sum.markdown(f"<div style='text-align:right; font-weight:700'>{buy_label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
# Entry gate module
        # -------------------------
        st.markdown("<div class='ff-module ff-module-gate " + ("active" if active_module == "entry" else "") + "'>", unsafe_allow_html=True)
        with st.container():
            top_a, top_b = st.columns([0.62, 0.38])
            with top_a:
                st.markdown("**Entry gate**")
                if str(buy_mode).strip().lower() == "signal":
                    st.caption("Controls when we are in accumulate mode. While true, buys may fire (spaced by your max frequency). Mechanics only (not recommendations).")
                else:
                    st.caption("Controls whether scheduled buy attempts are allowed to fire. Mechanics only (not recommendations).")
            with top_b:
                gate_sum = st.empty()

            with st.expander("Configure", expanded=(active_module == "entry")):
                entry_mode = st.radio(
                    "Entry logic mode",
                    options=["Simple (one filter)", "Logic builder (regime + triggers)"],
                    index=0,
                    horizontal=True,
                    key="new.entry_mode",
                )

                def _cond_ui(prefix: str) -> Optional[Dict[str, Any]]:
                    """Builder UI for one condition. Returns condition dict or None if disabled."""
                    cond_type = st.selectbox(
                        "Condition",
                        options=[
                            "— (disabled)",
                            "price_vs_ema",
                            "rsi_14",
                            "bb_z_20",
                            "macd_hist_12_26_9",
                            "adx_14",
                            "donch_pos_20",
                        ],
                        index=0,
                        key=f"{prefix}.type",
                        format_func=lambda v: {
                            "— (disabled)": "— (disabled)",
                            "price_vs_ema": "Price vs EMA",
                            "rsi_14": "RSI(14)",
                            "bb_z_20": "Bollinger z-score(20)",
                            "macd_hist_12_26_9": "MACD histogram(12,26,9)",
                            "adx_14": "ADX(14)",
                            "donch_pos_20": "Donchian position(20)",
                        }.get(v, str(v)),
                    )
                    if cond_type.startswith("—"):
                        return None

                    if cond_type == "price_vs_ema":
                        op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                        ln = int(st.selectbox("EMA length", options=[10, 20, 50, 100, 200], index=4, key=f"{prefix}.ema_len"))
                        return {"indicator": "close", "operator": op, "ref_indicator": f"ema_{ln}", "threshold": 0.0}

                    if cond_type == "rsi_14":
                        op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                        thr = float(st.slider("Threshold", 5.0, 95.0, 40.0, 1.0, key=f"{prefix}.thr"))
                        return {"indicator": "rsi_14", "operator": op, "threshold": thr}

                    if cond_type == "bb_z_20":
                        op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                        thr = float(st.slider("Threshold", -3.0, 3.0, -1.0, 0.1, key=f"{prefix}.thr"))
                        return {"indicator": "bb_z_20", "operator": op, "threshold": thr}

                    if cond_type == "macd_hist_12_26_9":
                        op = st.selectbox("Operator", options=[">=", "<="], index=0, key=f"{prefix}.op")
                        thr = float(st.number_input("Threshold", value=0.0, step=0.1, key=f"{prefix}.thr"))
                        return {"indicator": "macd_hist_12_26_9", "operator": op, "threshold": thr}

                    if cond_type == "adx_14":
                        op = st.selectbox("Operator", options=[">=", "<="], index=0, key=f"{prefix}.op")
                        thr = float(st.slider("Threshold", 1.0, 80.0, 20.0, 1.0, key=f"{prefix}.thr"))
                        return {"indicator": "adx_14", "operator": op, "threshold": thr}

                    if cond_type == "donch_pos_20":
                        op = st.selectbox("Operator", options=["<=", ">="], index=0, key=f"{prefix}.op")
                        thr = float(st.slider("Threshold", 0.0, 1.0, 0.20, 0.05, key=f"{prefix}.thr"))
                        return {"indicator": "donch_pos_20", "operator": op, "threshold": thr}

                    return None

                if entry_mode.startswith("Simple"):
                    FILTER_LABELS = {
                        "none": "Always buy (no filter)",
                        "below_ema": "Buy dips below EMA",
                        "rsi_below": "Oversold (RSI low)",
                        "bb_z_below": "Oversold (Bollinger stretch)",
                        "macd_bull": "Momentum (MACD bullish)",
                        "adx_above": "Trend strength (ADX)",
                        "donch_pos_below": "Range bottom (Donchian)",
                    }
                    FILTER_DESC = {
                        "none": "Buys fire on schedule (subject to max allocation).",
                        "below_ema": "Only buy when price is below a moving average (dip gate).",
                        "rsi_below": "Only buy when RSI is below a threshold (oversold gate).",
                        "bb_z_below": "Only buy when price is stretched below its Bollinger midline (z‑score).",
                        "macd_bull": "Only buy when momentum is bullish (MACD histogram ≥ threshold).",
                        "adx_above": "Only buy when trend strength is above a threshold (ADX).",
                        "donch_pos_below": "Only buy near the bottom of the recent Donchian range.",
                    }

                    buy_filter = st.selectbox(
                        "Entry filter (TradingView‑style)",
                        options=[
                            "none",
                            "below_ema",
                            "rsi_below",
                            "bb_z_below",
                            "macd_bull",
                            "adx_above",
                            "donch_pos_below",
                        ],
                        index=0,
                        key="new.buy_filter",
                        format_func=lambda v: FILTER_LABELS.get(v, v),
                    )
                    st.caption(FILTER_DESC.get(buy_filter, ""))

                    if buy_filter == "below_ema":
                        ema_len = int(st.selectbox("EMA length", options=[10, 20, 50, 100, 200], index=4, key="new.ema_len"))
                    elif buy_filter == "rsi_below":
                        rsi_thr = float(st.slider("RSI threshold (buy when RSI ≤ threshold)", 5.0, 80.0, 40.0, 1.0, key="new.rsi_thr"))
                    elif buy_filter == "bb_z_below":
                        bb_z_thr = float(st.slider("Bollinger z-score threshold (buy when z ≤ threshold)", -3.0, 0.0, -1.0, 0.1, key="new.bb_z_thr"))
                    elif buy_filter == "macd_bull":
                        macd_hist_thr = float(st.number_input("MACD histogram threshold (hist ≥ threshold)", value=0.0, step=0.1, key="new.macd_hist_thr"))
                    elif buy_filter == "adx_above":
                        adx_thr = float(st.slider("ADX threshold (buy when ADX ≥ threshold)", 5.0, 60.0, 20.0, 1.0, key="new.adx_thr"))
                    elif buy_filter == "donch_pos_below":
                        donch_pos_thr = float(st.slider("Donchian position threshold (pos ≤ threshold)", 0.0, 1.0, 0.20, 0.05, key="new.donch_pos_thr"))

                    entry_logic = _simple_filter_to_entry_logic(
                        buy_filter,
                        ema_len=ema_len,
                        rsi_thr=rsi_thr,
                        macd_hist_thr=macd_hist_thr,
                        bb_z_thr=bb_z_thr,
                        adx_thr=adx_thr,
                        donch_pos_thr=donch_pos_thr,
                    )

                    # Tiny “reality check” in simple mode
                    if df_feat is not None and not df_feat.empty:
                        try:
                            pct = _filter_true_pct(
                                df_feat,
                                buy_filter,
                                ema_len=ema_len,
                                rsi_thr=rsi_thr,
                                macd_hist_thr=macd_hist_thr,
                                bb_z_thr=bb_z_thr,
                                adx_thr=adx_thr,
                                donch_pos_thr=donch_pos_thr,
                            )
                            atr_med = None
                            if "atr_pct" in df_feat.columns:
                                atr_med = float(np.nanmedian(pd.to_numeric(df_feat["atr_pct"], errors="coerce")))
                            build_atr_med = atr_med
                            msg = None
                            if pct is not None:
                                msg = f"On this dataset: filter true ~{pct:.0f}% of days"
                            if atr_med is not None and math.isfinite(atr_med):
                                msg = (msg + f" · median daily ATR ≈ {atr_med:.1f}%") if msg else f"Median daily ATR ≈ {atr_med:.1f}%"
                            if msg:
                                st.caption(msg + ".")
                            if pct is not None and pct < 5:
                                st.info("FYI: this filter triggers on <5% of days here. That typically reduces trade count and increases outcome variability.")
                        except Exception:
                            pass
                else:
                    st.caption("Builder: define a small set of gates (regime) and a few entry trigger clauses (any-of). Caps keep this auditable.")

                    # Progressive builder state (show only what exists)
                    if "new.logic.uid" not in st.session_state:
                        st.session_state["new.logic.uid"] = 0
                    if "new.logic.regime_ids" not in st.session_state:
                        st.session_state["new.logic.regime_ids"] = []
                    if "new.logic.clause_ids" not in st.session_state:
                        st.session_state["new.logic.clause_ids"] = []

                    def _new_uid() -> int:
                        st.session_state["new.logic.uid"] = int(st.session_state.get("new.logic.uid", 0) or 0) + 1
                        return int(st.session_state["new.logic.uid"])

                    def _clear_keys_with_prefix(prefix: str) -> None:
                        # Best-effort cleanup so removed items don't reappear when re-added.
                        try:
                            ks = [k for k in st.session_state.keys() if str(k).startswith(prefix)]
                            for k in ks:
                                del st.session_state[k]
                        except Exception:
                            pass

                    shape_slot = st.empty()
                    preview_slot = st.empty()
                    plain_slot = st.empty()
                    now_slot = st.empty()
                    strip_slot = st.empty()
                    snapshots_slot = st.empty()

                    reg_conds: List[Dict[str, Any]] = []
                    reg_ids: List[int] = list(st.session_state.get("new.logic.regime_ids") or [])

                    st.markdown("##### Regime gates (AND)")
                    cols_r = st.columns([0.70, 0.30])
                    with cols_r[0]:
                        st.caption("All regime conditions must be true. Optional.")
                    with cols_r[1]:
                        if len(reg_ids) < 2:
                            if st.button("+ Add regime condition", key="new.logic.add_regime"):
                                reg_ids.append(_new_uid())
                                st.session_state["new.logic.regime_ids"] = reg_ids
                                st.rerun()

                    if not reg_ids:
                        st.caption("No regime filters (always in regime).")
                    else:
                        for i, rid in enumerate(list(reg_ids)):
                            st.markdown(f"**Regime condition {i+1}**")
                            ccols = st.columns([0.92, 0.08])
                            with ccols[0]:
                                c = _cond_ui(f"new.regime.{rid}")
                            with ccols[1]:
                                if st.button("✕", key=f"new.logic.rm_regime.{rid}"):
                                    reg_ids = [x for x in reg_ids if x != rid]
                                    st.session_state["new.logic.regime_ids"] = reg_ids
                                    _clear_keys_with_prefix(f"new.regime.{rid}")
                                    st.rerun()
                            if c:
                                reg_conds.append(c)

                    st.divider()

                    st.markdown("##### Trigger clauses (OR of AND)")
                    cols_c = st.columns([0.70, 0.30])
                    with cols_c[0]:
                        st.caption("Any clause can trigger. Inside a clause, all conditions must be true.")
                    with cols_c[1]:
                        if len(st.session_state.get("new.logic.clause_ids") or []) < 3:
                            if st.button("+ Add trigger clause", key="new.logic.add_clause"):
                                cid = _new_uid()
                                clause_ids = list(st.session_state.get("new.logic.clause_ids") or [])
                                clause_ids.append(cid)
                                st.session_state["new.logic.clause_ids"] = clause_ids
                                st.session_state[f"new.logic.clause.{cid}.cond_ids"] = []
                                st.rerun()

                    clauses: List[List[Dict[str, Any]]] = []
                    clause_ids: List[int] = list(st.session_state.get("new.logic.clause_ids") or [])

                    if not clause_ids:
                        st.caption("No trigger clauses yet. Add at least one clause + condition to make this gate do something.")
                    else:
                        for ci, cid in enumerate(list(clause_ids)):
                            clause_name = chr(65 + ci)  # A/B/C
                            hdr = st.columns([0.78, 0.22])
                            with hdr[0]:
                                st.markdown(f"**Clause {clause_name} (AND)**")
                            with hdr[1]:
                                if st.button("Remove clause", key=f"new.logic.rm_clause.{cid}"):
                                    clause_ids = [x for x in clause_ids if x != cid]
                                    st.session_state["new.logic.clause_ids"] = clause_ids
                                    _clear_keys_with_prefix(f"new.clause.{cid}")
                                    if f"new.logic.clause.{cid}.cond_ids" in st.session_state:
                                        del st.session_state[f"new.logic.clause.{cid}.cond_ids"]
                                    st.rerun()

                            cond_ids_key = f"new.logic.clause.{cid}.cond_ids"
                            cond_ids: List[int] = list(st.session_state.get(cond_ids_key) or [])

                            cl: List[Dict[str, Any]] = []
                            if not cond_ids:
                                st.caption("No conditions yet.")
                            for j, cond_id in enumerate(list(cond_ids)):
                                st.markdown(f"Condition {j+1}")
                                row = st.columns([0.92, 0.08])
                                with row[0]:
                                    c = _cond_ui(f"new.clause.{cid}.{cond_id}")
                                with row[1]:
                                    if st.button("✕", key=f"new.logic.rm_cond.{cid}.{cond_id}"):
                                        cond_ids = [x for x in cond_ids if x != cond_id]
                                        st.session_state[cond_ids_key] = cond_ids
                                        _clear_keys_with_prefix(f"new.clause.{cid}.{cond_id}")
                                        st.rerun()
                                if c:
                                    cl.append(c)

                            # Duplicate condition check (within this clause)
                            try:
                                dups = _find_duplicate_conditions(cl)
                                if dups:
                                    st.warning("Duplicate conditions in this clause (redundant): " + "; ".join(dups))
                            except Exception:
                                pass

                            if len(cond_ids) < 3:
                                if st.button("+ Add condition", key=f"new.logic.add_cond.{cid}"):
                                    cond_ids.append(_new_uid())
                                    st.session_state[cond_ids_key] = cond_ids
                                    st.rerun()

                            if cl:
                                clauses.append(cl)

                            if ci < (len(clause_ids) - 1):
                                st.markdown("<div style='text-align:center; font-weight:800; opacity:0.65; margin: 6px 0;'>OR</div>", unsafe_allow_html=True)
                                st.divider()

                    if not clauses:
                        st.warning("No trigger clause is enabled yet. Add at least one condition to at least one clause.")

                    # Build logic from session state to match the visual/summary diagnostics.
                    entry_logic = _entry_logic_from_builder_state(st.session_state)

                    # Live formula (shape) + human preview
                    try:
                        r_n = len(entry_logic.get("regime") or [])
                        c_n = len(entry_logic.get("clauses") or [])
                        if c_n == 0 and r_n == 0:
                            rule = "Buy always allowed (no filters)."
                        elif c_n == 0:
                            rule = "Buy allowed when Regime is true."
                        elif r_n == 0:
                            rule = "Buy allowed when any Trigger clause is true."
                        else:
                            rule = "Buy allowed when Regime is true AND any Trigger clause is true."
                        shape_slot.markdown(f"**Rule:** {rule}")
                        human = _human_entry_logic(entry_logic)
                        if human:
                            preview_slot.caption(f"Current gate (preview): {human}")
                    except Exception:
                        pass


                    # Plain-English explanation + gate-now sanity (train slice; not performance)
                    try:
                        pe = _entry_logic_plain_english(entry_logic)
                        if pe:
                            plain_slot.markdown("**This gate means:**\n" + pe)
                    except Exception:
                        pass
                    
                    # Gate snapshot settings (window size for Early/Mid/Late dot bars)
                    snap_pct_ui = float(st.session_state.get('new.gate_snap_pct', 2.0) or 2.0)
                    snap_max_bars = int(st.session_state.get('new.gate_snap_max_bars', 200) or 200)
                    with st.expander('Gate snapshot settings', expanded=False):
                        snap_pct_ui = float(st.slider('Snapshot window (% of train)', min_value=1.0, max_value=5.0, value=float(snap_pct_ui), step=0.5, key='new.gate_snap_pct'))
                        snap_max_bars = int(st.slider('Max bars per snapshot window', min_value=30, max_value=500, value=int(snap_max_bars), step=10, key='new.gate_snap_max_bars'))
                    snap_max_bars = int(max(30, min(500, snap_max_bars)))
                    
                    try:
                        snap = None
                        try:
                            _cached = st.session_state.get("cache.gate.snap")
                            if isinstance(_cached, dict) and _cached.get("ok"):
                                try:
                                    if abs(float(_cached.get("snap_window_pct", 0.0)) - (float(snap_pct_ui) / 100.0)) < 1e-12 and int(_cached.get("snap_max_bars", 0)) == int(snap_max_bars):
                                        snap = _cached
                                except Exception:
                                    pass
                        except Exception:
                            snap = None
                        if snap is None:
                            snap = _entry_logic_snapshot(df_feat, entry_logic, train_frac=0.70, snap_window_pct=float(snap_pct_ui)/100.0, snap_min_bars=1, snap_max_bars=snap_max_bars, snap_max_dots=120)
                        try:
                            st.session_state["cache.gate.snap"] = snap
                        except Exception:
                            pass
                        miss = snap.get("missing") or []
                        if miss:
                            now_slot.markdown(
                                "<div style='font-size:0.86rem; opacity:0.78;'>"
                                "<b>Now (train latest):</b> (can’t evaluate) — missing: "
                                + ", ".join([str(x) for x in miss])
                                + "</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            g = snap.get("gate_now", None)
                            if g is None:
                                now_slot.markdown("<div style='font-size:0.86rem; opacity:0.78;'><b>Gate now:</b> (can’t evaluate)</div>", unsafe_allow_html=True)
                            else:
                                g = bool(g)
                                bg = "rgba(46, 204, 113, 0.16)" if g else "rgba(231, 76, 60, 0.14)"
                                txt = "TRUE" if g else "FALSE"
                                blocker = str(snap.get("blocker_now") or "").strip()
                                sub = ""
                                if blocker:
                                    bl = blocker.strip()
                                    m_tr = re.search(r"\(clause\s*([A-Z,\s]+)\)", bl, flags=re.I)
                                    if g and m_tr:
                                        toks = [t for t in re.split(r"[\s,]+", m_tr.group(1).upper().strip()) if t]
                                        letters = [t for t in toks if (len(t) == 1 and t.isalpha())]
                                        if letters:
                                            if len(letters) == 1:
                                                sub = f" — Triggered by Clause {letters[0]}"
                                            else:
                                                sub = " — Triggered by " + ", ".join([f"Clause {x}" for x in letters])
                                    elif bl.lower() != "gate is true":
                                        sub = " — " + bl
                                now_slot.markdown(
                                    f"<span style='display:inline-block; padding:4px 10px; border-radius:999px; border:1px solid rgba(49,51,63,0.18); background:{bg}; font-size:0.82rem; font-weight:650;'>Now (train latest): {txt}</span>"
                                    f"<span style='font-size:0.82rem; opacity:0.70; margin-left:8px;'>{sub}</span>",
                                    unsafe_allow_html=True,
                                )

                                                # Removed: "Last N bars" strip. We now show Early/Middle/Late snapshots instead.
                        try:
                            strip_slot.empty()
                        except Exception:
                            pass

# Coverage snapshots (train slice): Early / Middle / Late (sanity check across time; not performance)
                        try:
                            snaps = snap.get("snapshots") or []
                            if snaps:
                                rows = []
                                for s in snaps:
                                    nm = str(s.get("name") or "")
                                    bits_s = s.get("bits") or []
                                    tn = int(s.get("true") or 0)
                                    nn = int(s.get("n") or 0)
                                    pct = s.get("pct")
                                    stat = f"{tn}/{nn}"
                                    try:
                                        if isinstance(pct, (int, float)) and math.isfinite(float(pct)):
                                            stat = f"{tn}/{nn} ({float(pct):.0f}%)"
                                    except Exception:
                                        pass
                                    strip = _gate_strip_html([bool(b) for b in bits_s]) if bits_s else ""
                                    rows.append(
                                        "<div style='display:flex; align-items:center; gap:10px; margin:6px 0;'>"
                                        f"<div style='width:52px; font-size:0.78rem; opacity:0.70;'>{_escape_html(nm)}</div>"
                                        + strip
                                        + f"<div style='width:96px; text-align:right; font-size:0.78rem; opacity:0.70; white-space:nowrap; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;'>{_escape_html(stat)}</div>"
                                        + "</div>"
                                    )
                                title = "<div style='font-size:0.78rem; opacity:0.70; margin-top:8px; margin-bottom:2px;'><b>How often the gate opens (train)</b> — Early / Mid / Late</div>"
                                note_txt = str(snap.get("snap_note") or "").strip()
                                if not note_txt:
                                    note_txt = "Frequency only (not a performance estimate)."
                                note = "<div style='font-size:0.74rem; opacity:0.60; margin-bottom:4px;'>" + _escape_html(note_txt) + "</div>"
                                snapshots_slot.markdown(title + note + "<div>" + "".join(rows) + "</div>", unsafe_allow_html=True)
                            else:
                                try:
                                    snapshots_slot.empty()
                                except Exception:
                                    pass
                        except Exception:
                            pass



                    except Exception:
                        pass

                    # Gate coverage diagnostics (sanity check, not performance)
                    with st.expander("Gate coverage (sanity check)", expanded=False):
                        st.caption("Sanity check: how often the gate is true on the train slice (first 70%). Not a performance estimate.")
                        st.caption("Too rare → low trade counts & high variability. Too frequent → behaves like no gate.")
                        if df_feat is None or df_feat.empty:
                            st.info("No dataset loaded, so coverage can’t be computed here.")
                        else:
                            try:
                                df_cov = df_feat
                                n = int(len(df_cov))
                                n_train = int(max(1, round(n * 0.70)))
                                df_cov = df_cov.iloc[:n_train].copy()

                                # Keep ATR median for other UI parts
                                if "atr_pct" in df_cov.columns:
                                    try:
                                        build_atr_med = float(np.nanmedian(pd.to_numeric(df_cov["atr_pct"], errors="coerce")))
                                    except Exception:
                                        pass

                                # Required column check (best-effort)
                                need = set()
                                def _add_need(name: Any) -> None:
                                    nm = str(name or "").strip()
                                    if not nm:
                                        return
                                    if nm.lower() in {"open", "high", "low", "close", "volume", "vol"}:
                                        return
                                    need.add(nm)

                                for c in (entry_logic.get("regime") or []):
                                    if isinstance(c, dict):
                                        _add_need(c.get("indicator") or c.get("feature") or c.get("lhs"))
                                        _add_need(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))
                                for cl in (entry_logic.get("clauses") or []):
                                    for c in (cl or []):
                                        if isinstance(c, dict):
                                            _add_need(c.get("indicator") or c.get("feature") or c.get("lhs"))
                                            _add_need(c.get("ref_indicator") or c.get("rhs") or c.get("rhs_indicator"))

                                missing = sorted([c for c in need if c not in df_cov.columns])
                                if missing:
                                    st.warning("Missing required fields for coverage: " + ", ".join(missing))
                                else:
                                    masks = _entry_logic_masks(df_cov, entry_logic)
                                    if masks is None:
                                        st.warning("Could not compute gate masks (missing columns or invalid condition).")
                                    else:
                                        r_mask, e_mask, c_mask = masks
                                        cov = float(c_mask.mean() * 100.0)

                                        # Bands (sanity)
                                        if cov < 5:
                                            band = "Rare"
                                        elif cov < 40:
                                            band = "Moderate"
                                        elif cov < 90:
                                            band = "Frequent"
                                        else:
                                            band = "Always-on"

                                        st.metric("Gate coverage", f"{cov:.1f}% ({band})")
                                        # Coverage snapshots (sanity check across time; not performance)
                                        try:
                                            snap_pct_ui = float(st.session_state.get('new.gate_snap_pct', 2.0) or 2.0)
                                            snap_window_pct = float(snap_pct_ui) / 100.0
                                            snap_max_bars = int(st.session_state.get('new.gate_snap_max_bars', 200) or 200)
                                            snap_max_bars = int(max(30, min(500, snap_max_bars)))
                                            w = _snap_window_len(len(c_mask), snap_window_pct, min_bars=1, max_bars=snap_max_bars)
                                            if w >= 5:
                                                c_arr = np.asarray(c_mask.values, dtype=bool)
                                                n_tr = int(len(c_arr))
                                                mid_center = n_tr // 2
                                                mid_start = int(max(0, min(n_tr - w, mid_center - (w // 2))))
                                                # Early = last w bars of the first third (avoids warmup bias)
                                                third_end = int(max(w, n_tr // 3))
                                                third_end = int(min(n_tr, third_end))
                                                early = c_arr[int(max(0, third_end - w)) : third_end]
                                        
                                                snaps = [
                                                    ('Early', early),
                                                    ('Mid', c_arr[mid_start:mid_start + w]),
                                                    ('Late', c_arr[-w:]),
                                                ]
                                        
                                                st.markdown('**How often the gate opens (train)**')
                                                note_txt = f"Window = {snap_pct_ui:.1f}% of train (capped at {snap_max_bars} bars) → {w} bars. Frequency only."
                                                st.caption(note_txt + ' Early = end of 1st third; Mid = center; Late = end.')
                                                s_cols = st.columns(3)
                                                for col, (nm, arr) in zip(s_cols, snaps):
                                                    arr_list = [bool(x) for x in arr]
                                                    tn = int(sum(1 for x in arr_list if x))
                                                    nn = int(len(arr_list))
                                                    pct = (tn / float(nn)) * 100.0 if nn else 0.0
                                                    strip_bits = _downsample_bool_bits(arr_list, max_dots=120)
                                                    strip_html = _gate_strip_html(strip_bits)
                                                    col.markdown(
                                                        f"<div style='border:1px solid rgba(49,51,63,0.10); border-radius:12px; padding:10px 10px 8px 10px; background:rgba(255,255,255,0.45);'>"
                                                        f"<div style='font-weight:650; font-size:0.86rem; margin-bottom:6px;'>{nm}</div>"
                                                        f"{strip_html}"
                                                        f"<div style='font-size:0.78rem; opacity:0.72; margin-top:6px;'>{tn}/{nn} bars true ({pct:.1f}%)</div>"
                                                        f"</div>",
                                                        unsafe_allow_html=True,
                                                    )
                                            else:
                                                st.caption('Not enough bars in the train slice for coverage snapshots.')
                                        except Exception:
                                            pass

                                        if cov < 2:
                                            st.warning("Very rare gate (<2%): expect low trade counts and high variability.")
                                        if cov > 95:
                                            st.info("Gate is almost always true (>95%): this behaves like 'no gate' most of the time.")

                                        cols = st.columns(2)

                                        with cols[0]:
                                            st.markdown("**Regime blockers (most often false)**")
                                            reg = entry_logic.get("regime") or []
                                            if not reg:
                                                st.caption("No regime filters.")
                                            else:
                                                rows = []
                                                for c in reg:
                                                    cm = _cond_mask(df_cov, c)
                                                    if cm is None:
                                                        continue
                                                    fail = float((1.0 - float(cm.mean())) * 100.0)
                                                    rows.append((fail, _human_condition(c)))
                                                rows = sorted(rows, reverse=True)[:5]
                                                if rows:
                                                    for fail, label in rows:
                                                        st.write(f"- {label} — false {fail:.0f}%")
                                                else:
                                                    st.caption("Could not compute blockers (missing data).")

                                        with cols[1]:
                                            st.markdown("**Clause coverage (how often each clause is true)**")
                                            cls = entry_logic.get("clauses") or []
                                            if not cls:
                                                st.caption("No clauses (always allowed).")
                                            else:
                                                for i, cl in enumerate(cls):
                                                    cm_all = pd.Series(True, index=df_cov.index)
                                                    ok = True
                                                    for c in (cl or []):
                                                        cm = _cond_mask(df_cov, c)
                                                        if cm is None:
                                                            ok = False
                                                            break
                                                        cm_all &= cm
                                                    if not ok:
                                                        st.write(f"- Clause {chr(65+i)}: (missing data)")
                                                    else:
                                                        pct = float(cm_all.mean() * 100.0)
                                                        st.write(f"- Clause {chr(65+i)}: {pct:.1f}% true")
                            except Exception:
                                st.warning("Coverage diagnostics failed (unexpected input).")

            # Entry gate summary (right-aligned)
            if str(entry_mode).startswith("Simple"):
                gate_lbl = {
                    "none": "No gate",
                    "below_ema": f"EMA{ema_len}",
                    "rsi_below": f"RSI≤{rsi_thr:.0f}",
                    "bb_z_below": f"BB z≤{bb_z_thr:.1f}",
                    "macd_bull": "MACD hist",
                    "adx_above": f"ADX≥{adx_thr:.0f}",
                    "donch_pos_below": "Donchian low",
                }.get(str(buy_filter), "Custom")
            else:
                try:
                    r_n = len(entry_logic.get("regime") or [])
                    c_n = len(entry_logic.get("clauses") or [])
                except Exception:
                    r_n, c_n = 0, 0
                gate_lbl = f"Builder ({r_n}R/{c_n}C)"
            gate_sum.markdown(f"<div style='text-align:right; font-weight:700'>{gate_lbl}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # Allocation module
        # -------------------------
        st.markdown("<div class='ff-module ff-module-alloc " + ("active" if active_module == "allocation" else "") + "'>", unsafe_allow_html=True)
        with st.container():
            top_a, top_b = st.columns([0.62, 0.38])
            with top_a:
                st.markdown("**Allocation**")
                st.caption("Caps how much equity can be allocated at once.")
            with top_b:
                alloc_sum = st.empty()

            with st.expander("Configure", expanded=(active_module == "allocation")):
                max_alloc_pct = float(
                    st.slider(
                        "Max allocation (fraction of equity)",
                        min_value=0.05,
                        max_value=1.00,
                        value=1.00,
                        step=0.05,
                        key="new.max_alloc_pct",
                    )
                )

            alloc_sum.markdown(f"<div style='text-align:right; font-weight:700'>{_fmt_pct(float(max_alloc_pct), digits=0)}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------
        # Risk controls module
        # -------------------------
        st.markdown("<div class='ff-module ff-module-risk " + ("active" if active_module == "risk" else "") + "'>", unsafe_allow_html=True)
        with st.container():
            top_a, top_b = st.columns([0.62, 0.38])
            with top_a:
                st.markdown("**Risk controls**")
                st.caption("Optional exits. Values are experiment knobs (not recommendations).")
            with top_b:
                risk_sum = st.empty()

            with st.expander("Configure", expanded=(active_module == "risk")):
                if build_atr_med is not None and math.isfinite(build_atr_med):
                    st.caption(f"For scale only: median daily ATR ≈ {build_atr_med:.1f}% on this dataset.")

                # Sub-skill expanders: open the focused one if a modifier node was clicked.
                focus = active_mod if (active_module == "risk") else ""
                exp_sl = (focus == "stop_loss")
                exp_tp = (focus == "take_profit")
                exp_time = (focus == "time_stop")
                exp_trail = (focus == "trailing")

                with st.expander("Stop loss", expanded=exp_sl):
                    sl_ui = float(st.slider("Stop loss (%) (0 disables)", 0.0, 95.0, 0.0, 0.25, key="new.sl_pct_ui"))

                with st.expander("Take profit", expanded=exp_tp):
                    tp_ui = float(st.slider("Take profit (%) (0 disables)", 0.0, 500.0, 0.0, 0.5, key="new.tp_pct_ui"))
                    if tp_ui > 0:
                        tp_sell_fraction = float(st.slider("On take profit: sell fraction of position", 0.0, 1.0, 1.0, 0.05, key="new.tp_sell_frac"))
                        reserve_frac = float(st.slider("Reserve fraction of TP proceeds (keep as cash)", 0.0, 1.0, 0.0, 0.05, key="new.reserve_frac"))
                    else:
                        tp_sell_fraction = 1.0
                        reserve_frac = 0.0

                with st.expander("Time stop", expanded=exp_time):
                    max_hold_bars = int(st.number_input("Max holding period (bars) (0 disables)", min_value=0, value=0, step=5, key="new.max_hold_bars"))

                with st.expander("Trailing stop", expanded=exp_trail):
                    trail_ui = float(st.slider("Trailing stop (%) from peak (0 disables)", 0.0, 95.0, 0.0, 0.25, key="new.trail_pct_ui"))

                # Scale captions (if ATR exists)
                try:
                    if build_atr_med is not None and math.isfinite(build_atr_med) and build_atr_med > 0:
                        if sl_ui > 0:
                            st.caption(f"Stop loss scale: {sl_ui:.2f}% ≈ {sl_ui / build_atr_med:.1f}× median ATR.")
                        if tp_ui > 0:
                            st.caption(f"Take profit scale: {tp_ui:.2f}% ≈ {tp_ui / build_atr_med:.1f}× median ATR.")
                        if trail_ui > 0:
                            st.caption(f"Trailing scale: {trail_ui:.2f}% ≈ {trail_ui / build_atr_med:.1f}× median ATR.")
                except Exception:
                    pass

                sl_pct = float(sl_ui) / 100.0
                tp_pct = float(tp_ui) / 100.0
                trail_pct = float(trail_ui) / 100.0

            exits = []
            if float(sl_pct) > 0:
                exits.append(f"SL {sl_pct*100:.1f}%")
            if float(tp_pct) > 0:
                exits.append(f"TP {tp_pct*100:.1f}%")
            if int(max_hold_bars) > 0:
                exits.append(f"Time {int(max_hold_bars)}")
            if float(trail_pct) > 0:
                exits.append(f"Trail {trail_pct*100:.1f}%")
            risk_label = ", ".join(exits) if exits else "None"
            risk_sum.markdown(f"<div style='text-align:right; font-weight:700'>{risk_label}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Params (schema unchanged)
    params = {
        "deposit_freq": deposit_freq,
        "deposit_amount_usd": float(deposit_amount),
        "buy_freq": buy_freq,
        "buy_amount_usd": float(buy_amount),
        "buy_mode": str(buy_mode),
        "max_buys_per_gate": int(max_buys_per_gate),
        "buy_filter": buy_filter,  # legacy (still used by grid/back-compat)
        "ema_len": int(ema_len),
        "rsi_thr": float(rsi_thr),
        "macd_hist_thr": float(macd_hist_thr),
        "bb_z_thr": float(bb_z_thr),
        "adx_thr": float(adx_thr),
        "donch_pos_thr": float(donch_pos_thr),
        "entry_logic": entry_logic,  # new
        "max_alloc_pct": float(max_alloc_pct),
        "sl_pct": float(sl_pct),
        "tp_pct": float(tp_pct),
        "tp_sell_fraction": float(tp_sell_fraction),
        "reserve_frac_of_proceeds": float(reserve_frac),
        "max_hold_bars": int(max_hold_bars),
        "trail_pct": float(trail_pct),
    }

    # Human-readable build summary
    parts: List[str] = []
    if str(buy_mode).strip().lower() == "signal":
        _mb = int(max_buys_per_gate or 0)
        _lim = f" (max {_mb} buys per signal window)" if _mb > 0 else ""
        parts.append(f"Deposit {deposit_freq} ${int(round(params['deposit_amount_usd']))} and buy ≤{params['buy_freq']} ${int(round(params['buy_amount_usd']))} while the gate is true{_lim}.")
    else:
        parts.append(f"Deposit {deposit_freq} ${int(round(params['deposit_amount_usd']))} and buy {params['buy_freq']} ${int(round(params['buy_amount_usd']))} on schedule.")
    if entry_mode.startswith("Simple"):
        if buy_filter == "below_ema":
            parts.append(f"Only buy if close ≤ EMA({params['ema_len']}).")
        elif buy_filter == "rsi_below":
            parts.append(f"Only buy if RSI(14) ≤ {params['rsi_thr']:.0f}.")
        elif buy_filter == "bb_z_below":
            parts.append(f"Only buy if BB z-score(20) ≤ {params['bb_z_thr']:.1f}.")
        elif buy_filter == "macd_bull":
            parts.append(f"Only buy if MACD histogram ≥ {params['macd_hist_thr']:.2f}.")
        elif buy_filter == "adx_above":
            parts.append(f"Only buy if ADX(14) ≥ {params['adx_thr']:.0f}.")
        elif buy_filter == "donch_pos_below":
            parts.append(f"Only buy near range bottom (Donchian pos ≤ {params['donch_pos_thr']:.2f}).")
        else:
            parts.append("Entry gate: (none).")
    else:
        parts.append(_human_entry_logic(entry_logic) or "Entry logic: (none).")

    parts.append(f"Never allocate more than {_fmt_pct(params['max_alloc_pct'], digits=0)} of equity.")

    if params.get("sl_pct", 0.0) > 0:
        parts.append("Stop loss: {:.2f}%.".format(params["sl_pct"] * 100))
    if params.get("tp_pct", 0.0) > 0:
        parts.append("Take profit: {:.2f}% (sell {}% per hit).".format(params["tp_pct"] * 100, int(round(params["tp_sell_fraction"] * 100))))
    if params.get("max_hold_bars", 0) > 0:
        parts.append(f"Time stop: exit after {params['max_hold_bars']} bars in-position.")
    if params.get("trail_pct", 0.0) > 0:
        parts.append("Trailing stop: {:.2f}% from peak (ratchets up).".format(params["trail_pct"] * 100))

    summary_text = " ".join([p for p in parts if p.strip()])

    # Build label + completeness (neutral, non-advice)
    gate_label = "No gate"
    if entry_mode.startswith("Simple"):
        gate_label = {
            "none": "No gate",
            "below_ema": f"Gate: EMA{ema_len}",
            "rsi_below": f"Gate: RSI≤{rsi_thr:.0f}",
            "bb_z_below": f"Gate: BB z≤{bb_z_thr:.1f}",
            "macd_bull": "Gate: MACD hist",
            "adx_above": f"Gate: ADX≥{adx_thr:.0f}",
            "donch_pos_below": "Gate: Donchian low",
        }.get(buy_filter, "Gate: custom")
    else:
        # Builder mode: show counts only
        try:
            r_n = len(entry_logic.get("regime") or [])
            c_n = len(entry_logic.get("clauses") or [])
        except Exception:
            r_n, c_n = 0, 0
        gate_label = f"Gate: builder ({r_n} regime, {c_n} clause)"

    build_label = f"{buy_freq.capitalize()} DCA · {gate_label}"

    total_slots = 5
    slots = 0
    if deposit_freq != "none" and float(deposit_amount) > 0:
        slots += 1
    if float(buy_amount) > 0:
        slots += 1
    if (entry_mode.startswith("Simple") and buy_filter != "none") or (not entry_mode.startswith("Simple") and (entry_logic.get("clauses") or [])):
        slots += 1
    if float(max_alloc_pct) < 1.0:
        slots += 1
    if (sl_pct > 0) or (tp_pct > 0) or (max_hold_bars > 0) or (trail_pct > 0):
        slots += 1

    completeness = slots / float(total_slots) if total_slots else 0.0

    # Fill header + right preview
    with header_slot.container():
        with st.container():
            st.caption("Build completeness")
            st.progress(completeness)
            st.caption(f"{slots}/{total_slots} modules configured")


            # --- Phase 4: Build manifest + exports (read-only) ---
            with st.expander("Build manifest (read-only)", expanded=False):
                # Baseline config that can be fed into the engine (spot, long-only).
                cfg_out = {
                    "strategy_name": str(st.session_state.get("new.strategy_name") or "dca_swing"),
                    "side": "long",
                    "market_mode": "spot",
                    "params": params,
                }

                # Extra metadata for auditability (no performance claims).
                manifest = {
                    "build_label": build_label,
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "notes": "Mechanics-only configuration. Spot only. Read-only build page.",
                    "cfg": cfg_out,
                    "ui_context": {
                        "template": str(st.session_state.get("new.template") or "DCA/Swing"),
                        "data_path": st.session_state.get("new.data_path"),
                    },
                }

                st.caption("These exports describe mechanics only. They do not predict performance and are not trading advice.")

                st.markdown("**Summary text**")
                st.code(summary_text)

                st.markdown("**Baseline config (JSON)**")
                st.code(json.dumps(cfg_out, indent=2), language="json")

                st.markdown("**Build manifest (JSON)**")
                st.code(json.dumps(manifest, indent=2), language="json")

                dl1, dl2, dl3 = st.columns(3, gap="small")
                with dl1:
                    st.download_button(
                        "Download baseline config",
                        data=json.dumps(cfg_out, indent=2),
                        file_name="baseline_config.json",
                        mime="application/json",
                    )
                with dl2:
                    st.download_button(
                        "Download manifest",
                        data=json.dumps(manifest, indent=2),
                        file_name="build_manifest.json",
                        mime="application/json",
                    )
                with dl3:
                    st.download_button(
                        "Download summary",
                        data=summary_text,
                        file_name="build_summary.txt",
                        mime="text/plain",
                    )
    return params


def _write_baseline_json(tmp_dir: Path, *, strategy_name: str, side: str, params: Dict[str, Any]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"baseline_{_now_slug()}.json"
    cfg = {
        "strategy_name": strategy_name,
        "side": side,
        "params": params,
    }
    p.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return p


def _read_json(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
            f.write('\n')


# =============================================================================
# Trust layer (Sprint 4): manifest + comparability + strategy pack export
# =============================================================================

MANIFEST_SCHEMA_VERSION = 2


def _utc_iso(ts: float) -> str:
    try:
        return datetime.utcfromtimestamp(float(ts)).replace(microsecond=0).isoformat() + "Z"
    except Exception:
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _try_git_head(repo_root: Path) -> Optional[str]:
    """Best-effort git commit hash for receipts."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8", errors="ignore").strip()
        return s if re.fullmatch(r"[0-9a-fA-F]{7,40}", s or "") else None
    except Exception:
        return None


def _fingerprint_file(path: Path, *, full_max_bytes: int = 50_000_000, sample_bytes: int = 1_000_000) -> Dict[str, Any]:
    """
    Compute a stable-ish dataset fingerprint.
    - Full sha256 for small files.
    - Head+tail sha256 for large files.
    """
    stt = path.stat()
    size = int(stt.st_size)
    mtime = float(stt.st_mtime)

    mode = "full" if size <= int(full_max_bytes) else "sample"
    h = hashlib.sha256()

    with open(path, "rb") as f:
        if mode == "full":
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            head = f.read(int(sample_bytes))
            h.update(head)
            if size > int(sample_bytes):
                try:
                    f.seek(max(0, size - int(sample_bytes)))
                    tail = f.read(int(sample_bytes))
                    h.update(tail)
                except Exception:
                    pass

    fp: Dict[str, Any] = {
        "algo": "sha256",
        "mode": mode,
        "digest": h.hexdigest(),
        "size_bytes": size,
        "mtime_utc": _utc_iso(mtime),
    }
    if mode == "sample":
        fp["sample_bytes"] = int(sample_bytes)
    return fp





# -----------------------------------------------------------------------------
# Trust layer helpers (Sprint 5)
# -----------------------------------------------------------------------------
def _safe_relpath(path: Optional[Path], base: Path) -> Optional[str]:
    try:
        if path is None:
            return None
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return None


def _dataset_quick_meta(path: Path, *, max_scan_bytes: int = 200_000_000, tail_lines: int = 2000) -> Dict[str, Any]:
    """Best-effort dataset metadata for comparability (fast-ish, guarded for huge files)."""
    meta: Dict[str, Any] = {}
    try:
        stt = path.stat()
        if int(stt.st_size) > int(max_scan_bytes):
            meta["note"] = "Skipped deep scan (file too large)."
            return meta
    except Exception:
        return meta

    # Columns + schema hash + row count
    try:
        import csv

        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                meta["columns"] = header
                meta["schema_hash"] = hashlib.sha256(",".join(header).encode("utf-8")).hexdigest()
                cnt = 0
                for _ in reader:
                    cnt += 1
                meta["row_count"] = int(cnt)
    except Exception:
        pass

    # Time range hint (head/tail sampling)
    try:
        head: List[str] = []
        tail: deque[str] = deque(maxlen=int(tail_lines))
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i < 2000:
                    head.append(line)
                tail.append(line)

        cols = meta.get("columns") or []
        cand = None
        for c in ["ts", "timestamp", "dt", "date", "datetime"]:
            if c in cols:
                cand = c
                break

        if cand:
            def _parse(lines: List[str]) -> Tuple[Optional[float], Optional[float]]:
                if not lines:
                    return (None, None)
                try:
                    df = pd.read_csv(io.StringIO("".join(lines)))
                    if cand not in df.columns or df.empty:
                        return (None, None)
                    ser = df[cand]
                    if cand in ("ts", "timestamp"):
                        v = pd.to_numeric(ser, errors="coerce").dropna()
                        if v.empty:
                            return (None, None)
                        return (float(v.min()), float(v.max()))
                    v = pd.to_datetime(ser, errors="coerce", utc=True).dropna()
                    if v.empty:
                        return (None, None)
                    return (float(v.min().timestamp()), float(v.max().timestamp()))
                except Exception:
                    return (None, None)

            h0, h1 = _parse(head)
            t0, t1 = _parse(list(tail))
            xs = [x for x in [h0, h1, t0, t1] if x is not None]
            if xs:
                meta["time_range_hint_epoch"] = {"min": min(xs), "max": max(xs), "column": cand}
    except Exception:
        pass

    return meta


def _tests_signature(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tests = (manifest or {}).get("tests") or {}
    rs_runs = tests.get("rolling_starts") or []
    wf_runs = tests.get("walkforward") or []

    def _pick_rs(m: Dict[str, Any]) -> Dict[str, Any]:
        return {"start_step": m.get("start_step"), "min_bars": m.get("min_bars"), "top_n": m.get("top_n"), "windows_per_cfg": m.get("windows_per_cfg")}

    def _pick_wf(m: Dict[str, Any]) -> Dict[str, Any]:
        return {"window_days": m.get("window_days"), "step_days": m.get("step_days"), "min_bars": m.get("min_bars"), "top_n": m.get("top_n"), "windows": m.get("windows")}

    rs_sigs = []
    for r in rs_runs:
        if isinstance(r, dict):
            rs_sigs.append(_pick_rs((r.get("meta") or {})))
    wf_sigs = []
    for r in wf_runs:
        if isinstance(r, dict):
            wf_sigs.append(_pick_wf((r.get("meta") or {})))

    def _uniq(xs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for x in xs:
            key = json.dumps(x, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(x)
        return out

    return {"rolling_starts": _uniq(rs_sigs), "walkforward": _uniq(wf_sigs)}


def _compare_manifests(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    da = ((a or {}).get("dataset") or {}).get("fingerprint") or {}
    db = ((b or {}).get("dataset") or {}).get("fingerprint") or {}
    if str(da.get("digest") or "") and str(db.get("digest") or "") and str(da.get("digest")) != str(db.get("digest")):
        warns.append("Dataset fingerprints differ between runs (results are not directly comparable).")

    ga = str((a or {}).get("engine_git_head") or "")
    gb = str((b or {}).get("engine_git_head") or "")
    if ga and gb and ga != gb:
        warns.append("Engine git differs between runs (behavior may differ).")

    sa = _tests_signature(a)
    sb = _tests_signature(b)
    if json.dumps(sa.get("rolling_starts"), sort_keys=True) != json.dumps(sb.get("rolling_starts"), sort_keys=True):
        warns.append("Rolling Starts parameters differ between runs.")
    if json.dumps(sa.get("walkforward"), sort_keys=True) != json.dumps(sb.get("walkforward"), sort_keys=True):
        warns.append("Walkforward parameters differ between runs.")

    return warns


def _update_runs_index(runs_root: Path) -> None:
    """Maintain runs/_index.json for quick browsing + cross-run compare."""
    items: List[Dict[str, Any]] = []
    try:
        for d in sorted([p for p in runs_root.glob("batch_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            mp = d / "manifest.json"
            if not mp.exists():
                continue
            m = _read_json(mp)
            ds = (m.get("dataset") or {}) if isinstance(m, dict) else {}
            fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
            dig = str(fp.get("digest") or "")
            items.append(
                {
                    "run_id": d.name,
                    "created_at": m.get("created_at") if isinstance(m, dict) else _utc_iso(d.stat().st_mtime),
                    "dataset_digest": dig,
                    "dataset_digest_short": (dig[:10] + "…" + dig[-6:]) if dig else "",
                    "engine_git_head": (m.get("engine_git_head") if isinstance(m, dict) else None) or "",
                    "tests": _tests_signature(m if isinstance(m, dict) else {}),
                }
            )
    except Exception:
        pass

    try:
        out = {"schema_version": 1, "updated_at": _utc_iso(time.time()), "items": items}
        _write_json(runs_root / "_index.json", out)
    except Exception:
        pass
def _scan_test_runs(run_dir: Path) -> Dict[str, Any]:
    """Scan RS/WF output folders and collect meta for receipts + warnings.

    Supports:
      - RS: run_dir/rolling_starts/{files} OR run_dir/rolling_starts/rs_*/{files}
      - WF: run_dir/walkforward_*/{files} OR run_dir/walkforward/wf_*/{files} OR run_dir/walkforward/{files}
    """
    rs_root = run_dir / "rolling_starts"
    wf_root = run_dir / "walkforward"

    def _sorted_dirs(xs: List[Path]) -> List[Path]:
        xs = [p for p in xs if p is not None and p.exists()]
        return sorted(xs, key=lambda p: p.stat().st_mtime)

    rs_dirs: List[Path] = []
    if rs_root.exists():
        # Direct (CLI default)
        if (rs_root / "rolling_starts_summary.csv").exists() or (rs_root / "rolling_starts_detail.csv").exists():
            rs_dirs.append(rs_root)
        # UI subfolders
        rs_dirs.extend([p for p in rs_root.glob("rs_*") if p.is_dir()])

    rs_runs: List[Dict[str, Any]] = []
    for d in _sorted_dirs(rs_dirs):
        rs_runs.append(
            {
                "dir": str(d),
                "meta": _read_json(d / "rs_meta.json") if (d / "rs_meta.json").exists() else {},
                "summary": str(d / "rolling_starts_summary.csv") if (d / "rolling_starts_summary.csv").exists() else None,
                "detail": str(d / "rolling_starts_detail.csv") if (d / "rolling_starts_detail.csv").exists() else None,
                "mtime_utc": _utc_iso(d.stat().st_mtime),
            }
        )

    wf_dirs: List[Path] = []
    # CLI default: walkforward_* at run root
    wf_dirs.extend([p for p in run_dir.glob("walkforward_*") if p.is_dir()])
    if wf_root.exists():
        # Legacy direct files
        if (wf_root / "wf_summary.csv").exists() or (wf_root / "wf_results.csv").exists():
            wf_dirs.append(wf_root)
        # UI subfolders
        wf_dirs.extend([p for p in wf_root.glob("wf_*") if p.is_dir()])

    wf_runs: List[Dict[str, Any]] = []
    for d in _sorted_dirs(wf_dirs):
        wf_runs.append(
            {
                "dir": str(d),
                "meta": _read_json(d / "wf_meta.json") if (d / "wf_meta.json").exists() else {},
                "summary": str(d / "wf_summary.csv") if (d / "wf_summary.csv").exists() else None,
                "results": str(d / "wf_results.csv") if (d / "wf_results.csv").exists() else None,
                "mtime_utc": _utc_iso(d.stat().st_mtime),
            }
        )

    return {"rolling_starts": rs_runs, "walkforward": wf_runs}


def _build_manifest(
    run_dir: Path,
    *,
    compute_fingerprint: bool = True,
    existing_fp: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = _read_json(run_dir / "batch_meta.json") if (run_dir / "batch_meta.json").exists() else {}
    data_path = Path(str(meta.get("data") or meta.get("data_path") or "")).expanduser()
    dataset: Dict[str, Any] = {
        "path_abs": str(data_path) if str(data_path) else None,
        "path_rel_to_repo": _safe_relpath(data_path, REPO_ROOT) if str(data_path) else None,
        "basename": data_path.name if str(data_path) else None,
    }

    if str(data_path) and data_path.exists():
        # Avoid re-hashing on every UI rerun if an existing fingerprint is still valid.
        if (not compute_fingerprint) and isinstance(existing_fp, dict) and existing_fp:
            dataset["fingerprint"] = existing_fp
        else:
            dataset["fingerprint"] = _fingerprint_file(data_path)

        # Extra metadata for comparability (best-effort; guarded for huge files)
        try:
            dataset["meta"] = _dataset_quick_meta(data_path)
        except Exception:
            dataset["meta"] = {}

    created_guess = None
    # Try parse run folder name: batch_YYYYMMDD_HHMMSS_...
    m = re.match(r"batch_(\d{8})_(\d{6})_", run_dir.name)
    if m:
        try:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            created_guess = dt.replace(tzinfo=None).isoformat() + "Z"
        except Exception:
            created_guess = None

    app_fp = None
    try:
        app_fp = _fingerprint_file(Path(__file__).resolve(), full_max_bytes=5_000_000)
    except Exception:
        app_fp = None

    run_dir_abs = run_dir.resolve()
    manifest: Dict[str, Any] = {
        "schema_version": int(MANIFEST_SCHEMA_VERSION),
        "run_id": str(run_dir.name),
        "created_at": created_guess or _utc_iso(run_dir.stat().st_mtime),
        "repo_root": str(REPO_ROOT),
        "run_dir": {"abs": str(run_dir_abs), "rel_to_repo": _safe_relpath(run_dir_abs, REPO_ROOT)},
        "engine_git_head": _try_git_head(REPO_ROOT),
        "app_file_fingerprint": app_fp,
        "dataset": dataset,
        "batch_meta": meta,
        "tests": _scan_test_runs(run_dir),
    }
    return manifest
def _ensure_manifest(run_dir: Path) -> Dict[str, Any]:
    """Create/refresh manifest.json. Safe for old runs; avoids re-hashing unchanged datasets."""
    path = run_dir / "manifest.json"
    try:
        existing = _read_json(path) if path.exists() else {}
    except Exception:
        existing = {}

    # Decide whether we need to recompute the dataset hash (can be expensive for big CSVs).
    compute_fp = True
    existing_fp = None
    try:
        ds = (existing or {}).get("dataset") or {}
        existing_fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else None
        p = ds.get("path_abs") or ds.get("path")
        if p and isinstance(existing_fp, dict) and existing_fp:
            data_path = Path(str(p)).expanduser()
            if data_path.exists():
                stt = data_path.stat()
                cur_size = int(stt.st_size)
                cur_mtime = _utc_iso(float(stt.st_mtime))
                rec_size = int(existing_fp.get("size_bytes", -1))
                rec_mtime = str(existing_fp.get("mtime_utc") or "")
                if (rec_size != -1 and cur_size == rec_size) and (rec_mtime and cur_mtime == rec_mtime):
                    compute_fp = False
    except Exception:
        compute_fp = True
        existing_fp = None

    new = _build_manifest(run_dir, compute_fingerprint=compute_fp, existing_fp=existing_fp)

    # If an existing manifest exists, keep any unknown top-level keys.
    merged = dict(existing or {})
    merged.update(new)

    try:
        _write_json(path, merged)
    except Exception:
        # If writing fails (permissions), still return what we built.
        return merged

    # Update runs index (best-effort)
    try:
        _update_runs_index(REPO_ROOT / "runs")
    except Exception:
        pass

    return merged
def _comparability_warnings(manifest: Dict[str, Any]) -> List[str]:
    warns: List[str] = []

    ds = (manifest or {}).get("dataset") or {}
    p = ds.get("path_abs") or ds.get("path")
    fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}

    if not p:
        warns.append("No dataset path recorded in manifest. Comparability is weaker.")
        return warns

    data_path = Path(str(p)).expanduser()
    if not data_path.exists():
        warns.append("Dataset file no longer exists at the recorded path. Comparability is weaker.")
        return warns

    # Quick check: size + mtime
    try:
        stt = data_path.stat()
        cur_size = int(stt.st_size)
        cur_mtime = _utc_iso(float(stt.st_mtime))
        rec_size = int(fp.get("size_bytes", -1)) if isinstance(fp, dict) else -1
        rec_mtime = str(fp.get("mtime_utc") or "")
        drift = False
        if rec_size != -1 and cur_size != rec_size:
            warns.append("Dataset size differs from the recorded fingerprint (file may have changed).")
            drift = True
        if rec_mtime and cur_mtime != rec_mtime:
            warns.append("Dataset modification time differs from the recorded fingerprint (file may have changed).")
            drift = True
        # Only compute a hash comparison if cheap checks suggest drift.
        if drift:
            cur_fp = _fingerprint_file(data_path)
            rec_digest = str(fp.get("digest") or "")
            if rec_digest and str(cur_fp.get("digest")) != rec_digest:
                warns.append("Dataset fingerprint digest does not match (you are not running on the same data).")
    except Exception:
        warns.append("Could not validate dataset fingerprint. Comparability is weaker.")

    # Multiple RS/WF parameter sets
    tests = (manifest or {}).get("tests") or {}
    rs_runs = tests.get("rolling_starts") or []
    wf_runs = tests.get("walkforward") or []

    def _rs_sig(m: Dict[str, Any]) -> str:
        return f"step={m.get('start_step')}|min={m.get('min_bars')}|top_n={m.get('top_n')}|wins={m.get('windows_per_cfg')}"

    def _wf_sig(m: Dict[str, Any]) -> str:
        return f"win={m.get('window_days')}|step={m.get('step_days')}|min={m.get('min_bars')}|top_n={m.get('top_n')}|wins={m.get('windows')}"

    try:
        rs_sigs = {_rs_sig((r.get("meta") or {})) for r in rs_runs if isinstance(r, dict)}
        rs_sigs = {s for s in rs_sigs if "None" not in s}
        if len(rs_sigs) > 1:
            warns.append("Multiple Rolling Starts evidence sets exist with different parameters. Verdicts depend on which evidence set is used.")
    except Exception:
        pass

    try:
        wf_sigs = {_wf_sig((r.get("meta") or {})) for r in wf_runs if isinstance(r, dict)}
        wf_sigs = {s for s in wf_sigs if "None" not in s}
        if len(wf_sigs) > 1:
            warns.append("Multiple Walkforward evidence sets exist with different parameters. Verdicts depend on which evidence set is used.")
    except Exception:
        pass

    return warns


def _zip_add_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    zf.writestr(arcname, data)


def _zip_add_file(zf: zipfile.ZipFile, file_path: Path, arcname: str) -> None:
    try:
        zf.write(str(file_path), arcname=arcname)
    except Exception:
        # fallback: read bytes
        try:
            _zip_add_bytes(zf, arcname, file_path.read_bytes())
        except Exception:
            pass


def _build_strategy_pack_zip(
    *,
    run_dir: Path,
    run_name: str,
    config_id: str,
    manifest: Dict[str, Any],
    candidate_row: Dict[str, Any],
    cfg_norm: Dict[str, Any],
    rs_dir: Optional[Path],
    wf_dir: Optional[Path],
    top_art_dir: Optional[Path],
    include_replay: bool = True,
    include_dataset: bool = False,
) -> bytes:
    """Strategy Pack v2: structured, portable, verifiable."""
    buf = io.BytesIO()
    index: Dict[str, Any] = {"pack_version": 2, "created_at": _utc_iso(time.time()), "files": {}}

    def _sha256_bytes(b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def add_bytes(arc: str, b: bytes) -> None:
        _zip_add_bytes(zf, arc, b)
        index["files"][arc] = {"algo": "sha256", "digest": _sha256_bytes(b), "size_bytes": int(len(b))}

    def add_file(fp: Path, arc: str, *, hash_limit_bytes: int = 50_000_000) -> None:
        try:
            _zip_add_file(zf, fp, arc)
        except Exception:
            return
        try:
            size = int(fp.stat().st_size)
            digest = ""
            if size <= int(hash_limit_bytes):
                h = hashlib.sha256()
                with fp.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                digest = h.hexdigest()
            index["files"][arc] = {"algo": "sha256", "digest": digest, "size_bytes": size}
        except Exception:
            pass

    # Portable pack manifest (no absolute paths)
    ds = (manifest.get("dataset") or {}) if isinstance(manifest, dict) else {}
    ds_fp = (ds.get("fingerprint") or {}) if isinstance(ds, dict) else {}
    ds_meta = (ds.get("meta") or {}) if isinstance(ds, dict) else {}

    pack_manifest: Dict[str, Any] = {
        "schema_version": 2,
        "pack_version": 2,
        "source_run_id": str(run_name),
        "created_at": _utc_iso(time.time()),
        "engine_git_head": (manifest.get("engine_git_head") if isinstance(manifest, dict) else None) or "",
        "app_file_fingerprint": (manifest.get("app_file_fingerprint") if isinstance(manifest, dict) else None) or {},
        "dataset": {
            "basename": ds.get("basename") or (Path(str(ds.get("path_abs") or ds.get("path") or "")).name if (ds.get("path_abs") or ds.get("path")) else None),
            "fingerprint": ds_fp,
            "meta": ds_meta,
            "included_in_pack": bool(include_dataset),
        },
        "selected_config_id": str(config_id),
        "tests_signature": _tests_signature(manifest if isinstance(manifest, dict) else {}),
        "notes": "Portable strategy pack. Absolute paths removed; verify dataset via fingerprint.",
    }

    # README
    readme = f"""# Strategy Pack (v2)

This bundle contains receipts for a single strategy config.

- Source run: `{run_name}`
- Config: `{str(config_id)}`
- Engine git: `{pack_manifest.get('engine_git_head','')[:12]}`

## Verify
1. Use the in-app verifier to validate file hashes.
2. To verify your dataset, compare its fingerprint digest to `manifest.json -> dataset -> fingerprint -> digest`.

"""

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_bytes("README.md", readme.encode("utf-8"))
        add_bytes("manifest.json", json.dumps(pack_manifest, indent=2, ensure_ascii=False).encode("utf-8"))

        # Configs + evidence
        add_bytes("config/config_normalized.json", json.dumps(cfg_norm or {}, indent=2, ensure_ascii=False).encode("utf-8"))
        add_bytes("evidence/candidate_row.json", json.dumps(candidate_row or {}, indent=2, ensure_ascii=False).encode("utf-8"))

        # Resolved config line
        try:
            cfg_line = next((r for r in _load_jsonl(run_dir / "configs_resolved.jsonl") if str(r.get("config_id")) == str(config_id)), None)
            if cfg_line is not None:
                add_bytes("config/config_resolved.json", json.dumps(cfg_line, indent=2, ensure_ascii=False).encode("utf-8"))
        except Exception:
            pass

        # Batch meta
        if (run_dir / "batch_meta.json").exists():
            add_file(run_dir / "batch_meta.json", "meta/batch_meta.json")

        # RS evidence (filtered)
        if rs_dir is not None and rs_dir.exists():
            if (rs_dir / "rs_meta.json").exists():
                add_file(rs_dir / "rs_meta.json", "meta/rolling_starts/rs_meta.json")
            try:
                s = rs_dir / "rolling_starts_summary.csv"
                d = rs_dir / "rolling_starts_detail.csv"
                if s.exists():
                    sdf = pd.read_csv(s)
                    sdf["config_id"] = sdf["config_id"].astype(str).str.strip()
                    sdf1 = sdf[sdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/rolling_starts/summary_row.csv", sdf1.to_csv(index=False).encode("utf-8"))
                if d.exists():
                    ddf = pd.read_csv(d)
                    ddf["config_id"] = ddf["config_id"].astype(str).str.strip()
                    ddf1 = ddf[ddf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/rolling_starts/detail_rows.csv", ddf1.to_csv(index=False).encode("utf-8"))
            except Exception:
                pass

        # WF evidence (filtered)
        if wf_dir is not None and wf_dir.exists():
            if (wf_dir / "wf_meta.json").exists():
                add_file(wf_dir / "wf_meta.json", "meta/walkforward/wf_meta.json")
            try:
                s = wf_dir / "wf_summary.csv"
                r = wf_dir / "wf_results.csv"
                if s.exists():
                    sdf = pd.read_csv(s)
                    sdf["config_id"] = sdf["config_id"].astype(str).str.strip()
                    sdf1 = sdf[sdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/walkforward/summary_row.csv", sdf1.to_csv(index=False).encode("utf-8"))
                if r.exists():
                    rdf = pd.read_csv(r)
                    rdf["config_id"] = rdf["config_id"].astype(str).str.strip()
                    rdf1 = rdf[rdf["config_id"] == str(config_id)].copy()
                    add_bytes("evidence/walkforward/window_rows.csv", rdf1.to_csv(index=False).encode("utf-8"))
            except Exception:
                pass

        # Replay/top artifacts
        if include_replay and top_art_dir is not None and top_art_dir.exists():
            for fp in top_art_dir.rglob("*"):
                if fp.is_file():
                    rel = fp.relative_to(top_art_dir).as_posix()
                    add_file(fp, f"artifacts/{rel}", hash_limit_bytes=10_000_000)

        # Optional: include dataset
        if include_dataset:
            try:
                ds_path = ds.get("path_abs") or ds.get("path")
                if ds_path:
                    p = Path(str(ds_path)).expanduser()
                    if p.exists():
                        add_file(p, f"dataset/{p.name}", hash_limit_bytes=200_000_000)
            except Exception:
                pass

        # Pack index last
        add_bytes("meta/pack_index.json", json.dumps(index, indent=2, ensure_ascii=False).encode("utf-8"))

    return buf.getvalue()
def _baseline_row_from_base_json(base_path: Path) -> Dict[str, Any]:
    base = _read_json(base_path)
    if 'params' not in base or not isinstance(base['params'], dict):
        raise ValueError("Baseline JSON must contain a 'params' object")
    row = {
        'strategy_name': base.get('strategy_name', 'dca_swing'),
        'side': base.get('side', 'long'),
        'params': dict(base['params']),
    }
    # Tag so the UI can identify the baseline row in batch outputs
    row['params']['__baseline__'] = True
    return row

def _ensure_grid_has_baseline(grid_path: Path, base_path: Path, *, total_n: int) -> None:
    """Prepend baseline row to an existing grid JSONL (deduping any existing baseline row)."""
    baseline_row = _baseline_row_from_base_json(base_path)
    rows: List[Dict[str, Any]] = []
    if grid_path.exists():
        rows = [r for r in _load_jsonl(grid_path) if isinstance(r, dict)]
    # Drop any existing baseline row(s) to avoid duplicates
    cleaned: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if isinstance(r.get('params'), dict) and r['params'].get('__baseline__'):
                continue
        except Exception:
            pass
        cleaned.append(r)
    out = [baseline_row, *cleaned]
    if int(total_n) > 0:
        out = out[: int(total_n)]
    _write_jsonl(grid_path, out)


# =============================================================================
# UI: left rail (runs + mode)
# =============================================================================

with st.sidebar:
    st.header("Runs")
    runs = _list_runs()
    run_names = [p.name for p in runs]
    run_dirs = {p.name: p for p in runs}
    run_is_complete = {name: _has_any_results(run_dirs[name]) for name in run_names}

    # Persist selection across reruns (prefer the latest complete run)
    if "selected_run" not in st.session_state:
        picked = ""
        for nm in run_names:
            if run_is_complete.get(nm):
                picked = nm
                break
        st.session_state["selected_run"] = picked or (run_names[0] if run_names else "")

    # Programmatic selection handoff (must happen BEFORE the widget is created)
    nxt = st.session_state.pop("ui.open_run_next", None)
    if nxt and nxt in run_names:
        st.session_state["selected_run"] = nxt

    open_existing = st.selectbox(
        "Open existing run",
        options=["(new run)"] + run_names,
        index=(1 + run_names.index(st.session_state["selected_run"]) if st.session_state["selected_run"] in run_names else 0),
        format_func=lambda nm: (nm if nm == "(new run)" else (nm + ("  (incomplete)" if not run_is_complete.get(nm, False) else ""))),
        key="ui.open_run",
    )

    if open_existing != "(new run)":
        st.session_state["selected_run"] = open_existing
    else:
        # Selecting "(new run)" should always drop you into the Build & Run flow.
        st.session_state["ui.section"] = "1) Build & Run"

    st.divider()
    st.header("App")

    st.session_state.setdefault("ui.debug", False)
    st.checkbox(
        "Debug (show commands & logs)",
        key="ui.debug",
        help="Off by default to keep the UI clean. Turn on to see raw subprocess logs and full commands.",
    )


    # Keep stage keys for internal routing ("Next →" buttons, etc.)
    STAGES = [
        ("A) Batch", "batch"),
        ("B) Rolling Starts", "rs"),
        ("C) Walkforward", "wf"),
        ("D) Grand Verdict", "grand"),
    ]
    stage_labels = [x[0] for x in STAGES]
    stage_keys = [x[1] for x in STAGES]

    st.session_state.setdefault("ui.stage", "batch")
    st.session_state.setdefault("ui.batch.scroll_to_inspect", False)

    # MVP navigation: two sections only
    SECTION_OPTS = ["1) Build & Run", "2) Results & Autopsy"]
    if "ui.section" not in st.session_state:
        st.session_state["ui.section"] = SECTION_OPTS[0] if open_existing == "(new run)" else SECTION_OPTS[1]

    if "ui.section_next" in st.session_state:
        st.session_state["ui.section"] = st.session_state.pop("ui.section_next")

    st.radio("Section", options=SECTION_OPTS, key="ui.section")
    st.caption("Build & Run = define strategy + run tests. Results & Autopsy = filter, compare, inspect.")

# =============================================================================
# New run wizard (when "(new run)" is selected)
# =============================================================================

if open_existing == "(new run)":
    # This section is only the "Build & Run" half of the MVP UI.
    if str(st.session_state.get("ui.section", "1) Build & Run")).startswith("2)"):
        st.info("Results require an existing run. Switch to **Build & Run** to create one.")
        st.stop()
    st.subheader("Create a new run")

    # Step state
    if "new.step" not in st.session_state:
        st.session_state["new.step"] = 0  # 0=data,1=plan,2=grid,3=batch

    NEW_STEPS = ["1) Data", "2) Baseline plan", "3) Stress scope", "4) Run batch"]
    step = int(st.session_state["new.step"])
    step = max(0, min(step, len(NEW_STEPS) - 1))

    st.progress((step + 1) / len(NEW_STEPS))
    st.write(f"**{NEW_STEPS[step]}**")

    # Shared: staging dir for temp files
    tmp_dir = TMP_DIR


    # -------------------------------------------------------------------------
    # Step 0: Data
    # -------------------------------------------------------------------------
    if step == 0:
        st.write("Choose the market data you want to test against (spot, daily OHLCV).")

        catalog = _get_dataset_catalog()
        use_counts = st.session_state.get("data.use_counts", {}) if isinstance(st.session_state.get("data.use_counts", {}), dict) else {}

        if not catalog:
            st.error("No datasets found.")
            st.caption("Add datasets under ./data (or ./data/datasets) or provide a catalog.json.")
        else:
            left, right = st.columns([1, 1], gap="large")

            # Normalize ids + build lookup
            by_id: Dict[str, Dict[str, Any]] = {}
            for d in catalog:
                if not isinstance(d, dict):
                    continue
                did = str(d.get("id") or "").strip()
                if not did:
                    # best-effort id fallback
                    p = _resolve_dataset_path(d) or Path(str(d.get("file_path") or d.get("path") or ""))
                    did = f"{str(d.get('symbol') or _infer_symbol_from_filename(str(p.name)))}:{str(p)}"
                    d["id"] = did
                by_id[did] = d

            with left:
                q = st.text_input("Search coins", key="new.data_search", placeholder="e.g., BTC, ETH, Solana…")
                sort_by = st.selectbox(
                    "Sort by",
                    ["Most used", "Alphabetical", "Longest history", "Newest added"],
                    index=0,
                    key="new.data_sort",
                )
                st.caption("Timeframe: Daily (v1)")

                filtered = _sort_filter_catalog(list(by_id.values()), q, sort_by, use_counts=use_counts)
                ids = [str(d.get("id")) for d in filtered if str(d.get("id") or "").strip()]

                if not ids:
                    st.info("No matches. Try clearing the search.")
                    st.session_state.pop("new.data_id", None)
                    st.session_state.pop("new.data_path", None)
                else:
                    cur = str(st.session_state.get("new.data_id") or "")
                    if cur not in ids:
                        cur = ids[0]

                    sel = st.selectbox(
                        "Dataset",
                        ids,
                        index=ids.index(cur),
                        format_func=lambda _id: _dataset_option_label(by_id.get(_id, {})),
                        key="new.data_id",
                    )

                    entry = by_id.get(str(sel), {})
                    data_path = _resolve_dataset_path(entry)
                    if data_path is not None and data_path.exists():
                        st.session_state["new.data_path"] = str(data_path)
                    else:
                        st.session_state.pop("new.data_path", None)
                        st.warning("Dataset file not found on disk.")

                    # small scan table (read-only)
                    try:
                        rows = []
                        for d in filtered[:80]:
                            rows.append(
                                {
                                    "Symbol": str(d.get("symbol") or "").upper(),
                                    "Start": _safe_dt_str(d.get("start_dt") or d.get("start") or d.get("start_date")),
                                    "End": _safe_dt_str(d.get("end_dt") or d.get("end") or d.get("end_date")),
                                    "Bars": d.get("rows") or d.get("n_rows") or d.get("bars") or "",
                                }
                            )
                        if rows:
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                    except Exception:
                        pass

            with right:
                sel_id = str(st.session_state.get("new.data_id") or "")
                entry = by_id.get(sel_id, {}) if sel_id else {}
                data_path_str = st.session_state.get("new.data_path")
                if data_path_str:
                    p = Path(str(data_path_str))
                    st.markdown("##### Preview")
                    st.caption(_dataset_option_label(entry))

                    df_preview = _load_any_df_tail(p, n=2500)
                    if df_preview is not None and not df_preview.empty:
                        st.caption(f"Rows (loaded): {len(df_preview):,}  Columns: {list(df_preview.columns)}")
                        if px is not None:
                            dt_col = None
                            for c in ["dt", "date", "datetime", "timestamp", "ts"]:
                                if c in df_preview.columns:
                                    dt_col = c
                                    break
                            if dt_col and "close" in df_preview.columns:
                                try:
                                    dfx = df_preview.copy()
                                    dfx[dt_col] = pd.to_datetime(dfx[dt_col], errors="coerce", utc=True)
                                    dfx = dfx.dropna(subset=[dt_col])
                                    fig = px.line(dfx, x=dt_col, y="close", title="Close (tail)")
                                    _plotly(fig)
                                except Exception:
                                    pass
                    else:
                        st.warning("Could not preview dataset (parse failed).")
                else:
                    st.info("Pick a dataset to see a preview.")

        colL, colR = st.columns(2)
        with colL:
            if st.button("Next →", type="primary", disabled=("new.data_path" not in st.session_state)):
                # track usage (helps 'Most used' sort)
                try:
                    did = str(st.session_state.get("new.data_id") or "")
                    if did:
                        counts = st.session_state.get("data.use_counts")
                        if not isinstance(counts, dict):
                            counts = {}
                        counts[did] = int(counts.get(did, 0)) + 1
                        st.session_state["data.use_counts"] = counts
                except Exception:
                    pass

                st.session_state["new.step"] = 1
                st.rerun()

    # -------------------------------------------------------------------------
    # Step 1: Plan
    # -------------------------------------------------------------------------
    elif step == 1:
        st.write("Define your plan. This becomes the baseline that variations are generated around.")

        st.caption("Module: DCA/Swing (spot, long-only).")

        _render_plan_blueprint_import_ui()

        baseline_params = build_dca_baseline_params()
        st.session_state["new.baseline_params"] = baseline_params
        _render_current_plan_blueprint(baseline_params)
        st.session_state["new.template_path"] = "strategies.dca_swing:Strategy"
        st.session_state["new.grid_script"] = str(REPO_ROOT / "tools" / "make_dca_grid.py")
        st.session_state["new.market_mode"] = "spot"
        st.session_state["new.strategy_name"] = "dca_swing"

        colL, colR = st.columns(2)
        with colL:
            if st.button("← Back"):
                st.session_state["new.step"] = 0
                st.rerun()
        with colR:
            if st.button("Next →", type="primary"):
                st.session_state["new.step"] = 2
                st.rerun()

    # -------------------------------------------------------------------------
    # Step 2: Variations (grid)
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    # Step 2: Variations (grid)
    # -------------------------------------------------------------------------
    
    elif step == 2:
        st.write("Decide how this plan should be stress-tested.")

        # Baseline-aware entry mode detection (for accurate drift previews).
        baseline_params = st.session_state.get("new.baseline_params", {}) or {}
        try:
            base_el = baseline_params.get("entry_logic")
            base_is_logic = bool(
                isinstance(base_el, dict)
                and isinstance(base_el.get("clauses"), list)
                and len(base_el.get("clauses") or []) > 0
                and str(baseline_params.get("buy_filter", "none")).lower() in {"none", ""}
            )
        except Exception:
            base_is_logic = False

        LABELS = {
            "deposits": "Funding (deposits)",
            "buys": "Buy cadence",
            "filter": "Entry gate (simple filter)",
            "logic": "Entry gate (logic builder)",
            "alloc": "Allocation cap",
            "risk": "Risk & exits (SL/TP + time/trail)",
        }

        # ------------------------------------------------------------
        # Stress mode
        # ------------------------------------------------------------
        stress_mode = st.radio(
            "Stress mode",
            ["Around this plan (recommended)", "Explore the space (advanced)"],
            index=0,
            horizontal=False,
        )
        is_random = stress_mode.startswith("Explore")

        # ------------------------------------------------------------
        # Controls
        # ------------------------------------------------------------
        left, right = st.columns([2, 1], gap="large")

        with left:
            st.markdown("#### How many variants should we generate?")
            breadth_ui = st.select_slider(
                "Stress breadth",
                options=["Small", "Medium", "Large"],
                value="Medium",
                help="Controls coverage vs compute cost (not performance).",
            )
            breadth_map = {
                "Small": (300, "Light"),
                "Medium": (1000, "Medium"),
                "Large": (2000, "Heavy"),
            }
            n_default, compute_label = breadth_map[breadth_ui]

            # Let power users override the default count.
            n_variants = int(
                st.number_input(
                    "Estimated variants",
                    min_value=50,
                    max_value=10000,
                    value=int(st.session_state.get("new.grid_n", n_default)),
                    step=50,
                )
            )

            st.markdown("---")
            st.markdown("#### How far can things drift?")
            intensity_ui = st.radio(
                "Drift intensity",
                ["Conservative", "Balanced", "Aggressive"],
                index=1,
                horizontal=True,
                help="Controls the size of parameter changes (independent of how many variants you generate).",
            )
            intensity_key = {"Conservative": "conservative", "Balanced": "balanced", "Aggressive": "aggressive"}[intensity_ui]

            st.markdown("---")
            st.markdown("#### What can drift from the baseline?")

            # Entry gate style (random mode can mix filter + logic)
            if is_random:
                entry_style = st.radio(
                    "Entry gate style (if enabled)",
                    ["Simple filter", "Logic builder", "Mix"],
                    index=2,
                    horizontal=True,
                )
            else:
                entry_style = "Logic builder" if base_is_logic else "Simple filter"

            # Default checkboxes
            default_vary = set(st.session_state.get("new.grid_vary", ["deposits", "buys", "filter", "alloc", "risk"]))
            cols = st.columns(2)
            vary_groups: List[str] = []

            # Funding + buys
            with cols[0]:
                vary_deposits = st.checkbox(LABELS["deposits"], value=("deposits" in default_vary))
                vary_alloc = st.checkbox(LABELS["alloc"], value=("alloc" in default_vary))
            with cols[1]:
                vary_buys = st.checkbox(LABELS["buys"], value=("buys" in default_vary))
                vary_risk = st.checkbox(LABELS["risk"], value=("risk" in default_vary))

            if vary_deposits:
                vary_groups.append("deposits")
            if vary_buys:
                vary_groups.append("buys")
            if vary_alloc:
                vary_groups.append("alloc")
            if vary_risk:
                vary_groups.append("risk")

            # Entry gate toggle(s)
            entry_enabled = st.checkbox("Entry gate", value=(("filter" in default_vary) or ("logic" in default_vary)))
            logic_frac = float(st.session_state.get("new.logic_frac", 0.35))

            if entry_enabled:
                if entry_style == "Simple filter":
                    vary_groups.append("filter")
                elif entry_style == "Logic builder":
                    vary_groups.append("logic")
                else:
                    vary_groups.extend(["filter", "logic"])
                    logic_frac = float(
                        st.slider(
                            "Logic share (for entry gate variants)",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(logic_frac),
                            step=0.05,
                            help="When exploring the space, entry gates can be a mix of simple filters and the logic builder. This controls the mix.",
                        )
                    )

            # Pull width from the shared drift spec (Option A).
            try:
                from make_dca_grid import DCA_DRIFT_SPEC_V1, preview_drift_table
                width = str(DCA_DRIFT_SPEC_V1["intensities"][intensity_key]["width"])
            except Exception:
                width = "medium"
                preview_drift_table = None  # type: ignore

            # Mode string for generator
            gen_mode = "random" if is_random else "neighborhood"

            st.markdown("---")
            st.markdown("#### Range preview")
            if preview_drift_table is not None:
                rows = preview_drift_table(
                    mode=gen_mode,
                    intensity=intensity_key,
                    base_cfg={"params": baseline_params} if isinstance(baseline_params, dict) else None,
                    vary_groups=set(vary_groups),
                    logic_frac=logic_frac,
                    width=width,
                    overrides=st.session_state.get("new.drift_overrides") or None,
                )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Preview unavailable (could not import make_dca_grid).")

            st.markdown("---")
            # Seed: Auto by default (keeps UI clean; numeric seed only when needed)
            if "new.grid_seed_auto" not in st.session_state:
                st.session_state["new.grid_seed_auto"] = int(np.random.randint(1, 2_000_000_000))
            adv = st.expander("Advanced (reproducibility & inspection)")
            with adv:
                seed_mode = st.radio(
                    "Seed",
                    options=["Auto (recommended)", "Custom"],
                    index=0 if str(st.session_state.get("new.seed_mode", "Auto (recommended)")).startswith("Auto") else 1,
                    key="new.seed_mode",
                )

                if str(seed_mode).startswith("Custom"):
                    st.number_input(
                        "Seed value",
                        min_value=1,
                        max_value=2_000_000_000,
                        value=int(st.session_state.get("new.grid_seed_custom", 1)),
                        step=1,
                        key="new.grid_seed_custom",
                    )
                    st.caption("Same seed → same grid (use this when you want reproducible comparisons).")
                else:
                    st.caption("Auto seed will be chosen for this run. The seed used will be recorded in the run metadata.")

                seed = int(st.session_state["new.grid_seed_auto"]) if str(seed_mode).startswith("Auto") else int(st.session_state.get("new.grid_seed_custom", 1))

                st.markdown("##### Drift overrides (optional)")
                ov_enabled = st.checkbox(
                    "Override drift ranges",
                    value=bool(st.session_state.get("new.ov_enabled", False)),
                    key="new.ov_enabled",
                )
                overrides: Dict[str, Any] = {}

                if ov_enabled:
                    st.caption("Use this to narrow/widen what each drift group is allowed to do. Applies to preview and the actual grid.")
                    intensity_pct = {"conservative": 0.25, "balanced": 0.50, "aggressive": 1.00}.get(str(intensity_key), 0.50)

                    # Funding (deposits)
                    if "deposits" in vary_groups:
                        st.markdown("**Funding (deposits)**")
                        base_dep = float(baseline_params.get("deposit_amount_usd", 0.0) or 0.0)
                        dep_hi_default = (base_dep * (1.0 + intensity_pct)) if base_dep > 0 else 100.0
                        dep_lo_default = max(0.0, base_dep * (1.0 - intensity_pct))
                        dep_max = max(500.0, dep_hi_default * 2.0)
                        dep_range = st.slider(
                            "Deposit amount range ($)",
                            min_value=0.0,
                            max_value=float(dep_max),
                            value=(float(dep_lo_default), float(dep_hi_default)),
                            step=5.0,
                            key="new.ov.dep_amt",
                        )
                        overrides["deposit_amount_usd"] = {"min": float(dep_range[0]), "max": float(dep_range[1])}

                        dep_freq_universe = ["none", "weekly", "monthly"]
                        base_dep_freq = str(baseline_params.get("deposit_freq", "weekly") or "weekly").lower()
                        dep_freq_default = [base_dep_freq] if base_dep_freq in dep_freq_universe else ["weekly"]
                        dep_freq_opts = st.multiselect(
                            "Deposit cadence options",
                            options=dep_freq_universe,
                            default=dep_freq_default,
                            key="new.ov.dep_freq",
                        )
                        if dep_freq_opts:
                            overrides["deposit_freq"] = {"options": dep_freq_opts}

                    # Buys
                    if "buys" in vary_groups:
                        st.markdown("**Buy cadence**")
                        base_buy = float(baseline_params.get("buy_amount_usd", 0.0) or 0.0)
                        buy_hi_default = (base_buy * (1.0 + intensity_pct)) if base_buy > 0 else 100.0
                        buy_lo_default = max(0.0, base_buy * (1.0 - intensity_pct))
                        buy_max = max(500.0, buy_hi_default * 2.0)
                        buy_range = st.slider(
                            "Buy amount range ($)",
                            min_value=0.0,
                            max_value=float(buy_max),
                            value=(float(buy_lo_default), float(buy_hi_default)),
                            step=5.0,
                            key="new.ov.buy_amt",
                        )
                        overrides["buy_amount_usd"] = {"min": float(buy_range[0]), "max": float(buy_range[1])}

                        buy_freq_universe = ["weekly", "monthly"]
                        base_buy_freq = str(baseline_params.get("buy_freq", "weekly") or "weekly").lower()
                        buy_freq_default = [base_buy_freq] if base_buy_freq in buy_freq_universe else ["weekly"]
                        buy_freq_opts = st.multiselect(
                            "Buy cadence options",
                            options=buy_freq_universe,
                            default=buy_freq_default,
                            key="new.ov.buy_freq",
                        )
                        if buy_freq_opts:
                            overrides["buy_freq"] = {"options": buy_freq_opts}

                    # Entry gate (legacy filter mode)
                    if "filter" in vary_groups:
                        st.markdown("**Entry gate (filter mode)**")
                        try:
                            from make_dca_grid import DCA_DRIFT_SPEC_V1 as _SPEC
                            filt_universe = list((_SPEC.get("universes") or {}).get("buy_filters") or [])
                        except Exception:
                            filt_universe = ["none", "below_ema", "rsi_below", "macd_bull", "bb_z_below", "adx_above", "donch_pos_below"]

                        base_filt = str(baseline_params.get("buy_filter", "none") or "none").lower()
                        filt_default = [base_filt] if base_filt in filt_universe else ["none"]
                        filt_opts = st.multiselect(
                            "Allowed filters",
                            options=filt_universe,
                            default=filt_default,
                            key="new.ov.buy_filter",
                        )
                        if filt_opts:
                            overrides["buy_filter"] = {"options": filt_opts}

                    # Allocation
                    if "alloc" in vary_groups:
                        st.markdown("**Allocation cap**")
                        base_alloc = float(baseline_params.get("max_alloc_pct", 1.0) or 1.0) * 100.0
                        alloc_lo = max(0.0, base_alloc - 50.0 * intensity_pct)
                        alloc_hi = 100.0
                        alloc_rng = st.slider(
                            "Allocation cap range (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=(float(alloc_lo), float(alloc_hi)),
                            step=1.0,
                            key="new.ov.alloc",
                        )
                        overrides["max_alloc_pct"] = {"min": float(alloc_rng[0]) / 100.0, "max": float(alloc_rng[1]) / 100.0}

                    # Risk & exits
                    if "risk" in vary_groups:
                        st.markdown("**Risk & exits**")
                        base_sl = float(baseline_params.get("sl_pct", 0.0) or 0.0) * 100.0
                        base_tp = float(baseline_params.get("tp_pct", 0.0) or 0.0) * 100.0
                        base_tr = float(baseline_params.get("trail_pct", 0.0) or 0.0) * 100.0
                        base_hold = int(baseline_params.get("max_hold_bars", 0) or 0)

                        sl_hi = float(max(base_sl, 30.0 * (1.0 + intensity_pct)))
                        sl_rng = st.slider(
                            "Stop loss range (%)",
                            min_value=0.0,
                            max_value=60.0,
                            value=(0.0, float(min(60.0, sl_hi))),
                            step=1.0,
                            key="new.ov.sl",
                        )
                        overrides["sl_pct"] = {"min": float(sl_rng[0]) / 100.0, "max": float(sl_rng[1]) / 100.0}

                        tp_hi = float(max(base_tp, 60.0 * (1.0 + intensity_pct)))
                        tp_rng = st.slider(
                            "Take profit range (%)",
                            min_value=0.0,
                            max_value=200.0,
                            value=(0.0, float(min(200.0, tp_hi))),
                            step=1.0,
                            key="new.ov.tp",
                        )
                        overrides["tp_pct"] = {"min": float(tp_rng[0]) / 100.0, "max": float(tp_rng[1]) / 100.0}

                        tr_hi = float(max(base_tr, 20.0 * (1.0 + intensity_pct)))
                        tr_rng = st.slider(
                            "Trailing stop range (%)",
                            min_value=0.0,
                            max_value=60.0,
                            value=(0.0, float(min(60.0, tr_hi))),
                            step=1.0,
                            key="new.ov.trail",
                        )
                        overrides["trail_pct"] = {"min": float(tr_rng[0]) / 100.0, "max": float(tr_rng[1]) / 100.0}

                        hold_default = int(min(3650, max(base_hold, 365)))
                        hold_rng = st.slider(
                            "Max hold (days)",
                            min_value=0,
                            max_value=3650,
                            value=(0, int(hold_default)),
                            step=1,
                            key="new.ov.hold",
                        )
                        overrides["max_hold_bars"] = {"min": int(hold_rng[0]), "max": int(hold_rng[1])}

                        sell_rng = st.slider(
                            "TP sell fraction range",
                            min_value=0.05,
                            max_value=1.0,
                            value=(0.25, 1.0),
                            step=0.05,
                            key="new.ov.sell",
                        )
                        overrides["tp_sell_fraction"] = {"min": float(sell_rng[0]), "max": float(sell_rng[1])}

                        res_rng = st.slider(
                            "Reserve fraction range (sell proceeds)",
                            min_value=0.0,
                            max_value=1.0,
                            value=(0.0, 1.0),
                            step=0.05,
                            key="new.ov.res",
                        )
                        overrides["reserve_frac_of_proceeds"] = {"min": float(res_rng[0]), "max": float(res_rng[1])}

                st.session_state["new.drift_overrides"] = overrides if ov_enabled else {}


        with right:
            st.markdown("#### Stress impact")
            st.caption(f"Mode: {'Explore the space' if is_random else 'Around baseline'} · Intensity: {intensity_ui}")
            st.metric("Estimated variants", f"{n_variants:,}")
            st.metric("Compute load", compute_label)
            st.progress({"Small": 0.25, "Medium": 0.55, "Large": 0.85}[breadth_ui])

            st.caption("More variants increase coverage, not accuracy.")
            if vary_groups:
                pretty = " · ".join(LABELS.get(g, g) for g in vary_groups)
            else:
                pretty = "Nothing (fully pinned)."
            st.caption(f"Drifting: {pretty}")

        # Persist for step 3
        st.session_state["new.grid_mode2"] = gen_mode
        st.session_state["new.grid_n"] = int(n_variants)
        st.session_state["new.grid_seed"] = int(seed)
        st.session_state["new.grid_intensity"] = intensity_key
        st.session_state["new.grid_width"] = width
        st.session_state["new.grid_vary"] = list(dict.fromkeys(vary_groups))
        st.session_state["new.logic_frac"] = float(logic_frac)

        st.markdown("")
        nav_cols = st.columns([1, 1, 1])
        with nav_cols[0]:
            if st.button("← Back"):
                st.session_state["new.step"] = max(0, int(st.session_state["new.step"]) - 1)
                st.rerun()
        with nav_cols[2]:
            if st.button("Next →", type="primary"):
                st.session_state["new.step"] = int(st.session_state["new.step"]) + 1
                st.rerun()
    elif step == 3:
        st.write("Confirm run settings, then execute Batch. You can optionally run deeper stability checks afterward.")

        data_path = Path(st.session_state.get("new.data_path", ""))
        if not data_path.exists():
            st.error("Missing dataset. Go back to Step 1.")
            st.stop()

        # Run name
        default_name = f"batch_{_now_slug()}_{st.session_state.get('new.strategy_name','strategy')}_{_slug(data_path.stem)}"

        # Compute mode (future-proof for server execution later)
        compute_mode = "Local (auto)"

        # Auto workers (dev/ops knob, not a user knob)
        cpu = os.cpu_count() or 4
        jobs_auto = int(max(1, min(8, round(cpu * 0.5))))
        if "new.jobs" not in st.session_state:
            st.session_state["new.jobs"] = jobs_auto

        # Compute budget presets (maps to rerun + artifact retention)
        preset_defs = {
            "Quick (recommended)": {"rerun_n": 200, "save_details": "Normal"},
            "Balanced": {"rerun_n": 400, "save_details": "Normal"},
            "Deep": {"rerun_n": 900, "save_details": "Full"},
        }
        save_details_to_topk = {"Minimal": 0, "Normal": 50, "Full": 200}

        # Default preset (first time only)
        if "new.compute_preset" not in st.session_state:
            st.session_state["new.compute_preset"] = "Quick (recommended)"
        if "new.save_details" not in st.session_state:
            st.session_state["new.save_details"] = preset_defs[st.session_state["new.compute_preset"]]["save_details"]
        if "new.rerun_n" not in st.session_state:
            st.session_state["new.rerun_n"] = int(preset_defs[st.session_state["new.compute_preset"]]["rerun_n"])
        if "new.top_k" not in st.session_state:
            st.session_state["new.top_k"] = int(save_details_to_topk.get(st.session_state["new.save_details"], 50))

        # Run overview
        with st.container():
            st.markdown("#### Run overview")
            run_name = st.text_input("Run name (folder)", value=default_name, key="new.run_name")

            grid_n = int(st.session_state.get("new.grid_n", 0) or 0)
            strat = str(st.session_state.get("new.strategy_name", "strategy"))
            st.caption(f"Testing **{grid_n:,} variations** of **{strat}** on **{data_path.stem}**. Compute mode: **{compute_mode}**.")

        # Layout: left (controls) + right (receipt)
        left, right = st.columns([0.62, 0.38], gap="large")

        with left:
            st.markdown("### How big should this run be?")

            preset = st.radio(
                "Compute budget",
                options=list(preset_defs.keys()),
                index=list(preset_defs.keys()).index(str(st.session_state.get("new.compute_preset", "Quick (recommended)"))),
                key="new.compute_preset",
                horizontal=True,
            )

            # Apply preset on change (but do not fight manual overrides later)
            prev = st.session_state.get("new.compute_preset_prev")
            if preset != prev:
                st.session_state["new.rerun_n"] = int(preset_defs[preset]["rerun_n"])
                st.session_state["new.save_details"] = preset_defs[preset]["save_details"]
                st.session_state["new.top_k"] = int(save_details_to_topk.get(st.session_state["new.save_details"], 50))
                st.session_state["new.compute_preset_prev"] = preset

            st.caption("Compute budget controls confidence vs cost (not performance).")

            with st.expander("Advanced (power users)", expanded=False):
                st.caption("Only needed if you want to tune accuracy vs saved detail.")
                st.number_input(
                    "Re-test top survivors (more accurate ranking, slower)",
                    min_value=1,
                    max_value=10_000,
                    value=int(st.session_state.get("new.rerun_n", 200)),
                    step=25,
                    key="new.rerun_n",
                )
                save_details = st.selectbox(
                    "Save run details",
                    options=["Minimal", "Normal", "Full"],
                    index=["Minimal", "Normal", "Full"].index(str(st.session_state.get("new.save_details", "Normal"))),
                    key="new.save_details",
                )
                st.session_state["new.top_k"] = int(save_details_to_topk.get(save_details, 50))

            st.markdown("### Safety filters")
            st.caption("Stops misleading 'lucky' results from dominating. Keep these permissive for DCA.")

            use_rec = st.toggle("Use recommended filters", value=bool(st.session_state.get("new.use_rec_filters", True)), key="new.use_rec_filters")
            rec_prev = st.session_state.get("new.use_rec_filters_prev")
            if use_rec and rec_prev is False:
                st.session_state["new.min_trades"] = 0
                st.session_state["new.max_fee"] = 250.0
                st.session_state["new.max_best"] = 0.95
            st.session_state["new.use_rec_filters_prev"] = bool(use_rec)

            colG1, colG2 = st.columns(2, gap="large")
            with colG1:
                st.number_input(
                    "Starting capital ($)",
                    min_value=10.0,
                    max_value=1_000_000.0,
                    value=float(st.session_state.get("new.starting_eq", 1000.0)),
                    step=100.0,
                    key="new.starting_eq",
                )
                st.number_input(
                    "Minimum trades (optional)",
                    min_value=0,
                    max_value=10_000,
                    value=int(st.session_state.get("new.min_trades", 0)),
                    step=1,
                    key="new.min_trades",
                )
                st.caption("Use this to filter out 'one trade wonders'. 0 disables.")
            with colG2:
                st.number_input(
                    "Max fee drag (%)",
                    min_value=0.0,
                    max_value=10_000.0,
                    value=float(st.session_state.get("new.max_fee", 250.0)),
                    step=10.0,
                    key="new.max_fee",
                )
                st.caption("Reject plans where fees likely explain the gains.")
                st.number_input(
                    "Max one-trade dominance",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(st.session_state.get("new.max_best", 0.95)),
                    step=0.01,
                    key="new.max_best",
                )
                st.caption("Reject plans dominated by one outlier win (lower = stricter).")

            with st.container():
                st.markdown("### Increase confidence (recommended)")
                st.caption("Batch finds candidates. These checks test whether results survive timing luck and different market periods.")

                confidence_opts = [
                    "Batch only — fast scan",
                    "Batch + Rolling Starts — recommended",
                    "Batch + Rolling Starts + Walkforward — best",
                ]
                if "new.confidence_level" not in st.session_state:
                    st.session_state["new.confidence_level"] = confidence_opts[1]
                confidence_level = st.radio(
                    "Confidence level",
                    options=confidence_opts,
                    index=confidence_opts.index(str(st.session_state.get("new.confidence_level", confidence_opts[1]))),
                    key="new.confidence_level",
                )

                # Map confidence choice → stability checks (stored for downstream pipeline)
                if str(confidence_level).startswith("Batch only"):
                    st.session_state["new.do_rs"] = False
                    st.session_state["new.do_wf"] = False
                elif "Walkforward" in str(confidence_level):
                    st.session_state["new.do_rs"] = True
                    st.session_state["new.do_wf"] = True
                else:
                    st.session_state["new.do_rs"] = True
                    st.session_state["new.do_wf"] = False

                new_do_rs = bool(st.session_state.get("new.do_rs", False))
                new_do_wf = bool(st.session_state.get("new.do_wf", False))


            st.markdown("### Sort survivors by")

            score_labels = {
                "Balanced growth vs drawdown (recommended)": "calmar_equity",
                "Growth-first (ignore drawdown)": "profit",
                "Drawdown-first (defensive)": "profit_dd",
                "TWR vs drawdown": "twr_dd",
            }
            score_ids = list(score_labels.values())
            id_to_label = {v: k for k, v in score_labels.items()}
            st.selectbox(
                "Sort survivors by",
                options=score_ids,
                index=score_ids.index(str(st.session_state.get("new.score", "calmar_equity"))) if str(st.session_state.get("new.score", "calmar_equity")) in score_ids else 0,
                key="new.score",
                format_func=lambda v: id_to_label.get(v, str(v)),
            )
            st.slider("Hard drawdown limit (optional) — 0 disables", 0.0, 0.99, float(st.session_state.get("new.max_dd_filter", 0.0)), 0.01, key="new.max_dd_filter")

        with right:
            with st.container():
                st.markdown("### Run receipt")
                st.caption(f"Compute mode: **{compute_mode}**")
                st.metric("Estimated variants", f"{int(st.session_state.get('new.grid_n', 0)):,}")
                st.metric("Re-test survivors", f"{int(st.session_state.get('new.rerun_n', 200)):,}")
                st.metric("Saved detail", str(st.session_state.get("new.save_details", "Normal")))
                st.metric("Confidence level", "Highest" if bool(st.session_state.get("new.do_wf", False)) else ("Strong" if bool(st.session_state.get("new.do_rs", False)) else "Exploration"))
                st.caption("Higher confidence = more compute.")

                # Simple load signal (local-only framing)
                n = int(st.session_state.get("new.grid_n", 0))
                r = int(st.session_state.get("new.rerun_n", 200))
                extra = 0
                # Stability checks add extra compute after Batch
                if bool(st.session_state.get("new.do_rs", False)):
                    extra += r
                if bool(st.session_state.get("new.do_wf", False)):
                    extra += r
                rough = n + r + extra
                if rough <= 800:
                    load = "Low"
                elif rough <= 2200:
                    load = "Medium"
                else:
                    load = "High"
                st.metric("Compute load", load)
                st.caption("More compute increases coverage/confidence, not accuracy.")

        # Ensure variables exist for the runner below
        jobs = int(st.session_state.get("new.jobs", jobs_auto))
        rerun_n = int(st.session_state.get("new.rerun_n", 200))
        top_k = int(st.session_state.get("new.top_k", 50))
        min_trades = int(st.session_state.get("new.min_trades", 0))
        max_fee_impact = float(st.session_state.get("new.max_fee", 250.0))
        max_best_over_wins = float(st.session_state.get("new.max_best", 0.95))
        starting_equity = float(st.session_state.get("new.starting_eq", 1000.0))
        score = str(st.session_state.get("new.score", "calmar_equity"))
        max_dd_filter = float(st.session_state.get("new.max_dd_filter", 0.0))
        # Rough bars/day hint for defaults
        bars_per_day_hint = 1
        try:
            bar_ms_hint = _infer_bar_ms_from_csv(data_path)
            if bar_ms_hint:
                bars_per_day_hint = int(max(1, round(86_400_000 / float(bar_ms_hint))))
        except Exception:
            bars_per_day_hint = 1

        if new_do_rs:
            with st.expander("Rolling Starts settings", expanded=True):
                st.caption("Tests the same plan from many different start dates to see if results depend on lucky timing.")

                rs_presets = {
                    "Light (fast)": (14, 180),
                    "Standard (recommended)": (10, 270),
                    "Heavy (high confidence)": (7, 365),
                }
                preset_label = st.selectbox(
                    "Preset",
                    list(rs_presets.keys()),
                    index=list(rs_presets.keys()).index(str(st.session_state.get("new.rs.preset_label", "Standard (recommended)"))) if str(st.session_state.get("new.rs.preset_label", "Standard (recommended)")) in rs_presets else 1,
                    key="new.rs.preset_label",
                )
                preset_prev = st.session_state.get("new.rs.preset_prev_label", None)
                if preset_label != preset_prev:
                    step_days, min_days = rs_presets[preset_label]
                    st.session_state["new.rs.start_step"] = int(max(1, round(step_days * bars_per_day_hint)))
                    st.session_state["new.rs.min_bars"] = int(max(30, round(min_days * bars_per_day_hint)))
                    st.session_state["new.rs.step_days_ui"] = int(step_days)
                    st.session_state["new.rs.min_days_ui"] = int(min_days)
                    st.session_state["new.rs.preset_prev_label"] = preset_label

                start_step_bars = int(st.session_state.get("new.rs.start_step", max(1, int(round(10 * bars_per_day_hint)))))
                min_bars = int(st.session_state.get("new.rs.min_bars", max(30, int(round(270 * bars_per_day_hint)))))
                approx_step_days = max(1, int(round(start_step_bars / float(bars_per_day_hint))))
                approx_min_days = max(1, int(round(min_bars / float(bars_per_day_hint))))
                st.caption(f"Current: start spacing ≈ **{approx_step_days} days** · minimum test length ≈ **{approx_min_days} days**.")

                with st.expander("Advanced (power users)", expanded=False):
                    st.caption(f"Converted using ~{bars_per_day_hint} bars/day for this dataset.")
                    step_days_in = st.number_input(
                        "Start spacing (days)",
                        min_value=1,
                        max_value=3650,
                        value=int(st.session_state.get("new.rs.step_days_ui", approx_step_days)),
                        step=1,
                        key="new.rs.step_days_ui",
                    )
                    min_days_in = st.number_input(
                        "Minimum test length (days)",
                        min_value=30,
                        max_value=36500,
                        value=int(st.session_state.get("new.rs.min_days_ui", approx_min_days)),
                        step=10,
                        key="new.rs.min_days_ui",
                    )

                    # Convert day controls → bars used by the engine
                    st.session_state["new.rs.start_step"] = int(max(1, round(float(step_days_in) * bars_per_day_hint)))
                    st.session_state["new.rs.min_bars"] = int(max(30, round(float(min_days_in) * bars_per_day_hint)))

                    with st.expander("Raw (bars)", expanded=False):
                        st.number_input(
                            "Start spacing (bars)",
                            min_value=1,
                            max_value=500_000,
                            value=int(st.session_state.get("new.rs.start_step", start_step_bars)),
                            step=5,
                            key="new.rs.start_step",
                        )
                        st.number_input(
                            "Min bars per start",
                            min_value=30,
                            max_value=5_000_000,
                            value=int(st.session_state.get("new.rs.min_bars", min_bars)),
                            step=10,
                            key="new.rs.min_bars",
                        )
                        st.caption("If you override raw bars, the day-based labels above may not reflect the exact conversion.")

        if new_do_wf:
            with st.expander("Walkforward settings", expanded=True):
                st.caption("Splits history into time windows to check whether the plan survives across different market periods.")

                wf_presets = {
                    "Light (fast)": (30, 15),
                    "Standard (recommended)": (90, 30),
                    "Heavy (high confidence)": (180, 30),
                }
                preset_label = st.selectbox(
                    "Preset",
                    list(wf_presets.keys()),
                    index=list(wf_presets.keys()).index(str(st.session_state.get("new.wf.preset_label", "Standard (recommended)"))) if str(st.session_state.get("new.wf.preset_label", "Standard (recommended)")) in wf_presets else 1,
                    key="new.wf.preset_label",
                )
                preset_prev = st.session_state.get("new.wf.preset_prev_label", None)
                if preset_label != preset_prev:
                    window_days, step_days = wf_presets[preset_label]
                    st.session_state["new.wf.window_days"] = int(window_days)
                    st.session_state["new.wf.step_days"] = int(step_days)
                    expected_window_bars = int(max(1, round(window_days * bars_per_day_hint)))
                    st.session_state["new.wf.min_bars"] = int(max(1, round(0.8 * expected_window_bars)))
                    st.session_state["new.wf.min_days_ui"] = int(max(1, round(st.session_state["new.wf.min_bars"] / float(bars_per_day_hint))))
                    st.session_state["new.wf.preset_prev_label"] = preset_label

                # Workers are an execution detail (local auto / server-managed later)
                if "new.wf.jobs" not in st.session_state:
                    st.session_state["new.wf.jobs"] = int(st.session_state.get("new.jobs", 8))

                window_days = int(st.session_state.get("new.wf.window_days", 90))
                step_days = int(st.session_state.get("new.wf.step_days", 30))
                min_bars = int(st.session_state.get("new.wf.min_bars", max(1, int(round(0.8 * window_days * bars_per_day_hint)))))
                approx_min_days = max(1, int(round(min_bars / float(bars_per_day_hint))))
                st.caption(f"Current: window **{window_days} days** · step **{step_days} days** · minimum data ≈ **{approx_min_days} days** per window.")

                with st.expander("Advanced (power users)", expanded=False):
                    st.caption(f"Converted using ~{bars_per_day_hint} bars/day for this dataset.")
                    st.number_input(
                        "Window length (days)",
                        min_value=7,
                        max_value=3650,
                        value=int(window_days),
                        step=1,
                        key="new.wf.window_days",
                    )
                    st.number_input(
                        "Step forward (days)",
                        min_value=1,
                        max_value=3650,
                        value=int(step_days),
                        step=1,
                        key="new.wf.step_days",
                    )

                    min_days_in = st.number_input(
                        "Minimum data per window (days)",
                        min_value=1,
                        max_value=3650,
                        value=int(st.session_state.get("new.wf.min_days_ui", approx_min_days)),
                        step=1,
                        key="new.wf.min_days_ui",
                    )

                    st.session_state["new.wf.min_bars"] = int(max(1, round(float(min_days_in) * bars_per_day_hint)))

                    with st.expander("Raw (bars)", expanded=False):
                        st.number_input(
                            "Min bars per window",
                            min_value=1,
                            max_value=5_000_000,
                            value=int(st.session_state.get("new.wf.min_bars", min_bars)),
                            step=1,
                            key="new.wf.min_bars",
                        )
                        st.caption("Workers are auto-managed in local mode and server-managed in managed mode.")
        # Navigation + run
        do_run = False
        nav_cols = st.columns([1, 1, 1])
        with nav_cols[0]:
            if st.button("← Back"):
                st.session_state["new.step"] = max(0, int(st.session_state.get("new.step", 3)) - 1)
                st.rerun()
        with nav_cols[2]:
            if str(st.session_state.get("new.compute_preset", "")).startswith("Quick") and (not bool(st.session_state.get("new.do_rs", False))) and (not bool(st.session_state.get("new.do_wf", False))):
                st.info("Tip: Adding Rolling Starts often catches false positives caused by lucky timing.", icon="💡")
            do_run = st.button("Run stress test", type="primary")

        if do_run:
            try:
                t0 = time.time()
                tmp_run_dir = tmp_dir / f"run_{_now_slug()}"
                tmp_run_dir.mkdir(parents=True, exist_ok=True)

                # UI: unified run monitor (Sprint 3)
                st.subheader("Run monitor")
                grid_comp_ph = st.empty()
                stages: List[_PipelineStage] = [
                    _PipelineStage("grid", "Variants"),
                    _PipelineStage("batch", "Batch"),
                    _PipelineStage("post", "Postprocess"),
                ]
                if bool(st.session_state.get("new.do_rs", False)):
                    stages.append(_PipelineStage("rs", "Rolling Starts"))
                if bool(st.session_state.get("new.do_wf", False)):
                    stages.append(_PipelineStage("wf", "Walkforward"))
                pipe = _PipelineUI(stages)

                # 1) Write baseline
                base_path = _write_baseline_json(
                    tmp_run_dir,
                    strategy_name=st.session_state.get("new.strategy_name", "dca_swing"),
                    side="long",
                    params=st.session_state.get("new.baseline_params", {}),
                )

                # 2) Generate grid
                grid_path = tmp_run_dir / f"grid_{st.session_state['new.grid_n']}_seed{st.session_state['new.grid_seed']}.jsonl"
                grid_cmd: List[str] = [
                    PY,
                    st.session_state["new.grid_script"],
                    "--out",
                    str(grid_path),
                    "--n",
                    str(max(0, int(st.session_state["new.grid_n"]) - 1)),
                    "--seed",
                    str(int(st.session_state["new.grid_seed"])),
                ]
                if str(st.session_state.get("new.grid_mode2", "neighborhood")) == "random":
                    grid_cmd += [
                        "--mode",
                        "random",
                        "--base",
                        str(base_path),
                        "--intensity",
                        str(st.session_state.get("new.grid_intensity", "balanced")),
                        "--vary",
                        ",".join(st.session_state.get("new.grid_vary", ["deposits", "buys", "filter", "logic", "alloc", "risk"])),
                        "--logic-frac",
                        str(float(st.session_state.get("new.logic_frac", 0.35))),
                    ]
                else:
                    grid_cmd += [
                        "--mode",
                        "neighborhood",
                        "--base",
                        str(base_path),
                        "--width",
                        str(st.session_state.get("new.grid_width", "medium")),
                        "--intensity",
                        str(st.session_state.get("new.grid_intensity", "balanced")),
                        "--vary",
                        ",".join(st.session_state.get("new.grid_vary", ["deposits", "buys", "filter", "logic", "alloc", "risk"])),
                    ]



                # Attach drift overrides (if any)
                ov = st.session_state.get("new.drift_overrides") or {}
                if isinstance(ov, dict) and ov:
                    ov_path = tmp_run_dir / "drift_overrides.json"
                    _write_json(ov_path, ov)
                    grid_cmd += ["--overrides", str(ov_path)]

                pipe.run("grid", grid_cmd, cwd=REPO_ROOT)
                _ensure_grid_has_baseline(grid_path, base_path, total_n=int(st.session_state["new.grid_n"]))

                # Grid composition (dopamine loop): show + save
                try:
                    run_dir = RUNS_DIR / str(run_name)
                    comp = _summarize_grid_composition(Path(str(grid_path)))
                    _write_json(run_dir / "grid_meta.json", comp)
                    with grid_comp_ph.container():
                        _render_grid_composition(comp)
                except Exception as _e:
                    # Never fail the run on a summary widget
                    with grid_comp_ph.container():
                        st.caption(f"Grid composition unavailable: {_e}")

                # 3) Batch
                template_path = str(st.session_state.get("new.template_path", "strategies.dca_swing:Strategy"))
                market_mode = str(st.session_state.get("new.market_mode", "spot"))
                batch_cmd: List[str] = [
                    PY,
                    "-m",
                    "engine.batch",
                    "--data",
                    str(data_path),
                    "--grid",
                    str(grid_path),
                    "--template",
                    template_path,
                    "--market-mode",
                    market_mode,
                    "--run-name",
                    str(run_name),
                    "--out",
                    str(RUNS_DIR),
                    "--jobs",
                    str(jobs),
                    "--fast-sweep",
                    "--min-trades",
                    str(min_trades),
                    "--max-fee-impact-pct",
                    str(max_fee_impact),
                    "--max-best-over-wins",
                    str(max_best_over_wins),
                    "--sweep-sort-by",
                    "equity.net_profit_ex_cashflows",
                    "--sweep-sort-desc",
                    "--sort-by",
                    "equity.net_profit_ex_cashflows",
                    "--sort-desc",
                    "--rerun-n",
                    str(rerun_n),
                    "--top-k",
                    str(top_k),
                    "--starting-equity",
                    str(starting_equity),
                ]
                batch_progress = RUNS_DIR / str(run_name) / "progress" / "batch.jsonl"
                batch_progress.parent.mkdir(parents=True, exist_ok=True)
                # Persist enough metadata for replay tools (grid/data/template/etc.)
                run_dir = RUNS_DIR / str(run_name)
                meta_path = run_dir / "batch_meta.json"
                meta: Dict[str, Any] = {}
                try:
                    if meta_path.exists():
                        meta = _read_json(meta_path)  # type: ignore

                    # Baseline (user's original config in this run)
                    baseline_config_id = _find_baseline_config_id(run_dir)
                    st.session_state["baseline_config_id"] = baseline_config_id

                except Exception:
                    meta = {}

                # Estimate bars/day from the dataset so downstream Rolling Starts / Walkforward
                # interpret "min bars" correctly. Falls back to 1 (daily) if inference fails.
                bar_ms = _infer_bar_ms_from_csv(Path(str(data_path)))
                if bar_ms and bar_ms > 0:
                    bars_per_day = int(max(1, round(86_400_000 / float(bar_ms))))
                else:
                    bars_per_day = 1
                meta.update({
                    "run_name": str(run_name),
                    "grid_path": str(grid_path),
                    "data_path": str(data_path),
                    "template": str(template_path),
                    "market_mode": market_mode,
                    "bars_per_day": int(bars_per_day),
                    "ui_written_at": time.time(),
                })
                _write_json(meta_path, meta)
                batch_cmd += ["--no-progress", "--progress-file", str(batch_progress), "--progress-every", "25"]
                pipe.run("batch", batch_cmd, cwd=REPO_ROOT, progress_path=batch_progress.parent)

                run_dir = RUNS_DIR / str(run_name)
                if not run_dir.exists():
                    # fallback: find latest
                    runs2 = _list_runs()
                    run_dir = runs2[0] if runs2 else run_dir

                # 4) Postprocess ranking
                post_cmd: List[str] = [
                    PY,
                    str(REPO_ROOT / "tools" / "postprocess_batch_results.py"),
                    "--from-run",
                    str(run_dir),
                    "--score",
                    str(score),
                    "--top-n",
                    "200",
                ]
                if max_dd_filter and float(max_dd_filter) > 0:
                    post_cmd += ["--max-dd", str(float(max_dd_filter))]

                pipe.run("post", post_cmd, cwd=REPO_ROOT)

                # 5) Optional: run Rolling Starts / Walkforward immediately after Batch
                post_ok = True
                try:
                    do_rs = bool(st.session_state.get("new.do_rs", False))
                    do_wf = bool(st.session_state.get("new.do_wf", False))
                    if do_rs or do_wf:
                        st.info("Running selected stability checks…")

                        frames2 = load_batch_frames(run_dir)
                        survivors2, _src2 = pick_survivors(frames2)
                        survivors_ids = survivors2["config_id"].astype(str).tolist()
                        N = len(survivors_ids)

                        ids_file = run_dir / "post" / "survivor_ids.txt"
                        ids_file.parent.mkdir(parents=True, exist_ok=True)
                        ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

                        bars_per_day = _bars_per_day_from_run_meta(run_dir)

                        rs_root = run_dir / "rolling_starts"
                        wf_root = run_dir / "walkforward"

                        if do_rs and N > 0:
                            start_step = int(st.session_state.get("new.rs.start_step", max(1, int(round(10 * bars_per_day)))))
                            min_bars = int(st.session_state.get("new.rs.min_bars", max(30, int(round(270 * bars_per_day)))))

                            rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}"
                            rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
                            rs_progress.parent.mkdir(parents=True, exist_ok=True)

                            cmd = [
                                PY,
                                "-m",
                                "research.rolling_starts",
                                "--from-run",
                                str(run_dir),
                                "--out",
                                str(rs_out_dir),
                                "--ids",
                                str(ids_file),                                "--top-n",
                                str(N),
                                "--start-step",
                                str(start_step),
                                "--min-bars",
                                str(min_bars),
                                "--seed",
                                "1",
                                "--starting-equity",
                                str(float(st.session_state.get("new.starting_eq", 1000.0) or 1000.0)),
                                "--jobs", "8",
                                "--no-progress",
                                "--progress-file",
                                str(rs_progress),
                                "--progress-every",
                                "10",
                            ]
                            pipe.run("rs", cmd, cwd=REPO_ROOT, progress_path=rs_progress.parent)

                        if do_wf and N > 0:
                            window_days = int(st.session_state.get("new.wf.window_days", 90))
                            step_days = int(st.session_state.get("new.wf.step_days", 30))
                            min_bars = int(st.session_state.get("new.wf.min_bars", 1))
                            expected_window_bars = int(max(1, round(window_days * bars_per_day)))
                            min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
                            jobs = int(st.session_state.get("new.wf.jobs", 8))

                            wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
                            wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
                            wf_progress.parent.mkdir(parents=True, exist_ok=True)

                            cmd = [
                                PY,
                                "-m",
                                "engine.walkforward",
                                "--from-run",
                                str(run_dir),
                                "--out",
                                str(wf_out_dir),                                "--top-n",
                                str(N),
                                "--window-days",
                                str(window_days),
                                "--step-days",
                                str(step_days),
                                "--min-bars",
                                str(min_bars_effective),
                                "--seed",
                                "1",
                                "--starting-equity",
                                str(float(st.session_state.get("new.starting_eq", 1000.0) or 1000.0)),
                                "--jobs",
                                str(jobs),
                                "--no-progress",
                                "--progress-file",
                                str(wf_progress),
                                "--progress-every",
                                "10",
                            ]
                            pipe.run("wf", cmd, cwd=REPO_ROOT, progress_path=wf_progress.parent)

                except Exception as e:
                    post_ok = False
                    st.warning(f"Post-batch tests failed: {e}")


                # Trust layer: write/refresh manifest.json for this run
                try:
                    _ensure_manifest(run_dir)
                except Exception:
                    pass

                st.success(f"Done in {time.time()-t0:.1f}s. Run saved to: {run_dir.name}")

                # Switch to opening this run (so Results can find it)
                st.session_state["selected_run"] = run_dir.name
                st.session_state["ui.open_run_next"] = run_dir.name  # set on next rerun before widget instantiates

                # Reset wizard
                st.session_state["new.step"] = 0

                if post_ok:
                    # After a successful full run, jump to Results by default
                    st.session_state["ui.section_next"] = "2) Results & Autopsy"
                    st.session_state.setdefault("ui.stage", "batch")
                    st.rerun()
                else:
                    st.info("Some selected stability checks failed. Review the logs above, then open the run in Results when ready.")
                    if st.button("Open Results & Autopsy", key="run.goto_results_after_fail"):
                        st.session_state["ui.section_next"] = "2) Results & Autopsy"
                        st.session_state.setdefault("ui.stage", "batch")
                        st.rerun()

            except Exception as e:
                st.error(str(e))
                st.stop()

    st.stop()

# =============================================================================
# Existing run analysis
# =============================================================================

selected_run_name = st.session_state.get("selected_run", "")
if not selected_run_name:
    st.info("No runs found yet. Create a new run from the sidebar.")
    st.stop()

run_dir = RUNS_DIR / selected_run_name
if not run_dir.exists():
    st.error(f"Run folder not found: {run_dir}")
    st.stop()

st.subheader(f"Run: {selected_run_name}")
meta_path = run_dir / "batch_meta.json"
meta = _read_json(meta_path)
if not meta and not meta_path.exists():
    st.warning("This run is missing batch_meta.json (likely an interrupted run). You can still view any results that were written.")
if meta:
    st.caption(f"Data: {meta.get('data','?')}  |  Grid: {meta.get('grid','?')}  |  Template: {meta.get('template','?')}")

# Baseline (user\'s original config in this run) — used to anchor population charts
st.session_state["baseline_config_id"] = _find_baseline_config_id(run_dir)


frames = load_batch_frames(run_dir)
survivors, survivor_source = pick_survivors(frames)


with st.expander("Baseline anchor (debug)", expanded=False):
    _bid = st.session_state.get("baseline_config_id")
    st.write(f"baseline_config_id: `{_bid}`" if _bid else "baseline_config_id: (none)")
    if _bid:
        hits = {}
        try:
            for k in ["full_passed", "sweep_passed", "full_all", "sweep_all", "ranked"]:
                dfk = frames.get(k) if isinstance(frames, dict) else None
                hits[k] = bool(
                    dfk is not None
                    and (not dfk.empty)
                    and ("config_id" in dfk.columns)
                    and (dfk["config_id"].astype(str) == str(_bid)).any()
                )
        except Exception:
            pass
        try:
            hits["survivors_set"] = bool(
                survivors is not None
                and (not survivors.empty)
                and ("config_id" in survivors.columns)
                and (survivors["config_id"].astype(str) == str(_bid)).any()
            )
        except Exception:
            pass
        st.json(hits)


if survivors.empty:
    st.warning("No results found in this run folder (or everything is empty).")
    st.stop()

# Ranked frame (if present) gives score columns
ranked = frames.get("ranked")
if ranked is not None and not ranked.empty and "config_id" in ranked.columns:
    # Keep ranked order, but ensure gates.passed=True if available
    ranked2 = ranked.copy()
    if "gates.passed" in ranked2.columns:
        ranked2 = ranked2[ranked2["gates.passed"].astype(bool)].copy()
    survivors = survivors.drop(columns=[c for c in survivors.columns if c.startswith("score.")], errors="ignore")
    survivors = survivors.merge(
        ranked2[["config_id"] + [c for c in ranked2.columns if c.startswith("score.") or c.startswith("pareto.")]],
        how="left",
        on="config_id",
    )

top_map = _parse_top_artifact_dirs(run_dir)

# Stage directories
rs_root = run_dir / "rolling_starts"
wf_root = run_dir / "walkforward"

# Pick latest/appropriate RS/WF output folders (supports both CLI defaults and UI subfolders)
rs_latest = _pick_latest_rs_dir(run_dir)
wf_latest = _pick_latest_wf_dir(run_dir)

# =============================================================================
# MVP navigation: Build & Run vs Results
# =============================================================================

section_pick = str(st.session_state.get("ui.section", "2) Results & Autopsy"))

if section_pick.startswith("1)"):
    st.header("Build & Run")

    st.caption("Run additional stress tests on the *current* run’s survivors. Results live in the Results & Autopsy section.")
    st.write(f"Survivors detected: **{len(survivors):,}** (source: **{survivor_source}**).")

    bars_per_day = _bars_per_day_from_run_meta(run_dir)

    # ---- Test selection
    st.subheader("Choose tests to run")
    col_a, col_b = st.columns(2)
    with col_a:
        do_rs = st.checkbox("Rolling Starts (start-date fragility)", value=False, key="runner.do_rs")
        st.caption("Same strategy, many start dates → reveals ‘lucky start’ dependence.")
    with col_b:
        do_wf = st.checkbox("Walkforward (out-of-sample-ish windows)", value=False, key="runner.do_wf")
        st.caption("Repeats training/testing through time → reveals generalization vs overfit.")

    # ---- Config panels
    rs_out_dir = None
    wf_out_dir = None
    ids_file = None

    survivors_ids = survivors["config_id"].astype(str).tolist()
    N = len(survivors_ids)

    # Persist survivor ids for reproducible replays
    ids_file = run_dir / "post" / "survivor_ids.txt"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

    if do_rs:
        st.markdown("#### Rolling Starts settings")
        with st.expander("Rolling Starts settings", expanded=True):
            preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="rs.preset")
            # Apply preset defaults once per change (mirrors RS page behavior)
            preset_prev = st.session_state.get("rs.preset_prev", None)
            if preset != preset_prev:
                if preset == "Quick":
                    step_days, min_days = 14, 180
                elif preset == "Thorough":
                    step_days, min_days = 7, 365
                else:
                    step_days, min_days = 10, 270
                st.session_state["rs.start_step"] = int(max(1, round(step_days * bars_per_day)))
                st.session_state["rs.min_bars"] = int(max(30, round(min_days * bars_per_day)))
                st.session_state["rs.preset_prev"] = preset

            start_step = int(
                st.number_input(
                    "Start step (bars)",
                    1,
                    500000,
                    int(st.session_state.get("rs.start_step", max(1, int(round(7 * bars_per_day))))),
                    5,
                    key="rs.start_step",
                )
            )
            min_bars = int(
                st.number_input(
                    "Min bars per start",
                    30,
                    5000000,
                    int(st.session_state.get("rs.min_bars", max(30, int(round(365 * bars_per_day))))),
                    10,
                    key="rs.min_bars",
                )
            )

            rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}"
            st.caption(f"Output: {rs_out_dir}")

    if do_wf:
        st.markdown("#### Walkforward settings")
        with st.expander("Walkforward settings", expanded=True):
            preset = st.selectbox("Preset", ["Quick", "Standard", "Thorough"], index=1, key="wf.preset")
            preset_prev = st.session_state.get("wf.preset_prev", None)
            if preset != preset_prev:
                if preset == "Quick":
                    window_days, step_days = 30, 15
                elif preset == "Thorough":
                    window_days, step_days = 180, 30
                else:
                    window_days, step_days = 90, 30
                st.session_state["wf.window_days"] = int(window_days)
                st.session_state["wf.step_days"] = int(step_days)
                st.session_state["wf.preset_prev"] = preset

            window_days = int(st.number_input("Window (days)", min_value=1, max_value=3650, step=1, key="wf.window_days"))
            step_days = int(st.number_input("Step (days)", min_value=1, max_value=3650, step=1, key="wf.step_days"))

            expected_window_bars = int(max(1, round(window_days * bars_per_day)))
            st.caption(f"Expected bars per window: ~{expected_window_bars:,}. (Min bars must be ≤ this.)")

            max_mb = int(max(1, expected_window_bars))
            if "wf.min_bars" not in st.session_state:
                st.session_state["wf.min_bars"] = int(max_mb)
            if int(st.session_state.get("wf.min_bars", 1)) > int(max_mb):
                st.session_state["wf.min_bars"] = int(max_mb)

            min_bars = int(st.number_input(
                "Min bars per window",
                min_value=1,
                max_value=max_mb,
                step=1,
                key="wf.min_bars",
            ))
            if "wf.jobs" not in st.session_state:
                st.session_state["wf.jobs"] = int(max(1, min(8, (os.cpu_count() or 4))))
            jobs = int(st.session_state.get("wf.jobs", 8))

            min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
            if int(min_bars) != int(min_bars_effective):
                st.warning(
                    f"Min bars ({min_bars}) exceeds expected bars/window (~{expected_window_bars}). "
                    f"Will clamp to {min_bars_effective}."
                )

            wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
            st.caption(f"Output: {wf_out_dir}")

    st.divider()

    run_btn = st.button("Run selected tests", type="primary", disabled=(not do_rs and not do_wf))
    if run_btn:
        try:
            st.subheader("Run monitor")
            stages: List[_PipelineStage] = []
            if do_rs and rs_out_dir is not None:
                stages.append(_PipelineStage("rs", "Rolling Starts"))
            if do_wf and wf_out_dir is not None:
                stages.append(_PipelineStage("wf", "Walkforward"))
            pipe = _PipelineUI(stages)

            if do_rs and rs_out_dir is not None:
                rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
                rs_progress.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    PY,
                    "-m",
                    "research.rolling_starts",
                    "--from-run",
                    str(run_dir),
                    "--out",
                    str(rs_out_dir),
                    "--ids",
                    str(ids_file),
                    "--top-n",
                    str(N),
                    "--start-step",
                    str(int(st.session_state.get("rs.start_step", 1))),
                    "--min-bars",
                    str(int(st.session_state.get("rs.min_bars", 30))),
                    "--seed",
                    "1",
                    "--starting-equity",
                    str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                    "--jobs", "8",
                    "--no-progress",
                    "--progress-file",
                    str(rs_progress),
                    "--progress-every",
                    "10",
                ]
                pipe.run("rs", cmd, cwd=REPO_ROOT, progress_path=rs_progress.parent)

            if do_wf and wf_out_dir is not None:
                wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
                wf_progress.parent.mkdir(parents=True, exist_ok=True)
                window_days = int(st.session_state.get("wf.window_days", 90))
                step_days = int(st.session_state.get("wf.step_days", 30))
                min_bars = int(st.session_state.get("wf.min_bars", 1))
                expected_window_bars = int(max(1, round(window_days * bars_per_day)))
                min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
                cmd = [
                    PY,
                    "-m",
                    "engine.walkforward",
                    "--from-run",
                    str(run_dir),
                    "--out",
                    str(wf_out_dir),
                    "--top-n",
                    str(N),
                    "--window-days",
                    str(window_days),
                    "--step-days",
                    str(step_days),
                    "--min-bars",
                    str(min_bars_effective),
                    "--seed",
                    "1",
                    "--starting-equity",
                    str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                    "--jobs",
                    str(int(st.session_state.get("wf.jobs", 8))),
                    "--no-progress",
                    "--progress-file",
                    str(wf_progress),
                    "--progress-every",
                    "10",
                ]
                pipe.run("wf", cmd, cwd=REPO_ROOT, progress_path=wf_progress.parent)

            st.success("Selected tests completed.")

            # Trust layer: refresh manifest now that RS/WF evidence may have changed
            try:
                _ensure_manifest(run_dir)
            except Exception:
                pass
            st.session_state["ui.section_next"] = "2) Results & Autopsy"
            if do_wf:
                st.session_state["ui.stage"] = "wf"
            elif do_rs:
                st.session_state["ui.stage"] = "rs"
            else:
                st.session_state["ui.stage"] = "grand"
            st.rerun()

        except Exception as e:
            st.error(str(e))
            st.stop()

    st.stop()

# ---- Results section
st.header("Results & Autopsy")

# -----------------------------------------------------------------------------
# Run status strip (MVP contract: Results is view-only)
# -----------------------------------------------------------------------------
total_cfg = int(len(survivors)) if survivors is not None else 0

# Batch is "ready" if we have a run loaded.
batch_icon = "✅"
batch_label = "ready"

# Rolling Starts coverage
rs_done = 0
try:
    if rs_latest is not None:
        _rs_sum_status = load_rs_summary(run_dir, rs_latest)
        if _rs_sum_status is not None and not _rs_sum_status.empty and "config_id" in _rs_sum_status.columns:
            rs_done = int(_rs_sum_status["config_id"].astype(str).nunique())
except Exception:
    rs_done = 0

if rs_done <= 0:
    rs_icon = "⚠️"
    rs_label = "missing"
elif total_cfg > 0 and rs_done < total_cfg:
    rs_icon = "⚠️"
    rs_label = f"partial ({rs_done}/{total_cfg})"
else:
    rs_icon = "✅"
    rs_label = "ready"

# Walkforward coverage
wf_done = 0
try:
    if wf_latest is not None:
        _wf_sum_status = load_wf_summary(wf_latest)
        if _wf_sum_status is not None and not _wf_sum_status.empty and "config_id" in _wf_sum_status.columns:
            wf_done = int(_wf_sum_status["config_id"].astype(str).nunique())
except Exception:
    wf_done = 0

if wf_done <= 0:
    wf_icon = "⚠️"
    wf_label = "missing"
elif total_cfg > 0 and wf_done < total_cfg:
    wf_icon = "⚠️"
    wf_label = f"partial ({wf_done}/{total_cfg})"
else:
    wf_icon = "✅"
    wf_label = "ready"

missing_rs = (rs_done <= 0) or (total_cfg > 0 and rs_done < total_cfg)
missing_wf = (wf_done <= 0) or (total_cfg > 0 and wf_done < total_cfg)

strip1, strip2, strip3, strip4 = st.columns([1.1, 1.5, 1.2, 1.6])
with strip1:
    st.markdown(f"**Batch:** {batch_icon} {batch_label}")
with strip2:
    st.markdown(f"**Rolling Starts:** {rs_icon} {rs_label}")
with strip3:
    st.markdown(f"**Walkforward:** {wf_icon} {wf_label}")
with strip4:
    if missing_rs or missing_wf:
        if st.button("Go run missing tests", type="primary", key="results.go_run_missing"):
            st.session_state["runner.do_rs"] = bool(missing_rs)
            st.session_state["runner.do_wf"] = bool(missing_wf)
            st.session_state["ui.section_next"] = "1) Build & Run"
            st.rerun()
    else:
        st.caption("Results is view-only. Run tests from Build & Run.")


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Manifest (used for exports; Trust & comparability UI removed for now)
# -----------------------------------------------------------------------------
manifest: Dict[str, Any] = {}
try:
    manifest = _ensure_manifest(run_dir)
except Exception:
    manifest = {}

# Force cockpit mode: show the unified Grand Verdict cockpit only.
st.session_state["ui.stage"] = "grand"
stage_pick = "grand"

st.divider()
# =============================================================================
# Stage A: Batch
# =============================================================================

if stage_pick == "batch":
    st.write("### A) Batch results (sweep + rerun)")
    st.caption(f"Survivor source: **{survivor_source}**. (Increase rerun_n if you want more survivors in full_passed.)")

    # Questions
    with st.expander("Batch questions (filters)", expanded=True):
        batch_ans = _question_ui(batch_questions(), key_prefix="q.batch")
    dfA = apply_stage_eval(survivors, stage_key="batch", questions=batch_questions(), answers=batch_ans)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.batch.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.batch.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.batch.fail")

    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")

    df_show = dfA[dfA["batch.verdict"].isin(keep)].copy()

    # ---------------------------------------------------------------------
    # Visual scan (turn the table into something a human can reason about)
    # ---------------------------------------------------------------------
    st.write("#### Quick visual scan")

    id_col = "config_id"
    label_col = _pick_col(df_show, ["config.label", "label", "config_label"])
    verdict_col = _pick_col(df_show, ["batch.verdict", "verdict", "batch_verdict"])

    profit_col = _pick_col(df_show, ["equity.net_profit_ex_cashflows", "equity.net_profit", "equity.net_profit_ex_cashflow"])
    dd_col = _pick_col(df_show, ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown"])
    trades_col = _pick_col(df_show, ["trades_summary.trades_closed", "trades.closed", "trades_closed", "trades"])

    calmar_col = _pick_col(df_show, ["score.calmar_equity", "performance.calmar", "calmar"])

    if profit_col and dd_col and px is not None and go is not None and not df_show.empty:
        plot_df = df_show.copy()
        plot_df["_profit"] = _to_float_series(plot_df[profit_col])
        plot_df["_dd"] = _drawdown_to_frac(plot_df[dd_col])
        if trades_col:
            plot_df["_trades"] = _to_float_series(plot_df[trades_col]).fillna(0.0)
        else:
            plot_df["_trades"] = 0.0

        plot_df["_label"] = plot_df[label_col].astype(str) if label_col else ""
        plot_df["_verdict"] = plot_df[verdict_col].astype(str) if verdict_col else "?"

        # Frontier hygiene: by default, ignore ultra-low-activity configs so the frontier is meaningful.
        # (Otherwise you often get near-zero drawdown "do nothing" configs anchoring the left edge.)
        max_tr = int(max(0.0, float(pd.to_numeric(plot_df["_trades"], errors="coerce").max() or 0.0)))
        max_slider = max(0, min(200, max_tr))
        default_tr = 3 if max_slider >= 3 else max_slider
        frontier_min_trades = st.slider(
            "Pareto frontier: min trades",
            min_value=0,
            max_value=max_slider,
            value=default_tr,
            step=1,
            help="Frontier is shown only among configs with at least this many trades (to avoid 'do nothing' near-zero DD anchors).",
        )

        frontier_df = plot_df
        if frontier_min_trades > 0:
            frontier_df = plot_df[plot_df["_trades"] >= frontier_min_trades].copy()

        # Scatter: profit vs drawdown
        fig = px.scatter(
            plot_df,
            x="_dd",
            y="_profit",
            color="_verdict",
            size="_trades",
            hover_data={
                id_col: True,
                "_label": True,
                "_profit": ":.2f",
                "_dd": ":.4f",
                "_trades": ":.0f",
            },
            render_mode="webgl",
            title="Return vs max drawdown (each dot is a strategy)",
        )
        fig.update_layout(
            xaxis_title="Max drawdown (fraction, lower is better)",
            yaxis_title="Net profit (excluding deposits)",
            legend_title_text="Batch verdict",
            height=520,
            margin=dict(l=10, r=10, t=50, b=10),
        )

        # Pareto frontier overlay (can't improve profit without worsening drawdown)
        frontier = _pareto_frontier_rows(frontier_df, "_dd", "_profit")
        if not frontier.empty:
            hover_text = None
            if "_label" in frontier.columns and "config_id" in frontier.columns:
                hover_text = frontier["_label"].astype(str) + " • " + frontier["config_id"].astype(str)
            elif "_label" in frontier.columns:
                hover_text = frontier["_label"].astype(str)
            elif "config_id" in frontier.columns:
                hover_text = frontier["config_id"].astype(str)

            fig.add_trace(
                go.Scatter(
                    x=frontier["_dd"],
                    y=frontier["_profit"],
                    mode="lines+markers",
                    name="Pareto frontier",
                    text=hover_text,
                    hovertemplate="%{text}<br>dd=%{x:.4f}<br>profit=%{y:.2f}<extra></extra>",
                    line=dict(width=2),
                )
            )


        # Baseline marker (user's original config)
        try:
            _bid = st.session_state.get("baseline_config_id")
            if _bid:
                bx = by = None

                # Prefer the plotted survivor frame (fast path)
                if id_col in plot_df.columns:
                    _b = plot_df[plot_df[id_col].astype(str) == str(_bid)].head(1)
                    if not _b.empty:
                        bx = float(_b["_dd"].iloc[0])
                        by = float(_b["_profit"].iloc[0])

                # Fallback: even if baseline was filtered out of this view, try broader tables (full/sweep).
                if (
                    (bx is None or by is None or (not math.isfinite(float(bx))) or (not math.isfinite(float(by))))
                    and profit_col
                    and dd_col
                ):
                    df_full = None
                    for _k in ["full_all", "sweep_all", "sweep_passed"]:
                        _tmp = frames.get(_k)
                        if _tmp is not None and hasattr(_tmp, "empty") and (not _tmp.empty):
                            df_full = _tmp
                            break
                    if df_full is not None and (not df_full.empty) and ("config_id" in df_full.columns):
                        profit_cands = ["equity.net_profit_ex_cashflows", "equity.net_profit", "net_profit_ex_cashflows", "net_profit", "profit"]
                        dd_cands = ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown", "equity.max_dd", "max_drawdown", "max_dd", "dd"]

                        profit_use = profit_col if profit_col in df_full.columns else _pick_col(df_full, profit_cands)
                        dd_use = dd_col if dd_col in df_full.columns else _pick_col(df_full, dd_cands)

                        if profit_use and dd_use:
                            _b2 = df_full[df_full["config_id"].astype(str) == str(_bid)].head(1)
                            if not _b2.empty:
                                bx = float(_drawdown_to_frac(pd.to_numeric(_b2[dd_use], errors="coerce")).iloc[0])
                                by = float(pd.to_numeric(_b2[profit_use], errors="coerce").iloc[0])


                if bx is not None and by is not None and math.isfinite(float(bx)) and math.isfinite(float(by)):
                    fig.add_trace(
                        go.Scatter(
                            x=[float(bx)],
                            y=[float(by)],
                            mode="markers",
                            name="Your strategy",
                            marker=dict(
                                size=18,
                                symbol="star",
                                color="rgba(17,24,39,0.95)",
                                line=dict(width=2, color="rgba(255,255,255,0.95)"),
                            ),
                            hovertemplate="Your strategy (baseline)<br>config={}<br>dd={:.4f}<br>profit={:.2f}<extra></extra>".format(
                                str(_bid), float(bx), float(by)
                            ),
                            showlegend=True,
                        )
                    )
        except Exception:
            pass
        _plotly(fig)

        # Quick sanity: list frontier points (helps confirm it's "real" and not plotting artifacts)
        with st.expander("Pareto frontier points", expanded=False):
            if frontier.empty:
                st.caption("No frontier points available (check filters).")
            else:
                show_cols = [c for c in ["config_id", "_label", "_verdict", "_trades", "_dd", "_profit"] if c in frontier.columns]
                st.dataframe(frontier[show_cols].sort_values("_dd", ascending=True), width="stretch")
    else:
        st.info("Scatter plot unavailable (missing Plotly or required columns).")

    # ---------------------------------------------------------------------
    # Top candidates as cards (humans think in tradeoffs, not columns)
    # ---------------------------------------------------------------------
    st.write("#### Top candidates (cards)")
    st.markdown("<div style='color: rgba(49,51,63,0.75); font-size: 0.92rem; margin-top: -6px;'>Each strip summarizes many runs. Each value is total return % for that scenario. Box = typical zone (middle 50%). Whiskers = bad→good range.</div>", unsafe_allow_html=True)

    rank_col = _pick_col(df_show, ["score.profit", profit_col] if profit_col else ["score.profit"])
    if rank_col is None:
        rank_col = profit_col

    cards_df = df_show.copy()
    if rank_col:
        cards_df["_rank"] = _to_float_series(cards_df[rank_col])
        cards_df = cards_df.sort_values("_rank", ascending=False)
    cards_df = cards_df.head(12).copy()

    if cards_df.empty:
        st.info("No rows to show.")
    else:
        cols_cards = st.columns(3)
        for i, (_, r) in enumerate(cards_df.iterrows()):
            cfg_id = str(r.get(id_col, ""))
            label = str(r.get(label_col, cfg_id)) if label_col else cfg_id
            verdict = str(r.get(verdict_col, "")) if verdict_col else ""

            profit_v = float(r.get(profit_col, float("nan"))) if profit_col else float("nan")
            dd_v = float(r.get(dd_col, float("nan"))) if dd_col else float("nan")
            dd_frac = float(_drawdown_to_frac(pd.Series([dd_v])).iloc[0]) if dd_col else float("nan")
            trades_v = int(float(r.get(trades_col, 0))) if trades_col and str(r.get(trades_col, "")).strip() != "" else 0
            calmar_v = float(r.get(calmar_col, float("nan"))) if calmar_col else float("nan")

            with cols_cards[i % 3]:
                with st.container():
                    st.write(f"**{label}**")
                    st.caption(f"{cfg_id} • {verdict}")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Profit", _fmt_money(profit_v) if profit_col else "n/a")
                    m2.metric("Max DD", _fmt_pct(dd_frac) if dd_col else "n/a")
                    m3.metric("Trades", f"{trades_v}" if trades_col else "n/a")
                    if calmar_col:
                        st.caption(f"Calmar: {_fmt_num(calmar_v, digits=3)}")

                    if st.button("Inspect", key=f"batch.inspect.{cfg_id}"):
                        st.session_state["ui.batch.inspect_id"] = cfg_id
                        st.session_state["ui.batch.scroll_to_inspect"] = True
                        st.rerun()

    # ---------------------------------------------------------------------
    # Heatmap table (percentiles) for quick pattern recognition
    # ---------------------------------------------------------------------
    st.write("#### Quick scan heatmap (scores among shown rows — higher is better)")
    heat_base = df_show.copy()
    n = len(heat_base)

    heat = pd.DataFrame()
    heat["config_id"] = heat_base["config_id"].astype(str).str.strip()
    if label_col:
        heat["label"] = heat_base[label_col].astype(str)
    if verdict_col:
        heat["verdict"] = heat_base[verdict_col].astype(str)

    # Raw helpers
    if profit_col:
        heat["profit"] = _to_float_series(heat_base[profit_col])
        heat["profit_%ile"] = (_goodness_percentile(heat_base[profit_col], low_is_good=False) * 100.0).round(1)
    if dd_col:
        dd_frac = _drawdown_to_frac(heat_base[dd_col])
        heat["max_dd"] = dd_frac
        heat["dd_good_%ile"] = (_goodness_percentile(dd_frac, low_is_good=True) * 100.0).round(1)
    if trades_col:
        heat["trades"] = _to_float_series(heat_base[trades_col]).round(0)
        # More trades generally = more evidence; treat low-is-bad (so low_is_good=False)
        heat["trades_%ile"] = (_goodness_percentile(heat_base[trades_col], low_is_good=False) * 100.0).round(1)
    if calmar_col:
        heat["calmar"] = _to_float_series(heat_base[calmar_col])
        heat["calmar_%ile"] = (_goodness_percentile(heat_base[calmar_col], low_is_good=False) * 100.0).round(1)

    # Display with color on the percentile columns
    pct_cols = [c for c in heat.columns if c.endswith("%ile")]
    disp_cols = ["config_id"] + (["label"] if "label" in heat.columns else []) + (["verdict"] if "verdict" in heat.columns else [])
    disp_cols += [c for c in ["profit", "max_dd", "trades", "calmar"] if c in heat.columns]
    disp_cols += pct_cols

    heat_disp = heat[disp_cols].copy()
    sty = heat_disp.style
    for c in pct_cols:
        sty = sty.background_gradient(subset=[c], cmap="RdYlGn")

    st.dataframe(sty, width="stretch", height=420)

    # ---------------------------------------------------------------------
    # Inspect + Replay
    # ---------------------------------------------------------------------
    st.write("#### Inspect a strategy")
    default_pick = st.session_state.get("ui.batch.inspect_id")
    if default_pick not in set(df_show["config_id"].astype(str)):
        default_pick = str(df_show["config_id"].iloc[0]) if not df_show.empty else None

    if default_pick:
        pick = st.selectbox(
            "Choose a config_id to inspect / replay",
            options=df_show["config_id"].astype(str).tolist(),
            index=df_show["config_id"].astype(str).tolist().index(default_pick),
            key="ui.batch.pick",
        )
        batch_profit_ex_cf = None
        row = df_show[df_show["config_id"].astype(str) == str(pick)].head(1)
        if not row.empty:
            r = row.iloc[0].to_dict()
            # Keep a reference for replay sanity checks
            batch_profit_ex_cf = None
            try:
                if profit_col:
                    batch_profit_ex_cf = float(r.get(profit_col))
            except Exception:
                batch_profit_ex_cf = None

            c1, c2, c3, c4 = st.columns(4)
            if profit_col:
                c1.metric("Profit (ex cashflows)", _fmt_money(r.get(profit_col)))
            if dd_col:
                c2.metric("Max DD", _fmt_pct(_drawdown_to_frac(pd.Series([r.get(dd_col)])).iloc[0]))
            if trades_col:
                c3.metric("Trades", f"{int(float(r.get(trades_col, 0) or 0))}")
            if calmar_col:
                c4.metric("Calmar", _fmt_num(r.get(calmar_col), digits=3))

        # Artifacts locations
        replay_dir = run_dir / "replay_cache" / str(pick)
        art_dir = replay_dir if (replay_dir / "equity_curve.csv").exists() else top_map.get(str(pick), replay_dir)

        replay_dl_items: List[Tuple[str, bytes, str]] = []
        eq_path = art_dir / "equity_curve.csv"
        can_replay = (run_dir / "configs_resolved.jsonl").exists()
        if not eq_path.exists():

            if str(st.session_state.get("ui.replay.primary_controls_for", "")) == str(pick):

                st.info("Replay artifacts are missing. Use the **Generate replay artifacts** button above in the Build sheet.")

            else:

                _render_replay_artifacts_controls(

                    run_dir=run_dir,

                    pick=str(pick),

                    replay_dir=replay_dir,

                    has_core_artifacts=False,

                    can_replay=bool(can_replay),

                    key_prefix="replay.fallback.batch",

                    show_when_ready=False,

                )

        if eq_path.exists():
            try:
                eq_df = pd.read_csv(eq_path)
                st.caption(f"Replay artifacts dir: `{art_dir}`")

                if "equity" in eq_df.columns:
                    # Use dt column when present, otherwise fall back to first column
                    tcol = "dt" if "dt" in eq_df.columns else eq_df.columns[0]
                    t = pd.to_datetime(eq_df[tcol], errors="coerce")
                    equity = pd.to_numeric(eq_df["equity"], errors="coerce")

                    start_eq = float(equity.iloc[0]) if len(equity) else float("nan")
                    end_eq = float(equity.iloc[-1]) if len(equity) else float("nan")
                    cashflow_total = 0.0
                    if "cashflow" in eq_df.columns:
                        cashflow_total = float(pd.to_numeric(eq_df["cashflow"], errors="coerce").fillna(0.0).sum())
                    replay_profit_ex_cf = float(end_eq - start_eq - cashflow_total) if (math.isfinite(end_eq) and math.isfinite(start_eq)) else float("nan")

                    cA, cB, cC, cD = st.columns(4)
                    cA.metric("Replay start equity", _fmt_money(start_eq))
                    cB.metric("Replay end equity", _fmt_money(end_eq))
                    cC.metric("Replay cashflows", _fmt_money(cashflow_total))
                    cD.metric("Replay profit (ex cashflows)", _fmt_money(replay_profit_ex_cf))

                    # Warn when replay doesn't match the batch row (common sign of loading the wrong config/dataset)
                    try:
                        if batch_profit_ex_cf is not None and math.isfinite(replay_profit_ex_cf):
                            if abs(float(batch_profit_ex_cf) - float(replay_profit_ex_cf)) > max(5.0, 0.01 * abs(float(batch_profit_ex_cf))):
                                st.warning(
                                    "Replay result does **not** match the batch row profit. "
                                    "This usually means the replay loaded the wrong config payload (e.g., didn't unwrap normalized config), "
                                    "or the replay is using a different dataset/seed/starting equity."
                                )
                    except Exception:
                        pass

                                        # -----------------------------------------------------------------
                    # Replay charts (juicy + interpretable)
                    # -----------------------------------------------------------------
                    trades_path = art_dir / "trades.csv"
                    fills_path = art_dir / "fills.csv"

                    # Build a plot-friendly frame
                    try:
                        t_idx = t if t.notna().any() else pd.to_datetime(eq_df[tcol], errors="coerce")
                    except Exception:
                        t_idx = pd.to_datetime(eq_df[tcol], errors="coerce")

                    plot_df = pd.DataFrame(
                        {
                            "dt": t_idx,
                            "equity": pd.to_numeric(eq_df.get("equity"), errors="coerce"),
                            "cash": pd.to_numeric(eq_df.get("cash"), errors="coerce") if "cash" in eq_df.columns else np.nan,
                            "price": pd.to_numeric(eq_df.get("price"), errors="coerce") if "price" in eq_df.columns else np.nan,
                            "pos_qty": pd.to_numeric(eq_df.get("pos_qty"), errors="coerce") if "pos_qty" in eq_df.columns else np.nan,
                            "cashflow": pd.to_numeric(eq_df.get("cashflow"), errors="coerce") if "cashflow" in eq_df.columns else 0.0,
                        }
                    )
                    plot_df = plot_df.dropna(subset=["dt"]).sort_values("dt")
                    plot_df["pos_value"] = plot_df["pos_qty"] * plot_df["price"] if ("pos_qty" in plot_df.columns and "price" in plot_df.columns) else np.nan
                    plot_df["exposure"] = np.nan
                    try:
                        pv = pd.to_numeric(plot_df["pos_value"], errors="coerce")
                        eqv = pd.to_numeric(plot_df["equity"], errors="coerce")
                        plot_df["exposure"] = (pv.abs() / eqv.replace(0.0, np.nan)).clip(0.0, 5.0)
                    except Exception:
                        pass

                    # Load trades for markers (if present)
                    td = None
                    if trades_path.exists():
                        try:
                            td = pd.read_csv(trades_path)
                        except Exception:
                            td = None

                    def _nearest_y(ts: pd.Series) -> np.ndarray:
                        # Nearest equity values for a series of datetimes
                        x = plot_df["dt"].to_numpy(dtype="datetime64[ns]")
                        y = plot_df["equity"].to_numpy(dtype=float)
                        t_arr = pd.to_datetime(ts, errors="coerce").to_numpy(dtype="datetime64[ns]")
                        out = np.full(len(t_arr), np.nan, dtype=float)
                        if len(x) == 0:
                            return out
                        # Searchsorted gives insertion point; compare neighbors to pick nearest
                        idxs = np.searchsorted(x, t_arr, side="left")
                        idxs = np.clip(idxs, 0, len(x) - 1)
                        prev = np.clip(idxs - 1, 0, len(x) - 1)
                        # Choose prev when it's closer
                        choose_prev = np.abs(t_arr - x[prev]) <= np.abs(x[idxs] - t_arr)
                        best = np.where(choose_prev, prev, idxs)
                        out = y[best]
                        return out

                    tab_eq, tab_cash, tab_exp = st.tabs(["Equity + Trades", "Cash vs Position", "Exposure"])

                    with tab_eq:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["equity"], mode="lines", name="Equity"))

                        # Deposit / cashflow markers (spot/DCA interpretable moment)
                        try:
                            cf = pd.to_numeric(plot_df["cashflow"], errors="coerce").fillna(0.0)
                            mask = cf.abs() > 1e-9
                            if mask.any():
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_df.loc[mask, "dt"],
                                        y=plot_df.loc[mask, "equity"],
                                        mode="markers",
                                        name="Cashflow",
                                        marker=dict(size=6, symbol="circle-open"),
                                        hovertemplate="dt=%{x}<br>cashflow=%{customdata:,.2f}<extra></extra>",
                                        customdata=cf.loc[mask].to_numpy(),
                                    )
                                )
                        except Exception:
                            pass

                        # Trade markers
                        if td is not None and not td.empty and "entry_dt" in td.columns:
                            try:
                                entry_t = pd.to_datetime(td["entry_dt"], errors="coerce")
                                entry_y = _nearest_y(entry_t)
                                fig.add_trace(
                                    go.Scatter(
                                        x=entry_t,
                                        y=entry_y,
                                        mode="markers",
                                        name="Entry",
                                        marker=dict(size=8, symbol="triangle-up"),
                                        hovertemplate="entry=%{x}<extra></extra>",
                                    )
                                )
                            except Exception:
                                pass

                        if td is not None and not td.empty and "exit_dt" in td.columns:
                            try:
                                exit_t = pd.to_datetime(td["exit_dt"], errors="coerce")
                                exit_mask = exit_t.notna()
                                exit_t2 = exit_t[exit_mask]
                                exit_y = _nearest_y(exit_t2)
                                # If we have net_pnl, pass it as customdata so you can see winners/losers on hover
                                custom = None
                                if "net_pnl" in td.columns:
                                    custom = pd.to_numeric(td.loc[exit_mask, "net_pnl"], errors="coerce").to_numpy()
                                fig.add_trace(
                                    go.Scatter(
                                        x=exit_t2,
                                        y=exit_y,
                                        mode="markers",
                                        name="Exit",
                                        marker=dict(size=8, symbol="triangle-down"),
                                        hovertemplate="exit=%{x}<br>net_pnl=%{customdata:,.2f}<extra></extra>",
                                        customdata=custom,
                                    )
                                )
                            except Exception:
                                pass

                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=420, legend=dict(orientation="h"))
                        _plotly(fig)

                        # Drawdown (with max-DD episode shading)
                        try:
                            eq2 = pd.to_numeric(plot_df["equity"], errors="coerce").fillna(method="ffill")
                            dt2 = pd.to_datetime(plot_df["dt"], errors="coerce")
                            peak = eq2.cummax()
                            dd = (eq2 / peak - 1.0).fillna(0.0)

                            # Identify max drawdown episode (peak -> trough -> recovery if any)
                            x0 = None
                            x1 = None
                            try:
                                if len(dd):
                                    trough_i = int(np.nanargmin(dd.to_numpy()))
                                    peak_val = float(peak.iloc[trough_i])

                                    # start = last time equity touched the high-water mark before the trough
                                    pre_eq = eq2.iloc[:trough_i]  # exclude trough itself
                                    if len(pre_eq):
                                        hit = pre_eq >= peak_val * (1.0 - 1e-9)
                                        if hit.any():
                                            start_pos = int(np.flatnonzero(hit.to_numpy())[-1])
                                            x0 = dt2.iloc[start_pos]

                                    # recovery = first time AFTER the trough equity >= peak
                                    post_eq = eq2.iloc[trough_i + 1 :]
                                    if len(post_eq):
                                        rec_mask = post_eq >= peak_val * (1.0 - 1e-9)
                                        if rec_mask.any():
                                            rec_rel = int(np.flatnonzero(rec_mask.to_numpy())[0])
                                            x1 = dt2.iloc[trough_i + 1 + rec_rel]

                                    # If never recovered, shade peak->trough so the user still sees "the bad zone"
                                    if x0 is not None and x1 is None:
                                        x1 = dt2.iloc[min(trough_i, len(dt2) - 1)]
                            except Exception:
                                x0 = None
                                x1 = None

                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=dt2, y=dd, mode="lines", name="Drawdown"))

                            if x0 is not None and x1 is not None and pd.notna(x0) and pd.notna(x1):
                                fig2.add_vrect(
                                    x0=x0,
                                    x1=x1,
                                    fillcolor="rgba(200,0,0,0.08)",
                                    line_width=0,
                                    annotation_text="Max DD episode",
                                    annotation_position="top left",
                                    annotation_font_size=10,
                                )

                            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=260, legend=dict(orientation="h"))
                            _plotly(fig2)
                        except Exception:
                            pass

                    with tab_cash:
                        fig = go.Figure()
                        if "cash" in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["cash"], mode="lines", name="Cash"))
                        if "pos_value" in plot_df.columns:
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["pos_value"], mode="lines", name="Position value"))
                        fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["equity"], mode="lines", name="Equity"))
                        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=360, legend=dict(orientation="h"))
                        _plotly(fig)

                    with tab_exp:
                        if "exposure" in plot_df.columns and plot_df["exposure"].notna().any():
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=plot_df["dt"], y=plot_df["exposure"], mode="lines", name="Exposure (pos_value / equity)"))
                            fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320, legend=dict(orientation="h"))
                            _plotly(fig)
                        else:
                            st.caption("Exposure chart unavailable (missing pos_qty/price/equity columns).")


                    # -----------------------------------------------------------------
                    # Strategy story (interpretable summary from replay artifacts)
                    # -----------------------------------------------------------------
                    with st.expander("Strategy story (replay)", expanded=True):
                        # --- Drawdown story ---
                        try:
                            eq_series = pd.to_numeric(plot_df["equity"], errors="coerce").fillna(method="ffill")
                            dt_series = pd.to_datetime(plot_df["dt"], errors="coerce")
                            peak = eq_series.cummax()
                            dd = (eq_series / peak - 1.0).fillna(0.0)

                            dd_min = float(dd.min()) if len(dd) else float("nan")
                            dd_trough_i = int(np.nanargmin(dd.to_numpy())) if len(dd) else None

                            dd_start_dt = None
                            dd_trough_dt = None
                            dd_recover_dt = None
                            dd_peak_val = None

                            if dd_trough_i is not None and math.isfinite(dd_min):
                                dd_trough_dt = dt_series.iloc[dd_trough_i]
                                dd_peak_val = float(peak.iloc[dd_trough_i])
                            
                                # peak date = last time equity touched the high-water mark before the trough
                                pre_eq = eq_series.iloc[:dd_trough_i]  # exclude trough itself
                                if len(pre_eq):
                                    hit = pre_eq >= dd_peak_val * (1.0 - 1e-9)
                                    if hit.any():
                                        start_pos = int(np.flatnonzero(hit.to_numpy())[-1])
                                        dd_start_dt = dt_series.iloc[start_pos]
                                    else:
                                        dd_start_dt = dt_series.iloc[0]
                                else:
                                    dd_start_dt = dt_series.iloc[0]
                            
                                # recovery date = first time AFTER trough equity >= peak value
                                post_eq = eq_series.iloc[dd_trough_i + 1 :]
                                if len(post_eq):
                                    rec_mask = post_eq >= dd_peak_val * (1.0 - 1e-9)
                                    if rec_mask.any():
                                        rec_rel = int(np.flatnonzero(rec_mask.to_numpy())[0])
                                        dd_recover_dt = dt_series.iloc[dd_trough_i + 1 + rec_rel]

                            # underwater longest segment
                            underwater = eq_series < peak * (1.0 - 1e-12)
                            uw = underwater.astype(int)
                            seg = (uw.diff().fillna(0).abs() > 0).cumsum()

                            longest_days = None
                            longest_start = None
                            longest_end = None
                            if underwater.any():
                                for sid, grp in plot_df.loc[underwater].groupby(seg[underwater]):
                                    dts = pd.to_datetime(grp["dt"], errors="coerce")
                                    if dts.notna().any():
                                        dur = (dts.max() - dts.min()).total_seconds() / 86400.0
                                        if (longest_days is None) or (dur > longest_days):
                                            longest_days = dur
                                            longest_start = dts.min()
                                            longest_end = dts.max()

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Max drawdown", _fmt_pct(dd_min))
                            c2.metric("DD peak", f"{dd_start_dt.date()}" if dd_start_dt is not None else "n/a")
                            c3.metric("DD trough", f"{dd_trough_dt.date()}" if dd_trough_dt is not None else "n/a")
                            if dd_recover_dt is not None and dd_start_dt is not None:
                                days = (dd_recover_dt - dd_start_dt).total_seconds() / 86400.0
                                c4.metric("DD to recovery", f"{days:.0f} days")
                                st.caption(f"Recovery: {dd_recover_dt.date()}")
                            else:
                                c4.metric("DD to recovery", "not recovered" if dd_start_dt is not None else "n/a")

                            if longest_days is not None and longest_start is not None and longest_end is not None:
                                st.caption(f"Longest underwater: {longest_days:.0f} days ({longest_start.date()} → {longest_end.date()})")
                        except Exception:
                            st.info("Drawdown story unavailable (missing equity series).")

                        st.divider()

                        # --- Trades story ---
                        if td is None or td.empty:
                            st.info("No trades.csv found for this replay yet.")
                        else:
                            tdf = td.copy()
                            if "net_pnl" in tdf.columns:
                                pnl = pd.to_numeric(tdf["net_pnl"], errors="coerce")
                            elif "pnl" in tdf.columns:
                                pnl = pd.to_numeric(tdf["pnl"], errors="coerce")
                            else:
                                pnl = pd.Series([np.nan] * len(tdf))

                            wins = pnl > 0
                            win_rate = float(wins.mean()) if len(pnl.dropna()) else float("nan")
                            gross_win = float(pnl[wins].sum()) if wins.any() else 0.0
                            gross_loss = float(pnl[~wins].sum()) if (~wins).any() else 0.0
                            profit_factor = (gross_win / abs(gross_loss)) if gross_loss < 0 else float("inf") if gross_win > 0 else float("nan")

                            best_trade = float(pnl.max()) if pnl.notna().any() else float("nan")
                            worst_trade = float(pnl.min()) if pnl.notna().any() else float("nan")

                            hold_days = None
                            if "entry_dt" in tdf.columns and "exit_dt" in tdf.columns:
                                ed = pd.to_datetime(tdf["entry_dt"], errors="coerce")
                                xd = pd.to_datetime(tdf["exit_dt"], errors="coerce")
                                hold = (xd - ed).dt.total_seconds() / 86400.0
                                hold_days = float(hold.mean()) if hold.notna().any() else None

                            exp_mean = float(pd.to_numeric(plot_df.get("exposure"), errors="coerce").mean()) if "exposure" in plot_df.columns else float("nan")
                            exp_max = float(pd.to_numeric(plot_df.get("exposure"), errors="coerce").max()) if "exposure" in plot_df.columns else float("nan")

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Win rate", _fmt_pct(win_rate))
                            c2.metric("Profit factor", _fmt_num(profit_factor, digits=2) if math.isfinite(profit_factor) else "∞" if profit_factor == float("inf") else "n/a")
                            c3.metric("Best / Worst trade", f"{_fmt_money(best_trade)} / {_fmt_money(worst_trade)}")
                            c4.metric("Avg exposure", _fmt_pct(exp_mean))

                            if hold_days is not None:
                                st.caption(f"Avg hold time: {hold_days:.1f} days · Max exposure: {_fmt_pct(exp_max)}")

                            show_cols = []
                            for c in ["entry_dt", "exit_dt", "entry_price", "exit_price", "qty", "net_pnl", "fees", "reason", "exit_reason"]:
                                if c in tdf.columns:
                                    show_cols.append(c)
                            if "net_pnl" not in tdf.columns:
                                tdf["net_pnl"] = pnl

                            st.write("**Top trades**")
                            left, right = st.columns(2)
                            with left:
                                st.caption("Best (by net_pnl)")
                                st.dataframe(tdf.sort_values("net_pnl", ascending=False).head(5)[show_cols or ["net_pnl"]], width="stretch", height=220)
                            with right:
                                st.caption("Worst (by net_pnl)")
                                st.dataframe(tdf.sort_values("net_pnl", ascending=True).head(5)[show_cols or ["net_pnl"]], width="stretch", height=220)

                        st.divider()

                        # --- Cashflow story ---
                        try:
                            cf = pd.to_numeric(plot_df.get("cashflow"), errors="coerce").fillna(0.0)
                            mask = cf.abs() > 1e-9
                            if mask.any():
                                n = int(mask.sum())
                                total = float(cf[mask].sum())
                                avg = float(cf[mask].mean())
                                st.write("**Cashflows (deposits/withdrawals)**")
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Cashflow events", f"{n}")
                                c2.metric("Total cashflow", _fmt_money(total))
                                c3.metric("Avg event size", _fmt_money(avg))
                            else:
                                st.caption("No cashflows recorded in equity curve.")
                        except Exception:
                            pass

                # Stash replay exports for the Exports dropdown below
                replay_dl_items = [
                    (
                        "Download replay equity_curve.csv",
                        eq_path.read_bytes(),
                        f"{selected_run_name}_{pick}_equity_curve.csv",
                    )
                ]
                trades_path = art_dir / "trades.csv"
                fills_path = art_dir / "fills.csv"
                if trades_path.exists():
                    replay_dl_items.append(
                        (
                            "Download replay trades.csv",
                            trades_path.read_bytes(),
                            f"{selected_run_name}_{pick}_trades.csv",
                        )
                    )
                if fills_path.exists():
                    replay_dl_items.append(
                        (
                            "Download replay fills.csv",
                            fills_path.read_bytes(),
                            f"{selected_run_name}_{pick}_fills.csv",
                        )
                    )
            except Exception as e:
                st.warning(f"Could not load replay artifacts: {e}")
        else:
            st.caption("Replay artifacts not found yet for this config_id.")
    # ---------------------------------------------------------------------
    # Exports (advanced)
    # ---------------------------------------------------------------------
    with st.expander("Exports (advanced)", expanded=False):
        if replay_dl_items:
            st.write("**Replay exports**")
            for label, data, fname in replay_dl_items:
                st.download_button(label, data=data, file_name=fname)
            st.divider()

        st.write("**Batch exports**")
        st.download_button(
            "Download batch survivors (CSV)",
            data=df_show.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_run_name}_batch_view.csv",
        )

        show_raw = st.checkbox("Show raw table", value=False, key="ui.batch.show_raw")
        if show_raw:
                cols = [
                    "config_id",
                    "config.label",
                    "batch.verdict",
                    "equity.net_profit_ex_cashflows",
                    "performance.twr_total_return",
                    "performance.max_drawdown_equity",
                    "trades_summary.trades_closed",
                ]
                for c in ["score.calmar_equity", "score.profit_dd", "score.twr_dd", "score.profit"]:
                    if c in df_show.columns and c not in cols:
                        cols.append(c)
                cols = [c for c in cols if c in df_show.columns]
                st.dataframe(df_show[cols], width="stretch", height=520)

    st.divider()
    c_next, _ = st.columns([1, 4])
    with c_next:
        if st.button("Next: Rolling Starts →", type="primary"):
            st.session_state["ui.stage"] = "rs"
            st.rerun()
    st.caption("Next: run Rolling Starts to measure start-date fragility.")

# =============================================================================
# Stage B: Rolling Starts
# =============================================================================

if stage_pick == "rs":
    st.write("### B) Rolling Starts (start-date sensitivity)")
    with st.expander("How to read Rolling Starts", expanded=True):
        st.markdown(
            """
Rolling Starts reruns the **same** strategy many times — each run starts on a different date.
It answers the quant-flavored question: **is this edge real, or is it just “you started on the right day”?**

**How to read the summary numbers**
- **Windows**: number of start dates tested. More is better. **< 10 is noisy**.
- **Return p10 / p50 / p90**: pessimistic / typical / optimistic outcomes across start dates.
- **DD p90**: a “bad-but-plausible” max drawdown. Lower is better.
- **Underwater p90 (days)**: how long you’re likely to be stuck below your prior equity peak.
- **Fragility (spread p90 − p10)**: how wide outcomes swing across start dates. **Smaller = more stable.**

**How to read the charts**
- Each dot = one rolling-start window (one start date).
- Tight clusters are good. Wild scatter means **start-date luck** dominates.
- Dashed line = median (p50). Dotted lines = p10 / p90.

Rule of thumb: prefer strategies with a **decent p10** (survives bad starts), not just a spicy p50.
            """
        )

    st.caption("Runs the same strategy many times with different starting days, to measure fragility.")

    # RS selection / settings
    left, right = st.columns([2, 1])

    with left:
        rs_runs = []
        if rs_root.exists():
            rs_runs = [p for p in rs_root.glob("rs_*") if p.is_dir()]
            rs_runs = sorted(rs_runs, key=lambda p: p.stat().st_mtime, reverse=True)

        rs_choice = st.selectbox(
            "Rolling-start runs found",
            options=["(none)"] + [p.name for p in rs_runs],
            index=(1 if rs_runs else 0),
            key="rs.pick",
        )
        rs_dir = (rs_root / rs_choice) if (rs_choice != "(none)") else None

        with right:
            st.write("**Quick presets**")

            bars_per_day = _bars_per_day_from_run_meta(run_dir)
            bar_hint = _human_bar_interval_from_run(run_dir)
            st.caption(f"Detected timeframe: {bar_hint} (≈ {bars_per_day} bars/day)")

            preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="rs.preset")

            # Apply defaults only when preset changes (so number inputs don't reset constantly).
            prev = st.session_state.get("rs.preset_prev")
            if prev != preset:
                if bars_per_day <= 2:
                    # Daily-ish bars: space starts out in days, and require a long minimum history
                    if preset == "Quick":
                        step_days, min_days = 30, 365
                    elif preset == "Standard":
                        step_days, min_days = 14, 365
                    else:
                        step_days, min_days = 7, 365
                else:
                    # Intraday: still think in calendar days (convert to bars), but min history can be shorter
                    if preset == "Quick":
                        step_days, min_days = 7, 60
                    elif preset == "Standard":
                        step_days, min_days = 3, 90
                    else:
                        step_days, min_days = 1, 120

                st.session_state["rs.start_step"] = int(max(1, round(step_days * bars_per_day)))
                st.session_state["rs.min_bars"] = int(max(30, round(min_days * bars_per_day)))
                st.session_state["rs.preset_prev"] = preset

            start_step = int(
                st.number_input(
                    "Start step (bars)",
                    1,
                    500000,
                    int(st.session_state.get("rs.start_step", max(1, int(round(7 * bars_per_day))))),
                    5,
                    key="rs.start_step",
                )
            )
            min_bars = int(
                st.number_input(
                    "Min bars per start",
                    30,
                    5000000,
                    int(st.session_state.get("rs.min_bars", max(30, int(round(60 * bars_per_day))))),
                    30,
                    key="rs.min_bars",
                )
            )
            st.caption(f"Preset interpretation: start every ~{max(1, round(start_step / max(1e-9, bars_per_day))):.0f} days; require ~{max(1, round(min_bars / max(1e-9, bars_per_day))):.0f} days of data per start.")

            min_bars_effective = int(min_bars)  # placeholder for future clamping logic

    # Compute survivor ids (we stress test every survivor in full_passed if available, else sweep_passed)
    survivors_ids = survivors["config_id"].astype(str).tolist()
    ids_file = run_dir / "post" / "survivor_ids.txt"
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(survivors_ids) + "\n", encoding="utf-8")

    rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{len(survivors_ids)}"
    st.caption(f"Will run on survivors: {len(survivors_ids)} configs → output: {rs_out_dir}")

    can_run = True
    if len(survivors_ids) == 0:
        can_run = False
        st.error("No survivors IDs found.")
    if not (run_dir / "configs_resolved.jsonl").exists():
        can_run = False
        st.error("Missing configs_resolved.jsonl (needed for rolling starts).")

    if st.button("Run Rolling Starts for all survivors", type="primary", disabled=(not can_run)):
        try:
            cmd = [
                PY,
                "-m",
                "research.rolling_starts",
                "--from-run",
                str(run_dir),
                "--out",
                str(rs_out_dir),                "--top-n",
                str(len(survivors_ids)),
                "--start-step",
                str(start_step),
                "--min-bars",
                str(min_bars_effective),
                "--seed",
                "1",
                "--starting-equity",
                str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                "--jobs", "8",
            ]
            rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
            rs_progress.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--no-progress", "--progress-file", str(rs_progress), "--progress-every", "25"]
            _run_cmd(cmd, cwd=REPO_ROOT, label="Rolling Starts", progress_path=rs_progress)
            st.success("Rolling Starts complete.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
            st.stop()

    # Load chosen/latest summary
    rs_dir_effective = rs_dir or rs_out_dir
    rs_sum = load_rs_summary(run_dir, rs_dir_effective)
    rs_det = load_rs_detail(run_dir, rs_dir_effective)

    if rs_sum is None or rs_sum.empty:
        st.info("No rolling-start stats found yet for the chosen output folder.")
        st.stop()

    # Merge + evaluate
    base = survivors.copy()
    base = merge_stage(base, rs_sum, on="config_id", suffix="rs")

    cov = int(base["rs.measured"].sum()) if "rs.measured" in base.columns else 0
    st.success(f"Coverage: {cov}/{len(base)} configs have rolling-start stats in this folder.")

    with st.expander("Rolling-start questions (filters)", expanded=True):
        rs_ans = _question_ui(rolling_questions(), key_prefix="q.rs")

    dfB = apply_stage_eval(base, stage_key="rsq", questions=rolling_questions(), answers=rs_ans)

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.rs.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.rs.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.rs.fail")

    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")
    df_show = dfB[dfB["rsq.verdict"].isin(keep)].copy()

    
if stage_pick == "rs":
    # -------------------------------------------------------------------------
    # Rolling Starts: interpretability
    # -------------------------------------------------------------------------
    # Main table columns
    cols = [
        "config_id",
        "config.label",
        "rsq.verdict",
        "twr_p10",
        "twr_p50",
        "twr_p90",
        "dd_p50",
        "dd_p90",
        "uw_p50_days",
        "uw_p90_days",
        "util_p50",
        "util_p90",
        "robustness_score",
        "windows",
    ]
    cols = [c for c in cols if c in df_show.columns]
    if "rs.measured" in df_show.columns:
        cols.insert(2, "rs.measured")

    # Quick visual scan: stability map (return vs drawdown)
    st.subheader("Quick visual scan")
    df_plot = df_show.copy()
    for c in ["twr_p10", "twr_p50", "twr_p90", "dd_p50", "dd_p90", "robustness_score", "windows"]:
        if c in df_plot.columns:
            df_plot[c] = pd.to_numeric(df_plot[c], errors="coerce")


    # Derived measures (UI-only)
    if ("twr_p10" in df_plot.columns) and ("twr_p90" in df_plot.columns):
        df_plot["fragility_spread"] = pd.to_numeric(df_plot["twr_p90"], errors="coerce") - pd.to_numeric(df_plot["twr_p10"], errors="coerce")
    else:
        df_plot["fragility_spread"] = np.nan

    # (Micro-juice) Let users tighten the frontier so it doesn't get dominated by tiny sample sizes.
    min_windows = int(st.slider("Minimum rolling-start windows", 1, 200, 10, 1, key="rs.min_windows_ui"))
    # Filter by minimum windows (avoid df.get(...) returning an int when the column is missing)
    if "windows" in df_plot.columns:
        _win = pd.to_numeric(df_plot["windows"], errors="coerce").fillna(0).astype(int)
        df_plot = df_plot[_win >= min_windows].copy()
    else:
        st.info("No rolling-start 'windows' column found yet (run Rolling Starts first).")
        df_plot = df_plot.iloc[0:0].copy()



    # Cohort summary + fragility labels (relative to the current table)
    if not df_plot.empty:
        _sp = pd.to_numeric(df_plot.get("fragility_spread"), errors="coerce")
        if _sp.notna().any():
            q33 = float(_sp.quantile(0.33))
            q66 = float(_sp.quantile(0.66))
        else:
            q33, q66 = float("nan"), float("nan")

        def _frag_label(v: Any) -> str:
            try:
                x = float(v)
                if not math.isfinite(x):
                    return "—"
                if math.isfinite(q33) and x <= q33:
                    return "Low"
                if math.isfinite(q66) and x <= q66:
                    return "Medium"
                return "High"
            except Exception:
                return "—"

        df_plot["fragility"] = df_plot.get("fragility_spread").apply(_frag_label) if "fragility_spread" in df_plot.columns else "—"

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("Strategies (≥ min windows)", f"{len(df_plot)}")
        if "twr_p10" in df_plot.columns:
            cc2.metric("Median return p10", _fmt_pct(pd.to_numeric(df_plot["twr_p10"], errors="coerce").median()))
        if "dd_p90" in df_plot.columns:
            cc3.metric("Median DD p90", _fmt_pct(pd.to_numeric(df_plot["dd_p90"], errors="coerce").median()))
        if "fragility" in df_plot.columns:
            high_share = (df_plot["fragility"] == "High").mean() if len(df_plot) else 0.0
            cc4.metric("High fragility share", f"{high_share*100:.0f}%")

        st.caption("Each dot = one strategy. Up/right is better (higher median return, lower DD p90). Bigger dots = more rolling-start windows.")

        color_mode = st.selectbox("Color dots by", ["Verdict", "Fragility"], index=0, key="rs.scan.color")
        if color_mode == "Fragility" and "fragility" in df_plot.columns:
            color_col = "fragility"
        else:
            color_col = ("rsq.verdict" if "rsq.verdict" in df_plot.columns else None)
    else:
        color_col = None
    if df_plot.empty or ("dd_p90" not in df_plot.columns) or ("twr_p50" not in df_plot.columns):
        st.info("Not enough Rolling Starts data to plot yet.")
    else:
        fig = px.scatter(
            df_plot,
            x="dd_p90",
            y="twr_p50",
            color=color_col,
            size=("windows" if "windows" in df_plot.columns else None),
            hover_data=[c for c in ["config_id", "config.label", "rsq.verdict", "twr_p10", "twr_p50", "twr_p90", "dd_p90", "uw_p90_days", "windows", "fragility", "fragility_spread", "robustness_score"] if c in df_plot.columns],
            labels={
                "dd_p90": "Drawdown p90 (lower is better)",
                "twr_p50": "Return p50 (higher is better)",
            },
            title="Rolling Starts: return vs drawdown (stability map)",
        )

        # Baseline marker (user's original config)
        try:
            _bid = st.session_state.get("baseline_config_id")
            if _bid:
                bx = by = None
                filtered_out = False

                if "config_id" in df_plot.columns:
                    _b = df_plot[df_plot["config_id"].astype(str) == str(_bid)].head(1)
                    if not _b.empty:
                        bx = float(pd.to_numeric(_b["dd_p90"], errors="coerce").iloc[0])
                        by = float(pd.to_numeric(_b["twr_p50"], errors="coerce").iloc[0])

                # If the baseline exists in RS results but is filtered out of the chart view (e.g. min windows), still mark it.
                if (bx is None or by is None or (not math.isfinite(float(bx))) or (not math.isfinite(float(by)))) and ("config_id" in df_show.columns):
                    _b2 = df_show[df_show["config_id"].astype(str) == str(_bid)].head(1)
                    if not _b2.empty:
                        bx = float(pd.to_numeric(_b2.get("dd_p90"), errors="coerce").iloc[0])
                        by = float(pd.to_numeric(_b2.get("twr_p50"), errors="coerce").iloc[0])
                        filtered_out = True

                if bx is not None and by is not None and math.isfinite(float(bx)) and math.isfinite(float(by)):
                    fig.add_trace(
                        go.Scatter(
                            x=[float(bx)],
                            y=[float(by)],
                            mode="markers",
                            name="Your strategy",
                            marker=dict(
                                size=18,
                                symbol="star",
                                color="rgba(17,24,39,0.95)",
                                line=dict(width=2, color="rgba(255,255,255,0.95)"),
                            ),
                            opacity=0.7 if filtered_out else 1.0,
                            hovertemplate=(
                                "Your strategy (baseline)"
                                + ("<br><i>filtered out of this view</i>" if filtered_out else "")
                                + "<br>config={}<br>dd_p90={:.2%}<br>ret_p50={:.2%}<extra></extra>"
                            ).format(str(_bid), float(bx), float(by)),
                            showlegend=True,
                        )
                    )
        except Exception:
            pass

        _plotly(fig)

    st.subheader("Inspect a strategy (Rolling Starts)")
    if df_show.empty:
        st.info("No Rolling Starts rows to inspect.")
    else:
        # Pick a config to inspect (default: best robustness_score)
        df_rank = df_show.copy()
        if "robustness_score" in df_rank.columns:
            df_rank["robustness_score"] = pd.to_numeric(df_rank["robustness_score"], errors="coerce")
            df_rank = df_rank.sort_values("robustness_score", ascending=False)
        inspect_opts = df_rank["config_id"].astype(str).tolist()
        inspect_id = st.selectbox("Choose a config_id", options=inspect_opts, index=0, key="rs.inspect_id")

        row = df_show[df_show["config_id"].astype(str) == str(inspect_id)].head(1)
        if row.empty:
            st.warning("Could not load that config_id from the Rolling Starts view.")
        else:
            r0 = row.iloc[0]

            def _fmt_pct(x: float) -> str:
                try:
                    if pd.isna(x):
                        return "—"
                    return f"{float(x)*100:.2f}%"
                except Exception:
                    return "—"

            def _fmt_num(x: float) -> str:
                try:
                    if pd.isna(x):
                        return "—"
                    return f"{float(x):.3f}"
                except Exception:
                    return "—"

            # Mini story cards
            c1, c2, c3, c4, c5 = st.columns(5)
            if "windows" in row.columns:
                c1.metric("Windows", f"{int(pd.to_numeric(r0.get('windows'), errors='coerce') or 0)}")
            else:
                c1.metric("Windows", "—")
            c2.metric("Return p10", _fmt_pct(r0.get("twr_p10")))
            c3.metric("Return p50", _fmt_pct(r0.get("twr_p50")))
            c4.metric("DD p90", _fmt_pct(r0.get("dd_p90")))
            if "robustness_score" in row.columns:
                c5.metric("Stability score", _fmt_num(r0.get("robustness_score")))
            else:
                c5.metric("Stability score", "—")

            # Fragility: a fast, plain-English read (start-date sensitivity)
            frag = None
            try:
                frag = float(r0.get("twr_p90")) - float(r0.get("twr_p10"))
            except Exception:
                frag = None

            # Label fragility relative to the current cohort (so it adapts to different datasets)
            label = "—"
            try:
                cohort = df_show.copy()
                if ("twr_p10" in cohort.columns) and ("twr_p90" in cohort.columns):
                    cohort["_spread"] = pd.to_numeric(cohort["twr_p90"], errors="coerce") - pd.to_numeric(cohort["twr_p10"], errors="coerce")
                    cohort["_spread"] = cohort["_spread"].replace([np.inf, -np.inf], np.nan)
                    cohort_sp = cohort["_spread"].dropna()
                    if len(cohort_sp) >= 8 and frag is not None and math.isfinite(float(frag)):
                        q33 = float(cohort_sp.quantile(0.33))
                        q66 = float(cohort_sp.quantile(0.66))
                        if float(frag) <= q33:
                            label = "Low"
                        elif float(frag) <= q66:
                            label = "Medium"
                        else:
                            label = "High"
            except Exception:
                label = "—"

            # Plain-English reason (what hurts, how often)
            uw_p90 = r0.get("uw_p90_days") if "uw_p90_days" in row.columns else None
            win_n = None
            try:
                win_n = int(pd.to_numeric(r0.get("windows"), errors="coerce") or 0)
            except Exception:
                win_n = None

            parts = []
            if frag is not None and not pd.isna(frag):
                parts.append(f"spread(p90−p10) {_fmt_pct(frag)}")
            if "twr_p10" in row.columns:
                parts.append(f"p10 {_fmt_pct(r0.get('twr_p10'))}")
            if "dd_p90" in row.columns:
                parts.append(f"DD p90 {_fmt_pct(r0.get('dd_p90'))}")
            if uw_p90 is not None and not pd.isna(uw_p90):
                parts.append(f"UW p90 {int(float(uw_p90))}d")
            if win_n is not None:
                parts.append(f"{win_n} windows")

            msg = " · ".join(parts) if parts else "Not enough data."

            if label == "Low":
                st.success(f"Fragility: **Low** — {msg}")
            elif label == "Medium":
                st.warning(f"Fragility: **Medium** — {msg}")
            elif label == "High":
                st.error(f"Fragility: **High** — {msg}")
            else:
                st.info(f"Fragility: **—** — {msg}")

            # Detail plots (per-start windows)
            if rs_det is None or rs_det.empty:
                st.info("rolling_starts_detail.csv not found for this run yet. (You’ll still have the summary above.)")
            else:
                g = rs_det[rs_det["config_id"].astype(str) == str(inspect_id)].copy()
                if g.empty:
                    st.info("No per-start detail rows found for this config_id.")
                else:
                    # Parse & sort start date
                    if "start_dt" in g.columns:
                        g["start_dt"] = pd.to_datetime(g["start_dt"], errors="coerce")
                    if "start_i" in g.columns:
                        g["start_i"] = pd.to_numeric(g["start_i"], errors="coerce")
                    g = g.sort_values(["start_dt" if "start_dt" in g.columns else "start_i"])

                    # Normalize numeric fields
                    for c in ["performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days", "util_mean", "equity.net_profit_ex_cashflows"]:
                        if c in g.columns:
                            g[c] = pd.to_numeric(g[c], errors="coerce")

                    
                    # Quick pain points (what was the worst start date, and why?)
                    pains = []
                    try:
                        if "performance.twr_total_return" in g.columns:
                            rmin = g.dropna(subset=["performance.twr_total_return"]).nsmallest(1, "performance.twr_total_return").head(1)
                            if not rmin.empty:
                                rr = rmin.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Worst return start: **{s}** → return {_fmt_pct(rr.get('performance.twr_total_return'))}, DD {_fmt_pct(rr.get('performance.max_drawdown_equity'))}, UW {int(0 if pd.isna(pd.to_numeric(rr.get('uw_max_days'), errors='coerce')) else pd.to_numeric(rr.get('uw_max_days'), errors='coerce'))}d")
                        if "performance.max_drawdown_equity" in g.columns:
                            dmax = g.dropna(subset=["performance.max_drawdown_equity"]).nlargest(1, "performance.max_drawdown_equity").head(1)
                            if not dmax.empty:
                                rr = dmax.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Worst drawdown start: **{s}** → DD {_fmt_pct(rr.get('performance.max_drawdown_equity'))}, return {_fmt_pct(rr.get('performance.twr_total_return'))}")
                        if "uw_max_days" in g.columns:
                            umax = g.dropna(subset=["uw_max_days"]).nlargest(1, "uw_max_days").head(1)
                            if not umax.empty:
                                rr = umax.iloc[0]
                                s = rr.get("start_dt", rr.get("start_i", "—"))
                                pains.append(f"Longest underwater start: **{s}** → UW {int(0 if pd.isna(pd.to_numeric(rr.get('uw_max_days'), errors='coerce')) else pd.to_numeric(rr.get('uw_max_days'), errors='coerce'))}d, return {_fmt_pct(rr.get('performance.twr_total_return'))}")
                    except Exception:
                        pains = []

                    if pains:
                        st.markdown("**Worst-case highlights**  \\n" + "  \\n".join([f"- {p}" for p in pains]))

                    tabs = st.tabs(["Return vs start", "Drawdown vs start", "Underwater vs start", "Distributions", "Starts table"])

                    with tabs[0]:
                        if "performance.twr_total_return" not in g.columns:
                            st.info("Missing performance.twr_total_return in detail.")
                        else:
                            # Friendlier units for charts
                            if "performance.twr_total_return" in g.columns:
                                g["twr_pct"] = g["performance.twr_total_return"] * 100.0
                            if "performance.max_drawdown_equity" in g.columns:
                                g["dd_pct"] = g["performance.max_drawdown_equity"] * 100.0
                            fig_r = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="twr_pct",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.max_drawdown_equity", "uw_max_days"] if c in g.columns],
                                labels={"twr_pct": "Total return (%)"},
                                title="Rolling Starts: return by start date",
                            )
                            # Add p10/p50/p90 reference lines from the summary row
                            for qname, dash in [("twr_p10", "dot"), ("twr_p50", "dash"), ("twr_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_r.add_hline(y=float(r0.get(qname)) * 100.0, line_dash=dash)
                            st.caption("Dotted lines = p10/p90. Dashed line = median (p50). The less these dots care about where you start, the more 'real' the edge is.")
                            _plotly(fig_r)

                            # Worst / best start dates quick peek
                            g_rank = g.dropna(subset=["performance.twr_total_return"]).copy()
                            if not g_rank.empty:
                                worst = g_rank.nsmallest(5, "performance.twr_total_return")
                                best = g_rank.nlargest(5, "performance.twr_total_return")
                                cc1, cc2 = st.columns(2)
                                with cc1:
                                    st.write("**Worst starts (by return)**")
                                    st.dataframe(
                                        worst[[c for c in ["start_dt", "start_i", "performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days"] if c in worst.columns]],
                                        width="stretch",
                                        height=210,
                                    )
                                with cc2:
                                    st.write("**Best starts (by return)**")
                                    st.dataframe(
                                        best[[c for c in ["start_dt", "start_i", "performance.twr_total_return", "performance.max_drawdown_equity", "uw_max_days"] if c in best.columns]],
                                        width="stretch",
                                        height=210,
                                    )

                    with tabs[1]:
                        if "performance.max_drawdown_equity" not in g.columns:
                            st.info("Missing performance.max_drawdown_equity in detail.")
                        else:
                            fig_d = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="dd_pct",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.twr_total_return"] if c in g.columns],
                                labels={"dd_pct": "Max drawdown (%)"},
                                title="Rolling Starts: max drawdown by start date",
                            )
                            _plotly(fig_d)

                    with tabs[2]:
                        if "uw_max_days" not in g.columns:
                            st.info("Missing uw_max_days in detail.")
                        else:
                            fig_u = px.scatter(
                                g,
                                x=("start_dt" if "start_dt" in g.columns else "start_i"),
                                y="uw_max_days",
                                hover_data=[c for c in ["start_dt", "start_i", "bars", "performance.twr_total_return", "performance.max_drawdown_equity"] if c in g.columns],
                                labels={"uw_max_days": "Max underwater days"},
                                title="Rolling Starts: max underwater days by start date",
                            )
                            _plotly(fig_u)

                    
                    with tabs[3]:
                        # Distribution views help you see "how often does it hurt?"
                        cols_dist = st.columns(3)

                        if "performance.twr_total_return" in g.columns:
                            g["_twr_pct"] = pd.to_numeric(g["performance.twr_total_return"], errors="coerce") * 100.0
                            fig_hd = px.histogram(g.dropna(subset=["_twr_pct"]), x="_twr_pct", nbins=30, title="Return distribution (rolling starts)")
                            for qname, dash in [("twr_p10", "dot"), ("twr_p50", "dash"), ("twr_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_hd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            with cols_dist[0]:
                                _plotly(fig_hd)
                        else:
                            cols_dist[0].info("No return column in detail.")

                        if "performance.max_drawdown_equity" in g.columns:
                            g["_dd_pct"] = pd.to_numeric(g["performance.max_drawdown_equity"], errors="coerce") * 100.0
                            fig_dd = px.histogram(g.dropna(subset=["_dd_pct"]), x="_dd_pct", nbins=30, title="Drawdown distribution (rolling starts)")
                            for qname, dash in [("dd_p50", "dash"), ("dd_p90", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_dd.add_vline(x=float(r0.get(qname)) * 100.0, line_dash=dash)
                            with cols_dist[1]:
                                _plotly(fig_dd)
                        else:
                            cols_dist[1].info("No drawdown column in detail.")

                        if "uw_max_days" in g.columns:
                            fig_uw = px.histogram(g.dropna(subset=["uw_max_days"]), x="uw_max_days", nbins=30, title="Underwater days distribution (rolling starts)")
                            for qname, dash in [("uw_p50_days", "dash"), ("uw_p90_days", "dot")]:
                                if qname in row.columns and not pd.isna(r0.get(qname)):
                                    fig_uw.add_vline(x=float(r0.get(qname)), line_dash=dash)
                            with cols_dist[2]:
                                _plotly(fig_uw)
                        else:
                            cols_dist[2].info("No underwater column in detail.")

                        st.caption("Dashed line = median. Dotted lines = p10/p90 (or p90 for drawdown/underwater). Tight distributions are what 'robust' looks like.")
                    with tabs[4]:
                        st.dataframe(g, width="stretch", height=420)

                    with st.expander("Exports (advanced)", expanded=False):
                        st.download_button(
                            "Download rolling-start view (CSV)",
                            data=df_show.to_csv(index=False).encode("utf-8"),
                            file_name=f"{selected_run_name}_rolling_view.csv",
                        )
                        st.download_button(
                            "Download rolling-start detail for this config (CSV)",
                            data=g.to_csv(index=False).encode("utf-8"),
                            file_name=f"{selected_run_name}_rs_detail_{inspect_id}.csv",
                        )

        with st.expander("Rolling Starts table", expanded=False):
            st.dataframe(df_show[cols], width="stretch", height=520)

        # Next step: Walkforward
        if st.button("Next: Walkforward →", type="primary", key="rs.next_to_wf"):
            st.session_state["ui.stage"] = "wf"
            st.rerun()

# Stage C: Walkforward
# =============================================================================

if stage_pick == "wf":
    st.write("### C) Walkforward (generalization)")
    st.caption("Splits the history into rolling windows and measures how performance behaves out-of-sample-ish.")

    # Walkforward availability
    wf_module_ok = True
    try:
        __import__("engine.walkforward")
    except Exception:
        wf_module_ok = False

    if not wf_module_ok:
        st.warning("Walkforward module not found/importable yet (engine.walkforward). UI wiring is ready though.")
        st.stop()


    # Choose WF run dir + parameters
    left, right = st.columns([2, 1])
    run_clicked = False

    with left:
        wf_runs = []
        if wf_root.exists():
            wf_runs = [p for p in wf_root.glob("wf_*") if p.is_dir()]
            wf_runs = sorted(wf_runs, key=lambda p: p.stat().st_mtime, reverse=True)

        wf_choice = st.selectbox(
            "Walkforward runs found",
            options=["(none)"] + [p.name for p in wf_runs],
            index=(1 if wf_runs else 0),
            key="wf.pick",
        )
        wf_dir = (wf_root / wf_choice) if (wf_choice != "(none)") else None

    # Build WF command in the sidebar-ish column, but RUN it full-width (below columns)
    cmd: Optional[List[str]] = None
    wf_progress: Optional[Path] = None

    with right:
        st.write("**Quick presets**")

        bars_per_day = _bars_per_day_from_run_meta(run_dir)
        bar_hint = _human_bar_interval_from_run(run_dir)
        st.caption(f"Detected timeframe: {bar_hint} (≈ {bars_per_day} bars/day)")

        preset = st.selectbox("Preset", options=["Quick", "Standard", "Thorough"], index=0, key="wf.preset")

        # Apply defaults only when preset changes (so number inputs don't reset constantly).
        prev = st.session_state.get("wf.preset_prev")
        if prev != preset:
            if bars_per_day <= 2:
                # Daily-ish: longer windows make sense
                if preset == "Quick":
                    w_default, s_default, cov = 180, 30, 0.90
                elif preset == "Standard":
                    w_default, s_default, cov = 365, 30, 0.90
                else:
                    w_default, s_default, cov = 730, 30, 0.90
            else:
                # Intraday: shorter calendar windows still contain many bars
                if preset == "Quick":
                    w_default, s_default, cov = 30, 7, 0.95
                elif preset == "Standard":
                    w_default, s_default, cov = 60, 7, 0.95
                else:
                    w_default, s_default, cov = 90, 3, 0.95

            expected = int(max(1, round(w_default * bars_per_day)))
            mb_default = int(max(1, math.ceil(expected * cov)))

            st.session_state["wf.window_days"] = int(w_default)
            st.session_state["wf.step_days"] = int(s_default)
            st.session_state["wf.min_bars"] = int(mb_default)
            st.session_state["wf.jobs"] = int(st.session_state.get("wf.jobs", 8))
            st.session_state["wf.preset_prev"] = preset

        # Avoid Streamlit warning: don't set both a widget default and session_state.
        if "wf.window_days" not in st.session_state:
            st.session_state["wf.window_days"] = int(365)
        if "wf.step_days" not in st.session_state:
            st.session_state["wf.step_days"] = int(30)

        window_days = int(st.number_input("Window days", min_value=7, max_value=3650, step=5, key="wf.window_days"))
        step_days = int(st.number_input("Step days", min_value=1, max_value=3650, step=5, key="wf.step_days"))

        expected_window_bars = int(max(1, round(window_days * bars_per_day)))
        st.caption(f"Expected bars per window: ~{expected_window_bars:,}. (Min bars must be ≤ this.)")

        max_mb = int(max(1, expected_window_bars))
        if "wf.min_bars" not in st.session_state:
            st.session_state["wf.min_bars"] = int(max_mb)
        # Clamp current value to widget bounds to avoid Streamlit exceptions.
        if int(st.session_state.get("wf.min_bars", 1)) > int(max_mb):
            st.session_state["wf.min_bars"] = int(max_mb)

        min_bars = int(st.number_input(
            "Min bars per window",
            min_value=1,
            max_value=max_mb,
            step=1,
            key="wf.min_bars",
        ))

        if "wf.jobs" not in st.session_state:
            st.session_state["wf.jobs"] = int(max(1, min(8, (os.cpu_count() or 4))))
        jobs = int(st.session_state.get("wf.jobs", 8))

        survivors_ids = survivors["config_id"].astype(str).tolist()
        N = len(survivors_ids)

        # Clamp min_bars to something feasible for the chosen window
        expected_window_bars = int(max(1, round(window_days * bars_per_day))) if "bars_per_day" in locals() else int(window_days)
        min_bars_effective = int(min(int(min_bars), int(expected_window_bars)))
        if int(min_bars) != int(min_bars_effective):
            st.warning(f"Min bars ({min_bars}) exceeds expected bars/window (~{expected_window_bars}). Will clamp to {min_bars_effective}.")

        # WF output dir
        wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
        st.caption(f"Will run on survivors: {N} configs → output: {wf_out_dir}")

        run_clicked = st.button("Run Walkforward for all survivors", type="primary", disabled=(N == 0))

        if run_clicked:
            cmd = [
                PY,
                "-m",
                "engine.walkforward",
                "--from-run",
                str(run_dir),
                "--top-n",
                str(N),
                "--window-days",
                str(window_days),
                "--step-days",
                str(step_days),
                "--min-bars",
                str(min_bars_effective),
                "--jobs",
                str(jobs),
                "--out",
                str(wf_out_dir),
                "--sort-by",
                "gates.passed",  # stable, non-NaN, includes everyone selected by top-n
                "--sort-desc",
            ]
            wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
            wf_progress.parent.mkdir(parents=True, exist_ok=True)
            cmd += ["--no-progress", "--progress-file", str(wf_progress), "--progress-every", "25"]

    # Run full-width (NOT inside the right-side column) so the progress UI doesn't get squeezed.
    if run_clicked and cmd is not None and wf_progress is not None:
        st.markdown("---")
        try:
            _run_cmd(cmd, cwd=REPO_ROOT, label="Walkforward", progress_path=wf_progress)
            st.success("Walkforward complete.")
            st.rerun()
        except Exception as e:
            st.error(str(e))
            st.stop()

    wf_dir_effective = wf_dir or wf_out_dir
    wf_sum = load_wf_summary(wf_dir_effective)
    wf_rows = load_wf_results(wf_dir_effective)
    
    if wf_sum is None or wf_sum.empty:
        st.info("No walkforward stats found yet for the chosen output folder.")
        st.stop()
    
    base = merge_stage(survivors.copy(), wf_sum, on="config_id", suffix="wf")
    
    cov = int(base["wf.measured"].sum()) if "wf.measured" in base.columns else 0
    st.success(f"Coverage: {cov}/{len(base)} configs have walkforward stats in this folder.")
    
    with st.expander("Walkforward questions (filters)", expanded=True):
        wf_ans = _question_ui(walkforward_questions(), key_prefix="q.wf")
    
    dfC = apply_stage_eval(base, stage_key="wfq", questions=walkforward_questions(), answers=wf_ans)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pass = st.checkbox("Show PASS", value=True, key="f.wf.pass")
    with col2:
        show_warn = st.checkbox("Show WARN", value=True, key="f.wf.warn")
    with col3:
        show_fail = st.checkbox("Show FAIL", value=False, key="f.wf.fail")
    
    keep = []
    if show_pass:
        keep.append("PASS")
    if show_warn:
        keep.append("WARN")
    if show_fail:
        keep.append("FAIL")
    df_show = dfC[dfC["wfq.verdict"].isin(keep)].copy()
    
    
    cols = [
        "config_id",
        "config.label",
        "wfq.verdict",
        "return_p10",
        "return_p50",
        "return_p90",
        "dd_p90",
        "uw_days_p90",
        "pct_profitable_windows",
        "pct_windows_traded",
        "trades_p10",
        "trades_p50",
        "min_window_return",
        "median_window_return",
        "stitched_total_return",
        "stitched_max_drawdown",
        "windows",
    ]
    cols = [c for c in cols if c in df_show.columns]
    if "wf.measured" in df_show.columns:
        cols.insert(2, "wf.measured")
    st.dataframe(df_show[cols], width="stretch", height=520)

    st.download_button(
        "Download walkforward view (CSV)",
        data=df_show.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_run_name}_walkforward_view.csv",
    )

    st.markdown("---")
    st.write("### Inspect a strategy (Walkforward)")

    with st.expander("How to read Walkforward (what these charts mean)", expanded=True):
        st.write("Walkforward chops history into many rolling windows (episodes). Each dot in the charts is one episode.")
        st.write("You're looking for consistency across episodes — not one lucky stretch.")
        st.write("• Return over time: each dot is total return inside one window. Tight clusters beat wild scatter.")
        st.write("• Drawdown over time: each dot is the worst peak→trough drop inside that window. Spikes mean occasional pain.")
        st.write("• Underwater days: how long equity stayed below its prior peak inside the window (recovery time).")
        st.write("• Histogram: p10/p50/p90 are worst-typical / typical / best-typical window outcomes.")
        st.write("• Stitched curve: compounds non-overlapping step slices to avoid overlap. It's a stability visualization, not a promise of tradability.")


    if wf_rows is None or wf_rows.empty:
        st.info("No per-window walkforward rows found yet (wf_results.csv).")
    else:
        # Pick a config to inspect (default: highest typical return)
        if "return_p50" in df_show.columns:
            opts = (
                df_show.sort_values("return_p50", ascending=False)["config_id"].astype(str).tolist()
            )
        else:
            opts = df_show["config_id"].astype(str).tolist()

        if not opts:
            st.info("No configs in the current filter set.")
        else:
            pick_id = st.selectbox("Config", options=opts, index=0, key="wf.inspect.pick")

            wsub = wf_rows[wf_rows["config_id"].astype(str) == str(pick_id)].copy()
            wsub = wsub.sort_values("window_idx", kind="mergesort")

            sum_row = None
            try:
                ssub = wf_sum[wf_sum["config_id"].astype(str) == str(pick_id)]
                if ssub is not None and not ssub.empty:
                    sum_row = ssub.iloc[0].to_dict()
            except Exception:
                sum_row = None

            # Summary metrics
            if sum_row:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("WF p50 return", _fmt_pct(sum_row.get("return_p50")))
                with m2:
                    st.metric("WF p10 return", _fmt_pct(sum_row.get("return_p10")))
                with m3:
                    st.metric("WF dd_p90", _fmt_pct(sum_row.get("dd_p90")))
                with m4:
                    st.metric("% profitable windows", _fmt_pct(sum_row.get("pct_profitable_windows")))

            # Per-window timeline
            if "flags" in wsub.columns:
                wsub["has_flags"] = wsub["flags"].astype(str).str.len() > 0
            else:
                wsub["has_flags"] = False

            if "window_start_dt" in wsub.columns:
                x = "window_start_dt"
            else:
                x = "window_idx"

            yret = "window_return" if "window_return" in wsub.columns else "equity.total_return"
            fig = px.scatter(
                wsub,
                x=x,
                y=yret,
                color="has_flags",
                hover_data=[c for c in ["window_end_dt", "window_max_drawdown", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns],
                title="Walkforward windows: return over time",
            )
            # Quantile guides: teach the user to think in distributions (bad/typical/good windows).
            try:
                _vals = pd.to_numeric(wsub[yret], errors="coerce").dropna()
                if not _vals.empty:
                    r10 = float(_vals.quantile(0.10))
                    r50 = float(_vals.quantile(0.50))
                    r90 = float(_vals.quantile(0.90))
                    fig.add_hline(y=r50, line_dash="dash", annotation_text=f"p50 {r50:.1%}", annotation_position="top left")
                    fig.add_hline(y=r10, line_dash="dot", annotation_text=f"p10 {r10:.1%}", annotation_position="bottom left")
                    fig.add_hline(y=r90, line_dash="dot", annotation_text=f"p90 {r90:.1%}", annotation_position="top left")
            except Exception:
                pass
            fig.update_yaxes(tickformat=".0%")
            _plotly(fig)
            st.caption("Each dot is one window. Tight clusters beat lucky spikes. p10/p50/p90 lines show bad/typical/good window outcomes.")

            if "window_max_drawdown" in wsub.columns:
                fig2 = px.scatter(
                    wsub,
                    x=x,
                    y="window_max_drawdown",
                    color="has_flags",
                    hover_data=[c for c in ["window_return", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns],
                    title="Walkforward windows: max drawdown over time",
                )
                try:
                    d90 = float(pd.to_numeric(wsub["window_max_drawdown"], errors="coerce").dropna().quantile(0.90))
                    fig2.add_hline(y=d90, line_dash="dash", annotation_text=f"dd_p90 {d90:.1%}", annotation_position="top left")
                except Exception:
                    pass
                fig2.update_yaxes(tickformat=".0%")
                _plotly(fig2)
                st.caption("Each dot is the worst peak→trough drop inside that window. dd_p90 is your 'bad but typical' drawdown anchor.")

            if "window_underwater_days" in wsub.columns:
                fig_uw = px.scatter(
                    wsub,
                    x=x,
                    y="window_underwater_days",
                    color="has_flags",
                    hover_data=[c for c in ["window_return", "window_max_drawdown", "trades_closed", "flags"] if c in wsub.columns],
                    title="Walkforward windows: underwater days over time",
                )
                try:
                    uw90 = float(pd.to_numeric(wsub["window_underwater_days"], errors="coerce").dropna().quantile(0.90))
                    fig_uw.add_hline(y=uw90, line_dash="dash", annotation_text=f"uw_p90 {uw90:.0f}d", annotation_position="top left")
                except Exception:
                    pass
                _plotly(fig_uw)
                st.caption("Underwater days = time spent below the previous equity peak inside the window. High values mean long recovery / long boredom.")

            # Distribution (sanity check)
            if "window_return" in wsub.columns:
                fig3 = px.histogram(wsub, x="window_return", nbins=30, title="Window return distribution")
                try:
                    _vals = pd.to_numeric(wsub["window_return"], errors="coerce").dropna()
                    if not _vals.empty:
                        r10 = float(_vals.quantile(0.10))
                        r50 = float(_vals.quantile(0.50))
                        r90 = float(_vals.quantile(0.90))
                        fig3.add_vline(x=r50, line_dash="dash", annotation_text=f"p50 {r50:.1%}", annotation_position="top")
                        fig3.add_vline(x=r10, line_dash="dot", annotation_text=f"p10 {r10:.1%}", annotation_position="top")
                        fig3.add_vline(x=r90, line_dash="dot", annotation_text=f"p90 {r90:.1%}", annotation_position="top")
                except Exception:
                    pass
                fig3.update_xaxes(tickformat=".0%")
                _plotly(fig3)
                st.caption("Histogram of window returns. p10 is the 'worst-typical' anchor; p50 is typical; p90 is best-typical.")

            # Window leaderboard (failure modes)
            if "window_return" in wsub.columns:
                st.write("**Window leaderboard (failure modes)**")
                show_cols = [c for c in ["window_idx", "window_start_dt", "window_end_dt", "window_return", "window_max_drawdown", "window_underwater_days", "trades_closed", "flags"] if c in wsub.columns]

                t1, t2, t3 = st.tabs(["Worst return", "Worst drawdown", "Longest underwater"])
                with t1:
                    st.dataframe(
                        wsub.sort_values("window_return", ascending=True)[show_cols].head(10),
                        width="stretch",
                        height=260,
                    )
                with t2:
                    if "window_max_drawdown" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_max_drawdown", ascending=False)[show_cols].head(10),
                            width="stretch",
                            height=260,
                        )
                    else:
                        st.info("No drawdown column for this walkforward run.")
                with t3:
                    if "window_underwater_days" in wsub.columns:
                        st.dataframe(
                            wsub.sort_values("window_underwater_days", ascending=False)[show_cols].head(10),
                            width="stretch",
                            height=260,
                        )
                    else:
                        st.info("No underwater-days column for this walkforward run.")

            # Stitched curve (non-overlapping segments)
            stitched_path = None
            try:
                if sum_row and sum_row.get("stitched_path"):
                    stitched_path = wf_dir_effective / str(sum_row["stitched_path"])
                else:
                    stitched_path = wf_dir_effective / "stitched" / f"{pick_id}.csv"
            except Exception:
                stitched_path = wf_dir_effective / "stitched" / f"{pick_id}.csv"

            if stitched_path is not None and stitched_path.exists():
                st.write("**Stitched curve (non-overlapping segments)**")
                st.caption("This compounds step-sized slices to avoid overlap. It's a stability visualization, not a promise of tradability.")
                sdf = _load_csv(stitched_path)
                if sdf is not None and not sdf.empty and "stitched_twr" in sdf.columns:
                    fig4 = px.line(sdf, x="dt" if "dt" in sdf.columns else sdf.columns[0], y="stitched_twr", title="Stitched TWR index")
                    try:
                        fig4.add_hline(y=1.0, line_dash="dash", annotation_text="start (1.0)", annotation_position="bottom right")
                    except Exception:
                        pass
                    _plotly(fig4)

                    st.download_button(
                        "Download stitched curve (CSV)",
                        data=sdf.to_csv(index=False).encode("utf-8"),
                        file_name=f"{selected_run_name}_wf_stitched_{pick_id}.csv",
                    )
            else:
                st.info("No stitched curve found for this config (expected under wf_dir/stitched/).")

            # Downloads
            st.download_button(
                "Download per-window rows for this config (CSV)",
                data=wsub.to_csv(index=False).encode("utf-8"),
                file_name=f"{selected_run_name}_wf_windows_{pick_id}.csv",
            )

    
    # =============================================================================

# =============================================================================
# Stage D: Grand verdict + deep dive
# =============================================================================

# --- Grand Verdict: scoring lens (profile-adjusted ranking) --------------------
# Evidence (raw metrics) never changes. A "lens" only changes how we *rank* and
# how strict our defaults are for the limit radios.

DEFAULT_GRAND_LENS: Dict[str, float] = {
    # Component weights (must sum to 1.0 after normalization)
    "wf_w": 0.60,
    "rs_w": 0.30,
    "bt_w": 0.10,
    # Pain penalty coefficients
    "dd_k": 0.50,      # applied to WF/RS drawdown (dd_p90)
    "uw_k": 0.10,      # applied to underwater years (uw_days/365)
    "bt_dd_k": 0.50,   # applied to Batch max drawdown
    # Missing-evidence penalties (keeps unmeasured configs from floating to the top)
    "wf_missing_pen": 0.25,
    "rs_missing_pen": 0.10,
}

# "Casual → Quant" ladder: pick what you want your life to feel like.
# Each profile sets (A) default limit radios and (B) a scoring lens for ranking.
GRAND_PROFILE_SPECS: Dict[str, Dict[str, Any]] = {
    # ---- Core product profiles ----
    "steady_compounding": {
        "label": "Steady compounding (recommended)",
        "desc": "Balanced robustness. Wants decent worst-case returns and avoids nasty drawdowns without being overly strict.",
        "limits": {
            "batch": {"batch_drawdown": 1, "batch_profit": 0, "batch_fees": 1},
            "rs": {"rs_worst_return": 1, "rs_drawdown": 1, "rs_underwater": 2, "rs_util": 0},
            "wf": {"wf_typical": 0, "wf_worst_typical": 1, "wf_min": 1, "wf_dd": 1, "wf_consistency": 1, "wf_trading": 1},
        },
        "lens": {},
    },
    "low_drawdown": {
        "label": "Low drawdown (sleep-at-night)",
        "desc": "Strict pain limits. Prefers plans that don't go deep underwater and don't take huge hits.",
        "limits": {
            "batch": {"batch_drawdown": 0, "batch_profit": 0, "batch_fees": 0},
            "rs": {"rs_worst_return": 1, "rs_drawdown": 0, "rs_underwater": 1, "rs_util": 1},
            "wf": {"wf_typical": 0, "wf_worst_typical": 0, "wf_min": 0, "wf_dd": 0, "wf_consistency": 0, "wf_trading": 0},
        },
        "lens": {"wf_w": 0.70, "rs_w": 0.25, "bt_w": 0.05, "dd_k": 0.70, "uw_k": 0.15, "bt_dd_k": 0.70},
    },
    "bear_survivor": {
        "label": "Survives bear markets",
        "desc": "Prioritizes walk-forward robustness. Accepts longer recovery, but punishes drawdowns and weak windows.",
        "limits": {
            "batch": {"batch_drawdown": 1, "batch_profit": 0, "batch_fees": 1},
            "rs": {"rs_worst_return": 1, "rs_drawdown": 1, "rs_underwater": 3, "rs_util": 2},
            "wf": {"wf_typical": 0, "wf_worst_typical": 0, "wf_min": 1, "wf_dd": 1, "wf_consistency": 1, "wf_trading": 1},
        },
        "lens": {"wf_w": 0.75, "rs_w": 0.20, "bt_w": 0.05, "dd_k": 0.60, "uw_k": 0.05, "bt_dd_k": 0.60},
    },
    "vol_tolerant": {
        "label": "I can handle volatility (aggressive)",
        "desc": "Looser pain limits. Lets more strategies through and ranks more on upside/exploration.",
        "limits": {
            "batch": {"batch_drawdown": 2, "batch_profit": 0, "batch_fees": 2},
            "rs": {"rs_worst_return": 2, "rs_drawdown": 2, "rs_underwater": 3, "rs_util": 3},
            "wf": {"wf_typical": 0, "wf_worst_typical": 2, "wf_min": 2, "wf_dd": 2, "wf_consistency": 2, "wf_trading": 2},
        },
        "lens": {"wf_w": 0.50, "rs_w": 0.25, "bt_w": 0.25, "dd_k": 0.35, "uw_k": 0.05, "bt_dd_k": 0.35},
    },

    # ---- Legacy "tightness" presets (kept for familiarity; lens = default) ----
    "conservative": {
        "label": "Conservative (tight filters)",
        "desc": "Strict limits, but uses the default ranking lens.",
        "limits": {
            "batch": {"batch_drawdown": 0, "batch_profit": 0, "batch_fees": 0},
            "rs": {"rs_worst_return": 0, "rs_drawdown": 0, "rs_underwater": 0, "rs_util": 0},
            "wf": {"wf_typical": 0, "wf_worst_typical": 0, "wf_min": 0, "wf_dd": 0, "wf_consistency": 0, "wf_trading": 0},
        },
        "lens": {},
    },
    "balanced": {
        "label": "Balanced (default filters)",
        "desc": "Reasonable limits for exploration without being naive. Uses the default ranking lens.",
        "limits": {
            "batch": {"batch_drawdown": 1, "batch_profit": 0, "batch_fees": 1},
            "rs": {"rs_worst_return": 1, "rs_drawdown": 1, "rs_underwater": 2, "rs_util": 1},
            "wf": {"wf_typical": 0, "wf_worst_typical": 1, "wf_min": 1, "wf_dd": 1, "wf_consistency": 1, "wf_trading": 1},
        },
        "lens": {},
    },
    "aggressive": {
        "label": "Explorer (loose filters)",
        "desc": "Loose limits to see the landscape. Uses the default ranking lens (you can still sort by other metrics).",
        "limits": {
            "batch": {"batch_drawdown": 2, "batch_profit": 0, "batch_fees": 2},
            "rs": {"rs_worst_return": 2, "rs_drawdown": 2, "rs_underwater": 3, "rs_util": 2},
            "wf": {"wf_typical": 0, "wf_worst_typical": 2, "wf_min": 2, "wf_dd": 2, "wf_consistency": 2, "wf_trading": 2},
        },
        "lens": {},
    },
}

def _normalize_lens(lens: Dict[str, Any]) -> Dict[str, float]:
    """Return a float-only, weight-normalized lens dict."""
    out: Dict[str, float] = dict(DEFAULT_GRAND_LENS)
    if isinstance(lens, dict):
        for k, v in lens.items():
            try:
                out[k] = float(v)
            except Exception:
                pass

    # Normalize weights to sum to 1 (fall back to defaults if broken).
    wf_w = float(out.get("wf_w", DEFAULT_GRAND_LENS["wf_w"]))
    rs_w = float(out.get("rs_w", DEFAULT_GRAND_LENS["rs_w"]))
    bt_w = float(out.get("bt_w", DEFAULT_GRAND_LENS["bt_w"]))
    s = wf_w + rs_w + bt_w
    if not math.isfinite(s) or s <= 0:
        wf_w, rs_w, bt_w = DEFAULT_GRAND_LENS["wf_w"], DEFAULT_GRAND_LENS["rs_w"], DEFAULT_GRAND_LENS["bt_w"]
        s = wf_w + rs_w + bt_w
    out["wf_w"] = wf_w / s
    out["rs_w"] = rs_w / s
    out["bt_w"] = bt_w / s

    return out

def _get_active_grand_lens(*, use_profile_lens: bool = True) -> Dict[str, float]:
    """Active lens = default or profile-adjusted, depending on UI toggle."""
    if not use_profile_lens:
        return dict(DEFAULT_GRAND_LENS)
    return _normalize_lens(st.session_state.get("grand.lens_v1") or {})


def _apply_grand_preset(preset: str) -> None:
    """Apply a Grand Verdict profile preset.

    A profile sets two things:
      1) Default answers for the limit radios (PASS/WARN/FAIL behavior)
      2) A scoring lens used to rank candidates (profile-adjusted Stability)

    It never reruns compute or changes raw evidence.
    """
    raw = (preset or "").strip()
    if not raw:
        return

    s = raw.strip().lower()

    # Allow passing either the internal key or the display label.
    key = None
    if s in GRAND_PROFILE_SPECS:
        key = s
    else:
        # Match by label substring (robust to small UI label edits).
        for k, spec in GRAND_PROFILE_SPECS.items():
            try:
                lab = str(spec.get("label", "")).strip().lower()
            except Exception:
                lab = ""
            if lab and lab == s:
                key = k
                break
        if key is None:
            for k, spec in GRAND_PROFILE_SPECS.items():
                try:
                    lab = str(spec.get("label", "")).strip().lower()
                except Exception:
                    lab = ""
                if lab and (s in lab or lab in s):
                    key = k
                    break

    if key is None:
        # Handle legacy names
        if "steady" in s:
            key = "steady_compounding"
        elif "draw" in s or "sleep" in s:
            key = "low_drawdown"
        elif "bear" in s:
            key = "bear_survivor"
        elif "vol" in s or "aggress" in s:
            key = "vol_tolerant"
        elif "conserv" in s:
            key = "conservative"
        elif "balanc" in s:
            key = "balanced"
        elif "explor" in s:
            key = "aggressive"

    if key is None or key not in GRAND_PROFILE_SPECS:
        return

    spec = GRAND_PROFILE_SPECS.get(key) or {}
    limits = spec.get("limits") or {}
    lens_over = spec.get("lens") or {}

    # Apply default limit radio indexes.
    try:
        for qid, idx in (limits.get("batch") or {}).items():
            st.session_state[f"q.grand.batch.{qid}"] = int(idx)
        for qid, idx in (limits.get("rs") or {}).items():
            st.session_state[f"q.grand.rs.{qid}"] = int(idx)
        for qid, idx in (limits.get("wf") or {}).items():
            st.session_state[f"q.grand.wf.{qid}"] = int(idx)
    except Exception:
        pass

    # Apply scoring lens (store as overrides; normalization happens at use time).
    try:
        merged = dict(DEFAULT_GRAND_LENS)
        if isinstance(lens_over, dict):
            merged.update({k: float(v) for k, v in lens_over.items() if k in DEFAULT_GRAND_LENS})
        st.session_state["grand.lens_v1"] = merged
        st.session_state["grand.lens_source"] = str(spec.get("label") or key)
    except Exception:
        pass


def _grand_score_row(r: Dict[str, Any], *, lens: Optional[Dict[str, Any]] = None) -> float:
    """Profile-adjusted 'Stability' score used for ranking.

    Evidence is constant; this only changes how we combine it (weights/penalties).
    Defaults match the original score:
      Score = 0.60*WF + 0.30*RS + 0.10*Batch − missing_penalties
    """
    r = r or {}
    L = _normalize_lens(lens or {})

    def _nan0(x: float) -> float:
        return 0.0 if (x != x) else float(x)

    # Walkforward
    wf_r10 = _to_float(r.get("return_p10", float("nan")))
    if wf_r10 != wf_r10:
        wf_r10 = _to_float(r.get("return_p10.wf", float("nan")))
    wf_dd90 = _to_float(r.get("dd_p90.wf", float("nan"))) if ("dd_p90.wf" in r) else _to_float(r.get("dd_p90", float("nan")))
    wf_uw90 = _to_float(r.get("uw_days_p90.wf", float("nan"))) if ("uw_days_p90.wf" in r) else _to_float(r.get("uw_days_p90", float("nan")))

    # Rolling Starts
    rs_r10 = _to_float(r.get("twr_p10", float("nan")))
    if rs_r10 != rs_r10:
        rs_r10 = _to_float(r.get("twr_p10.rs", float("nan")))
    rs_dd90 = _to_float(r.get("dd_p90", float("nan")))  # RS drawdown is dd_p90 pre-WF merge
    rs_uw90 = _to_float(r.get("uw_p90_days", float("nan")))
    if rs_uw90 != rs_uw90:
        rs_uw90 = _to_float(r.get("uw_days_p90", float("nan")))

    # Batch
    b_r = _to_float(r.get("performance.twr_total_return", float("nan")))
    b_dd = _to_float(r.get("performance.max_drawdown_equity", float("nan")))

    dd_k = float(L.get("dd_k", DEFAULT_GRAND_LENS["dd_k"]))
    uw_k = float(L.get("uw_k", DEFAULT_GRAND_LENS["uw_k"]))
    bt_dd_k = float(L.get("bt_dd_k", DEFAULT_GRAND_LENS["bt_dd_k"]))

    wf = _nan0(wf_r10) - dd_k * _nan0(wf_dd90) - uw_k * (_nan0(wf_uw90) / 365.0)
    rs = _nan0(rs_r10) - dd_k * _nan0(rs_dd90) - uw_k * (_nan0(rs_uw90) / 365.0)
    bt = _nan0(b_r) - bt_dd_k * _nan0(b_dd)

    wf_pen = float(L.get("wf_missing_pen", DEFAULT_GRAND_LENS["wf_missing_pen"])) if (wf_r10 != wf_r10 or wf_dd90 != wf_dd90) else 0.0
    rs_pen = float(L.get("rs_missing_pen", DEFAULT_GRAND_LENS["rs_missing_pen"])) if (rs_r10 != rs_r10 or rs_dd90 != rs_dd90) else 0.0

    return float(L["wf_w"]) * wf + float(L["rs_w"]) * rs + float(L["bt_w"]) * bt - wf_pen - rs_pen



def _stability_breakdown(row: Dict[str, Any], *, lens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a breakdown dict for the profile-adjusted Stability score.

    NOTE: Raw evidence is constant. This only varies weights/penalty coefficients.
    """
    row = row or {}
    L = _normalize_lens(lens or {})

    def _pick(cands: List[str]) -> Tuple[float, Optional[str]]:
        for c in cands:
            if c in row:
                v = _to_float(row.get(c, float("nan")))
                return v, c
        return float("nan"), None

    def _is_nan(x: float) -> bool:
        return (x != x)

    def _nan0(x: float) -> float:
        return 0.0 if _is_nan(x) else float(x)

    dd_k = float(L.get("dd_k", DEFAULT_GRAND_LENS["dd_k"]))
    uw_k = float(L.get("uw_k", DEFAULT_GRAND_LENS["uw_k"]))
    bt_dd_k = float(L.get("bt_dd_k", DEFAULT_GRAND_LENS["bt_dd_k"]))

    # ---- Walkforward (WF) ----
    wf_ret, wf_ret_col = _pick(["return_p10.wf", "return_p10", "wf.return_p10"])
    wf_dd, wf_dd_col = _pick(["dd_p90.wf", "dd_p90", "wf.dd_p90"])
    wf_uw, wf_uw_col = _pick(["uw_days_p90.wf", "uw_days_p90", "wf.uw_days_p90", "uw_p90_days.wf", "uw_p90_days"])

    wf_raw = _nan0(wf_ret) - dd_k * _nan0(wf_dd) - uw_k * (_nan0(wf_uw) / 365.0)
    wf_w = float(L["wf_w"])
    wf_weighted = wf_w * wf_raw

    # ---- Rolling Starts (RS) ----
    rs_ret, rs_ret_col = _pick(["twr_p10.rs", "twr_p10", "rs.twr_p10", "return_p10.rs"])
    rs_dd, rs_dd_col = _pick(["dd_p90.rs", "dd_p90", "rs.dd_p90"])
    rs_uw, rs_uw_col = _pick(["uw_days_p90.rs", "uw_days_p90", "rs.uw_days_p90", "uw_p90_days.rs", "uw_p90_days", "rs.uw_p90_days"])

    rs_raw = _nan0(rs_ret) - dd_k * _nan0(rs_dd) - uw_k * (_nan0(rs_uw) / 365.0)
    rs_w = float(L["rs_w"])
    rs_weighted = rs_w * rs_raw

    # ---- Batch (single run) ----
    bt_ret, bt_ret_col = _pick(["performance.twr_total_return", "twr_total_return", "performance.total_return"])
    bt_dd, bt_dd_col = _pick(["performance.max_drawdown_equity", "max_drawdown_equity", "performance.max_drawdown", "performance.max_dd"])

    bt_raw = _nan0(bt_ret) - bt_dd_k * _nan0(bt_dd)
    bt_w = float(L["bt_w"])
    bt_weighted = bt_w * bt_raw

    penalties: List[Dict[str, Any]] = []
    pen_total = 0.0

    wf_missing = []
    if _is_nan(wf_ret):
        wf_missing.append("return_p10")
    if _is_nan(wf_dd):
        wf_missing.append("dd_p90")
    if wf_missing:
        amt = float(L.get("wf_missing_pen", DEFAULT_GRAND_LENS["wf_missing_pen"]))
        penalties.append(
            {
                "component": "Walkforward",
                "amount": amt,
                "reason": "Missing required WF stats (return_p10 and/or dd_p90).",
                "missing": wf_missing,
            }
        )
        pen_total += amt

    rs_missing = []
    if _is_nan(rs_ret):
        rs_missing.append("twr_p10")
    if _is_nan(rs_dd):
        rs_missing.append("dd_p90")
    if rs_missing:
        amt = float(L.get("rs_missing_pen", DEFAULT_GRAND_LENS["rs_missing_pen"]))
        penalties.append(
            {
                "component": "Rolling Starts",
                "amount": amt,
                "reason": "Missing required RS stats (twr_p10 and/or dd_p90).",
                "missing": rs_missing,
            }
        )
        pen_total += amt

    total = wf_weighted + rs_weighted + bt_weighted - pen_total

    def _fmt(x: float) -> str:
        try:
            if not math.isfinite(float(x)):
                return str(x)
            # Keep short but readable
            return f"{float(x):.2g}"
        except Exception:
            return str(x)

    return {
        "lens": {
            "wf_w": wf_w,
            "rs_w": rs_w,
            "bt_w": bt_w,
            "dd_k": dd_k,
            "uw_k": uw_k,
            "bt_dd_k": bt_dd_k,
        },
        "wf": {
            "inputs": {
                "return_p10": {"col": wf_ret_col, "value": wf_ret},
                "dd_p90": {"col": wf_dd_col, "value": wf_dd},
                "uw_days_p90": {"col": wf_uw_col, "value": wf_uw},
            },
            "formula_str": f"wf = r_p10 − {_fmt(dd_k)}*dd_p90 − {_fmt(uw_k)}*(uw_days/365)",
            "raw": wf_raw,
            "weight": wf_w,
            "weighted": wf_weighted,
            "missing": wf_missing,
        },
        "rs": {
            "inputs": {
                "twr_p10": {"col": rs_ret_col, "value": rs_ret},
                "dd_p90": {"col": rs_dd_col, "value": rs_dd},
                "uw_days_p90": {"col": rs_uw_col, "value": rs_uw},
            },
            "formula_str": f"rs = twr_p10 − {_fmt(dd_k)}*dd_p90 − {_fmt(uw_k)}*(uw_days/365)",
            "raw": rs_raw,
            "weight": rs_w,
            "weighted": rs_weighted,
            "missing": rs_missing,
        },
        "bt": {
            "inputs": {
                "total_return": {"col": bt_ret_col, "value": bt_ret},
                "max_drawdown": {"col": bt_dd_col, "value": bt_dd},
            },
            "formula_str": f"bt = ret − {_fmt(bt_dd_k)}*dd",
            "raw": bt_raw,
            "weight": bt_w,
            "weighted": bt_weighted,
            "missing": [],
        },
        "penalties": penalties,
        "penalty_total": pen_total,
        "total": total,
    }


if stage_pick == "grand":
    # -------------------------------------------------------------------------
    # Cockpit view (MVP): preferences → shortlist → evidence
    # -------------------------------------------------------------------------
    st.subheader("Results & Autopsy")
    st.caption("Set your lens → review the run overview → build a shortlist → open Evidence to inspect receipts.")

    # -------------------------------------------------------------------------
    # Run overview (preferences first, then population charts)
    # -------------------------------------------------------------------------
    overview_slot = st.container()
    with overview_slot:
        st.subheader("Run overview")
        # Run / dataset context (kept here so we only have one summary area)
        _run_name = str(getattr(run_dir, "name", "—"))
        _ds_path = None
        try:
            _ds = (manifest.get('dataset') or {}) if isinstance(manifest, dict) else {}
            _ds_path = _ds.get('path_abs') or _ds.get('path')
        except Exception:
            _ds_path = None
        
        _meta_bits = [f"Run: **{_run_name}**"]
        if _ds_path:
            _meta_bits.append(f"Dataset: `{_ds_path}`")
        if 'rs_label' in globals():
            _meta_bits.append(f"Rolling Starts: **{rs_label}**")
        if 'wf_label' in globals():
            _meta_bits.append(f"Walkforward: **{wf_label}**")
        st.caption(" · ".join(_meta_bits))

        st.caption("Set your lens first. These preferences change how the overview charts filter, color, and explain the run.")
        prefs_slot = st.container()
        charts_slot = st.container()

    # Load latest RS/WF if present
    rs_dir_effective = rs_latest
    wf_dir_effective = wf_latest

    rs_sum = load_rs_summary(run_dir, rs_dir_effective) if rs_dir_effective else None
    wf_sum = load_wf_summary(wf_dir_effective) if wf_dir_effective else None

    df = survivors.copy()
    df = _ensure_config_id(df)

    # =========================
    # Preferences wedge
    # =========================
    with prefs_slot:
        with st.expander("Lens & filters", expanded=True):
            st.caption("Pick your profile and limits. This only filters/labels/ranks candidates — it does not rerun compute.")

            preset = st.selectbox(
                "Goal profile",
                options=[
                    GRAND_PROFILE_SPECS["steady_compounding"]["label"],
                    GRAND_PROFILE_SPECS["low_drawdown"]["label"],
                    GRAND_PROFILE_SPECS["bear_survivor"]["label"],
                    GRAND_PROFILE_SPECS["vol_tolerant"]["label"],
                    GRAND_PROFILE_SPECS["aggressive"]["label"],
                    "Custom",
                ],
                index=0,
                key="grand.profile_preset_v3",
                help="Pick the outcome you want. Presets set default pain limits AND a ranking lens. 'Custom' keeps your current choices.",
            )

            # Explain the selected profile briefly (no math).
            _spec = None
            try:
                _spec = next((sp for sp in GRAND_PROFILE_SPECS.values() if str(sp.get("label", "")) == str(preset)), None)
            except Exception:
                _spec = None
            if _spec and _spec.get("desc"):
                st.caption(str(_spec.get("desc")))

            use_profile_lens = st.checkbox(
                "Use profile lens for ranking",
                value=bool(st.session_state.get("grand.use_profile_lens", True)),
                key="grand.use_profile_lens",
                help="ON = ranking uses the profile-adjusted Stability lens. OFF = uses the system/default Stability.",
            )
            _active_lens = _get_active_grand_lens(use_profile_lens=bool(use_profile_lens))
            st.caption(
                f"Ranking lens → WF {int(round(100*_active_lens['wf_w']))}% · RS {int(round(100*_active_lens['rs_w']))}% · Batch {int(round(100*_active_lens['bt_w']))}%"
                f" | penalties: dd×{_active_lens['dd_k']:.2g}, uw×{_active_lens['uw_k']:.2g}"
            )

            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                if st.button("Apply preset", key="grand.apply_preset_btn_v2", disabled=(preset == "Custom")):
                    # Capture current answers → apply preset → compute diff (so users can see what changed)
                    bqs = batch_questions()
                    rqs = rolling_questions()
                    wqs = walkforward_questions()

                    def _get_idx(prefix: str, q) -> int:
                        key = f"{prefix}.{q.id}"
                        try:
                            return int(st.session_state.get(key, int(q.default_index)))
                        except Exception:
                            return int(getattr(q, "default_index", 0) or 0)

                    def _choice_label(q, idx: int) -> str:
                        try:
                            idx2 = max(0, min(int(idx), len(q.choices) - 1))
                            return str(q.choices[idx2].label)
                        except Exception:
                            return str(idx)

                    before = {}
                    for q in bqs:
                        before[("Batch", q.id)] = _get_idx("q.grand.batch", q)
                    for q in rqs:
                        before[("Rolling Starts", q.id)] = _get_idx("q.grand.rs", q)
                    for q in wqs:
                        before[("Walkforward", q.id)] = _get_idx("q.grand.wf", q)

                    _apply_grand_preset(str(preset))

                    changes = []
                    for q in bqs:
                        a = _get_idx("q.grand.batch", q)
                        b = int(before.get(("Batch", q.id), a))
                        if a != b:
                            changes.append(("Batch", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))
                    for q in rqs:
                        a = _get_idx("q.grand.rs", q)
                        b = int(before.get(("Rolling Starts", q.id), a))
                        if a != b:
                            changes.append(("Rolling Starts", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))
                    for q in wqs:
                        a = _get_idx("q.grand.wf", q)
                        b = int(before.get(("Walkforward", q.id), a))
                        if a != b:
                            changes.append(("Walkforward", str(getattr(q, "title", getattr(q, "id", ""))), _choice_label(q, b), _choice_label(q, a)))

                    st.session_state["grand.last_preset_applied"] = str(preset)
                    st.session_state["grand.last_preset_changes"] = list(changes)
                    # Force a clean rerun so the radios below render with the new preset values
                    st.rerun()
            with c2:
                show_help = st.checkbox("Show explainer", value=False, key="grand.show_help")
            with c3:
                st.caption("Presets only change filters + ranking lens. They never modify your data or rerun anything.")


            last_preset = st.session_state.get("grand.last_preset_applied")
            last_changes = st.session_state.get("grand.last_preset_changes")
            if last_preset and isinstance(last_changes, list):
                if len(last_changes) == 0:
                    st.info(f"Preset **{last_preset}** matched your current limits (no changes).")
                else:
                    st.success(f"Applied preset **{last_preset}** → updated {len(last_changes)} limit choices.")
                    with st.expander("Show what the preset changed", expanded=True):
                        # Keep it readable: show the first N changes.
                        for section, q_label, old_label, new_label in last_changes[:40]:
                            st.write(f"**{section}** · {q_label}: `{old_label}` → `{new_label}`")
                        if len(last_changes) > 40:
                            st.caption(f"(Showing 40 of {len(last_changes)} changes.)")

            if show_help:
                st.markdown('''
        - **PASS / WARN / FAIL** are driven by the limit radios below.  
        - **PASS** = within limits. **WARN** = suspicious but maybe acceptable. **FAIL** = exceeds hard limits.  
        - If Rolling Starts / Walkforward are **missing**, either ignore them (early exploration) or require them (trust mode).
        '''.strip())

            # Limit radios (collapsed by default; the preset sets defaults)
            with st.expander("Batch limits", expanded=True):
                batch_ans = _question_ui(batch_questions(), key_prefix="q.grand.batch")
            df = apply_stage_eval(df, stage_key="batch", questions=batch_questions(), answers=batch_ans)

            rs_ans: Dict[str, int] = {}
            if rs_sum is not None and not rs_sum.empty:
                df = merge_stage(df, rs_sum, on="config_id", suffix="rs")
                with st.expander("Rolling Starts limits", expanded=True):
                    rs_ans = _question_ui(rolling_questions(), key_prefix="q.grand.rs")
                df = apply_stage_eval(df, stage_key="rsq", questions=rolling_questions(), answers=rs_ans)
            else:
                df["rs.measured"] = False
                df["rsq.verdict"] = "UNMEASURED"

            wf_ans: Dict[str, int] = {}
            if wf_sum is not None and not wf_sum.empty:
                df = merge_stage(df, wf_sum, on="config_id", suffix="wf")
                with st.expander("Walkforward limits", expanded=True):
                    wf_ans = _question_ui(walkforward_questions(), key_prefix="q.grand.wf")
                df = apply_stage_eval(df, stage_key="wfq", questions=walkforward_questions(), answers=wf_ans)
            else:
                df["wf.measured"] = False
                df["wfq.verdict"] = "UNMEASURED"

            st.divider()

            col1, col2, col3 = st.columns(3)
            with col1:
                req_batch = st.selectbox("Require Batch", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_batch")
            with col2:
                req_rs = st.selectbox("Require Rolling Starts", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_rs")
            with col3:
                req_wf = st.selectbox("Require Walkforward", options=["PASS only", "PASS or WARN", "Ignore"], index=1, key="grand.req_wf")

            # Verdict visibility toggles (global)
            vc1, vc2, vc3 = st.columns(3)
            with vc1:
                show_pass = st.checkbox("Show PASS", value=True, key="grand.show_pass")
            with vc2:
                show_warn = st.checkbox("Show WARN", value=True, key="grand.show_warn")
            with vc3:
                show_fail = st.checkbox("Show FAIL/UNMEASURED", value=False, key="grand.show_fail")

            st.markdown("#### Ranking")
            st.caption("The score is a ranking hint. The evidence tabs are the receipts.")

        # =========================
    # Build the shortlist (unified candidates table)
    # =========================
    def _keep(verdict: str, rule: str) -> bool:
        if rule.startswith("Ignore"):
            return True
        if rule.startswith("PASS only"):
            return verdict == "PASS"
        return verdict in {"PASS", "WARN"}

    keep_mask: List[bool] = []
    grand_verdicts: List[str] = []

    for _, r in df.iterrows():
        ok = True
        stage_vs: List[str] = []

        v_batch = str(r.get("batch.verdict", ""))
        ok = ok and _keep(v_batch, req_batch)
        if not req_batch.startswith("Ignore"):
            stage_vs.append(v_batch)

        v_rs = str(r.get("rsq.verdict", "UNMEASURED"))
        if v_rs == "UNMEASURED":
            ok = ok and (req_rs == "Ignore")
            if not req_rs.startswith("Ignore"):
                stage_vs.append("UNMEASURED")
        else:
            ok = ok and _keep(v_rs, req_rs)
            if not req_rs.startswith("Ignore"):
                stage_vs.append(v_rs)

        v_wf = str(r.get("wfq.verdict", "UNMEASURED"))
        if v_wf == "UNMEASURED":
            ok = ok and (req_wf == "Ignore")
            if not req_wf.startswith("Ignore"):
                stage_vs.append("UNMEASURED")
        else:
            ok = ok and _keep(v_wf, req_wf)
            if not req_wf.startswith("Ignore"):
                stage_vs.append(v_wf)

        keep_mask.append(bool(ok))

        if "FAIL" in stage_vs or "UNMEASURED" in stage_vs:
            gv = "FAIL" if "UNMEASURED" not in stage_vs else "UNMEASURED"
        elif "WARN" in stage_vs:
            gv = "WARN"
        else:
            gv = "PASS"
        grand_verdicts.append(gv)

    df["grand.verdict"] = grand_verdicts
    df2 = df[pd.Series(keep_mask, index=df.index)].copy()

    if not df2.empty:
        _lens_rank = _get_active_grand_lens(use_profile_lens=bool(st.session_state.get("grand.use_profile_lens", True)))
        df2["score.grand_robust"] = [_grand_score_row(r, lens=_lens_rank) for r in df2.to_dict(orient="records")]
        df2["score.grand_robust"] = pd.to_numeric(df2["score.grand_robust"], errors="coerce")

        # Keep a system/default stability score for truth-anchoring/debugging.
        df2["score.grand_system"] = [_grand_score_row(r, lens=DEFAULT_GRAND_LENS) for r in df2.to_dict(orient="records")]
        df2["score.grand_system"] = pd.to_numeric(df2["score.grand_system"], errors="coerce")

    sort_opts: List[str] = []
    for c in [
        "score.grand_robust",
        "score.grand_system",
        "score.calmar_equity",
        "robustness_score",
        "return_p10",
        "return_p50",
        "dd_p90",
        "pct_profitable_windows",
        "pct_windows_traded",
        "twr_p10",
        "twr_p50",
        "performance.twr_total_return",
        "performance.max_drawdown_equity",
        "equity.net_profit_ex_cashflows",
    ]:
        if c in df2.columns and c not in sort_opts:
            sort_opts.append(c)
    if not sort_opts:
        sort_opts = ["config_id"]

    sort_by = st.selectbox("Sort by", options=sort_opts, index=0, key="grand.sort_by")
    ascending = st.checkbox("Ascending", value=False, key="grand.asc")
    if sort_by in df2.columns and not df2.empty:
        df2[sort_by] = pd.to_numeric(df2[sort_by], errors="coerce")
        df2 = df2.sort_values(sort_by, ascending=bool(ascending))

    mask_v: List[bool] = []
    for v in df2.get("grand.verdict", pd.Series([], dtype=str)).astype(str):
        if v == "PASS" and show_pass:
            mask_v.append(True)
        elif v == "WARN" and show_warn:
            mask_v.append(True)
        elif v in {"FAIL", "UNMEASURED"} and show_fail:
            mask_v.append(True)
        else:
            mask_v.append(False)

    df_show = df2[pd.Series(mask_v, index=df2.index)] if (len(mask_v) == len(df2)) else df2


    with charts_slot:

        # Summary strip
        n_all = int(len(df)) if isinstance(df, pd.DataFrame) else 0
        n_req = int(len(df2)) if isinstance(df2, pd.DataFrame) else 0
        n_vis = int(len(df_show)) if isinstance(df_show, pd.DataFrame) else 0
        reject_rate = (1.0 - (n_req / max(1, n_all))) if n_all else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Survivors evaluated", f"{n_all:,}")
        c2.metric("Meets requirements", f"{n_req:,}")
        c3.metric("Visible now", f"{n_vis:,}")
        c4.metric("Reject rate", _fmt_pct(reject_rate, digits=1) if n_all else "—")

        # Quick narrative (very light-touch, no hype)
        try:
            if "grand.verdict" in df2.columns and len(df2) > 0:
                _vc = df2["grand.verdict"].astype(str).value_counts()
                top = str(_vc.index[0]) if len(_vc) else ""
                st.caption(f"Most candidates are currently **{top}** under your lens. Tighten/loosen limits to see how the population shifts.")
        except Exception:
            pass

        if px is None or go is None:
            st.info("Plotly is not available in this environment, so overview charts are disabled.")
        else:
            df_all = df.copy()
            df_req = df2.copy()
            df_vis = df_show.copy()

            # Build pipeline snapshot (optional)
            with st.expander("Run pipeline snapshot (optional)", expanded=False):
                with st.expander("Advanced (chart options)", expanded=False):
                    pipe_show_counts = st.checkbox("Show verdict mix as counts (advanced)", value=False, key="pipe_show_counts")

                # Funnel counts
                n_eval = int(len(df_all))
                n_req = int(len(df_req))
                n_vis = int(len(df_vis))
                drop_req = max(0, n_eval - n_req)
                drop_vis = max(0, n_req - n_vis)

                funnel = pd.DataFrame({
                    "Stage": ["Survivors evaluated", "Meets requirements", "Visible now"],
                    "Count": [n_eval, n_req, n_vis],
                })
                funnel_text = [
                    "",
                    (f"−{drop_req:,} removed" if drop_req else "0 removed"),
                    (f"−{drop_vis:,} hidden" if drop_vis else "0 hidden"),
                ]
                fig_funnel = go.Figure(
                    go.Funnel(
                        y=funnel["Stage"],
                        x=funnel["Count"],
                        text=funnel_text,
                        textinfo="value+percent initial+text",
                        marker=dict(color=[ACCENT_BLUE, NEUTRAL_COLOR, PASS_COLOR]),
                    )
                )
                _style_fig(fig_funnel, title="Survivor funnel")
                notes = []
                if drop_req == 0:
                    notes.append("No candidates rejected by requirements.")
                else:
                    notes.append(f"{drop_req:,} rejected by requirements.")
                if drop_vis == 0:
                    notes.append("All remaining candidates are visible.")
                else:
                    notes.append(f"{drop_vis:,} hidden by visibility cutoff.")
                funnel_note = " • ".join(notes) if notes else ""

                # Verdict mix by stage
                rows = []
                stage_specs = [
                    ("Batch", "batch.verdict"),
                    ("Rolling Starts", "rsq.verdict"),
                    ("Walkforward", "wfq.verdict"),
                    ("Grand (overall)", "grand.verdict"),
                ]
                for stage_label, col in stage_specs:
                    if col in df_all.columns:
                        vc = df_all[col].fillna("UNMEASURED").astype(str).value_counts()
                        for v, cnt in vc.items():
                            rows.append({"Stage": stage_label, "Verdict": str(v), "Count": int(cnt)})
                df_stage = pd.DataFrame(rows)

                fig_stage = None
                if not df_stage.empty:
                    # Legend order + consistent colors
                    verdict_order = ["PASS", "WARN", "FAIL"]
                    extra = [v for v in df_stage["Verdict"].unique() if v not in verdict_order]
                    verdict_order = verdict_order + sorted(extra)
                
                    if pipe_show_counts:
                        # Raw counts
                        fig_stage = px.bar(
                            df_stage,
                            x="Stage",
                            y="Count",
                            color="Verdict",
                            barmode="stack",
                            category_orders={"Verdict": verdict_order, "Stage": [s[0] for s in stage_specs]},
                            color_discrete_map={k: _verdict_color(k) for k in verdict_order},
                        )
                        _style_fig(fig_stage, title="Verdict mix by stage")
                        fig_stage.update_layout(xaxis_title=None, yaxis_title=None)
                    else:
                        # Percent-stacked (default)
                        d = df_stage.copy()
                        d["StageTotal"] = d.groupby("Stage")["Count"].transform("sum")
                        d["SharePct"] = (100.0 * d["Count"] / d["StageTotal"]).fillna(0.0)
                        # label big segments only (keeps it clean)
                        d["Label"] = d.apply(lambda r: (f"{r['SharePct']:.0f}%" if r['SharePct'] >= 12 else ""), axis=1)
                        fig_stage = px.bar(
                            d,
                            x="Stage",
                            y="SharePct",
                            color="Verdict",
                            text="Label",
                            barmode="stack",
                            category_orders={"Verdict": verdict_order, "Stage": [s[0] for s in stage_specs]},
                            color_discrete_map={k: _verdict_color(k) for k in verdict_order},
                            custom_data=["Count", "StageTotal"],
                        )
                        _style_fig(fig_stage, title="Verdict mix by stage")
                        fig_stage.update_layout(xaxis_title=None, yaxis_title=None, yaxis=dict(range=[0, 100]))
                        fig_stage.update_traces(textposition="inside", insidetextanchor="middle")
                        fig_stage.update_traces(
                            hovertemplate="%{x}<br>%{fullData.name}: %{customdata[0]} / %{customdata[1]} (%{y:.1f}%)<extra></extra>"
                        )

                    # Place legend to the right so it doesn't collide with the title
                    fig_stage.update_layout(
                        title_x=0, title_xanchor="left",
                        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="right", x=1.0),
                        margin=dict(t=95),
                    )

                pc1, pc2 = st.columns([1.0, 1.2])
                with pc1:
                    _plotly(fig_funnel)
                    if funnel_note:
                        st.caption(funnel_note)
                with pc2:
                    if fig_stage is not None:
                        _plotly(fig_stage)
                    else:
                        st.caption("No verdict columns found to build stage distribution.")
            # Core chart: Risk/return map (population view)
            dd_col = _pick_col(df_all, ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown"])
            ret_col = _pick_col(df_all, ["performance.twr_total_return", "equity.net_profit_ex_cashflows", "equity.net_profit"])

            if dd_col and ret_col and dd_col in df_all.columns and ret_col in df_all.columns and go is not None:
                base = df_all.copy()
                base["_dd"] = _drawdown_to_frac(pd.to_numeric(base[dd_col], errors="coerce"))
                base["_ret"] = pd.to_numeric(base[ret_col], errors="coerce")
                base = base.dropna(subset=["_dd", "_ret"])

                if "config_id" in base.columns:
                    base["_cid"] = base["config_id"].astype(str)
                else:
                    base["_cid"] = ""


                hi = df_vis.copy()
                if dd_col in hi.columns and ret_col in hi.columns:
                    hi["_dd"] = _drawdown_to_frac(pd.to_numeric(hi[dd_col], errors="coerce"))
                    hi["_ret"] = pd.to_numeric(hi[ret_col], errors="coerce")
                    hi = hi.dropna(subset=["_dd", "_ret"])
                else:
                    hi = pd.DataFrame()

                score_col = None
                for c in ["score.grand_robust", "robustness_score"]:
                    if c in hi.columns:
                        score_col = c
                        break
                robust_pct = {}
                if score_col and "config_id" in hi.columns:
                    try:
                        _s = pd.to_numeric(hi[score_col], errors="coerce")
                        _r = _s.rank(pct=True, ascending=True)
                        robust_pct = {str(cid): float(p) for cid, p in zip(hi["config_id"].astype(str), _r)}
                    except Exception:
                        robust_pct = {}

                def _verdict_symbol(v: str) -> str:
                    vv = str(v or "").upper()
                    if vv == "PASS":
                        return "circle"
                    if vv == "WARN":
                        return "triangle-up"
                    if vv in {"FAIL", "UNMEASURED"}:
                        return "x"
                    return "diamond"

                def _reason_snippet(row_dict: Dict[str, Any]) -> str:
                    """One-line reason snippet (first/highest-severity violation under current prefs)."""
                    sev_rank = {"critical": 0, "warn": 1, "info": 2}
                    best = None
                    try:
                        out_b = evaluate_row_with_questions(row_dict, batch_questions(), batch_ans)
                        for v in getattr(out_b, "violations", []) or []:
                            r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                            msg = str(v.get("message", "")).strip()
                            if msg and (best is None or r < best[0]):
                                best = (r, "Batch", msg)
                    except Exception:
                        pass
                    try:
                        if rs_sum is not None and not rs_sum.empty:
                            out_r = evaluate_row_with_questions(row_dict, rolling_questions(), rs_ans)
                            for v in getattr(out_r, "violations", []) or []:
                                r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                                msg = str(v.get("message", "")).strip()
                                if msg and (best is None or r < best[0]):
                                    best = (r, "RS", msg)
                    except Exception:
                        pass
                    try:
                        if wf_sum is not None and not wf_sum.empty:
                            out_w = evaluate_row_with_questions(row_dict, walkforward_questions(), wf_ans)
                            for v in getattr(out_w, "violations", []) or []:
                                r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                                msg = str(v.get("message", "")).strip()
                                if msg and (best is None or r < best[0]):
                                    best = (r, "WF", msg)
                    except Exception:
                        pass
                    if best is None:
                        return ""
                    return f"{best[1]}: {best[2]}"

                is_currency = ("net_profit" in str(ret_col).lower()) or ("profit" in str(ret_col).lower() and "return" not in str(ret_col).lower())

                fig_rr = go.Figure()
                fig_rr.add_trace(
                    go.Scattergl(
                        x=base["_dd"],
                        y=base["_ret"],
                        mode="markers",
                        marker=dict(size=6, color="rgba(17,24,39,0.12)", symbol="circle"),
                        hoverinfo="skip",
                        customdata=base[["_cid"]].to_numpy() if "_cid" in base.columns else None,
                        showlegend=False,
                    )
                )

                hi2 = None
                if not hi.empty and "grand.verdict" in hi.columns:
                    hi2 = hi.copy()
                    hi2["_cid"] = hi2["config_id"].astype(str) if "config_id" in hi2.columns else ""
                    label_col = _pick_col(hi2, ["config.label", "label", "config_label"])
                    hi2["_label"] = hi2[label_col].astype(str) if label_col and label_col in hi2.columns else ""

                    _allowed = {"PASS", "WARN", "FAIL", "UNMEASURED"}
                    hi2["_grand_raw"] = hi2["grand.verdict"].astype(str)
                    hi2["_grand"] = hi2["_grand_raw"].str.upper()
                    hi2["_grand"] = hi2["_grand"].where(hi2["_grand"].isin(_allowed), "OTHER")

                    hi2["_rob"] = hi2["_cid"].map(robust_pct)
                    hi2["_rob_str"] = hi2["_rob"].apply(lambda p: f"{int(round(float(p)*100))}th pct" if pd.notna(p) else "—")

                    hi2["_reason"] = ""
                    try:
                        _n = int(min(250, len(hi2)))
                        _sub = hi2.head(_n)
                        hi2.loc[_sub.index, "_reason"] = [_reason_snippet(r) for r in _sub.to_dict("records")]
                    except Exception:
                        pass

                    order = ["PASS", "WARN", "FAIL", "UNMEASURED", "OTHER"]
                    for v in order:
                        g = hi2[hi2["_grand"].astype(str) == str(v)]
                        if g.empty:
                            continue
                        custom = np.stack(
                            [
                                g["_cid"].to_numpy(),
                                g["_label"].to_numpy(),
                                g["_grand"].to_numpy(),
                                g["_rob_str"].to_numpy(),
                                g["_reason"].to_numpy(),
                            ],
                            axis=1,
                        )
                        hover_ret = "%{y:.2%}" if not is_currency else "$%{y:,.0f}"
                        fig_rr.add_trace(
                            go.Scattergl(
                                x=g["_dd"],
                                y=g["_ret"],
                                mode="markers",
                                name=("Other" if str(v) == "OTHER" else str(v)),
                                showlegend=(str(v) != "OTHER"),
                                marker=dict(
                                    size=11,
                                    color=_verdict_color(v),
                                    symbol=_verdict_symbol(v),
                                    opacity=0.95,
                                    line=dict(width=0.8, color="rgba(17,24,39,0.35)"),
                                ),
                                customdata=custom,
                                hovertemplate=(
                                    "config=%{customdata[0]}<br>"
                                    "label=%{customdata[1]}<br>"
                                    "grand verdict=%{customdata[2]}<br>"
                                    "stability=%{customdata[3]}<br>"
                                    "max DD=%{x:.2%}<br>"
                                    f"return={hover_ret}<br>"
                                    "reason=%{customdata[4]}<extra></extra>"
                                ),
                            )
                        )

                st.markdown("#### Risk/return map (population view)")
                st.caption("Faint = all survivors · Bold = visible shortlist · Markers: PASS ○ · WARN ▲ · FAIL ✕")
                fig_rr.update_layout(title=None)
                _style_fig(fig_rr, title=None)
                fig_rr.update_layout(
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                        font=dict(size=12),
                    )
                )
                for tr in fig_rr.data:
                    try:
                        tr.legendgrouptitle = None
                    except Exception:
                        pass
                    try:
                        tr.legendgroup = None
                    except Exception:
                        pass

                fig_rr.update_layout(margin=dict(l=20, r=20, t=55, b=20))
                fig_rr.update_xaxes(title="Max drawdown (lower is better)", tickformat=".0%")
                if is_currency:
                    fig_rr.update_yaxes(title="Net profit (excluding deposits)", tickprefix="$", separatethousands=True)
                else:
                    fig_rr.update_yaxes(title="Total return (higher is better)", tickformat=".0%")

                try:
                    _d = fig_rr.to_dict()
                    if "layout" in _d and "legend" in _d["layout"] and isinstance(_d["layout"]["legend"], dict):
                        _d["layout"]["legend"]["title"] = {"text": "\u00A0", "font": {"color": "rgba(0,0,0,0)", "size": 1}}  # defined-but-invisible legend title avoids Plotly.js "undefined"
                    for _tr in _d.get("data", []):
                        if isinstance(_tr, dict):
                            _tr.pop("legendgrouptitle", None)
                            _tr.pop("legendgroup", None)
                    fig_rr = go.Figure(_d)
                except Exception:
                    pass


                # Baseline marker (user's original config)
                try:
                    _bid = st.session_state.get("baseline_config_id")
                    if _bid:
                        bx = by = None

                        # Prefer the plotted survivor population for coordinates
                        if "config_id" in df_all.columns:
                            _b0 = df_all[df_all["config_id"].astype(str) == str(_bid)].head(1)
                            if not _b0.empty and dd_col and ret_col and (dd_col in _b0.columns) and (ret_col in _b0.columns):
                                bx = _drawdown_to_frac(pd.to_numeric(_b0[dd_col], errors="coerce")).iloc[0]
                                by = pd.to_numeric(_b0[ret_col], errors="coerce").iloc[0]

                        # Fallback: baseline may be filtered out of survivors; pull from full/sweep tables.
                        if (bx is None or by is None or pd.isna(bx) or pd.isna(by)) and dd_col and ret_col:
                            # Try broader tables in priority order
                            df_full = None
                            for _k in ["full_all", "sweep_all", "sweep_passed"]:
                                _tmp = frames.get(_k)
                                if _tmp is not None and hasattr(_tmp, "empty") and (not _tmp.empty):
                                    df_full = _tmp
                                    break
                            if df_full is not None and (not df_full.empty) and ("config_id" in df_full.columns):
                                # Prefer the chart's chosen columns; otherwise pick best available from df_full.
                                dd_cands = ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown", "equity.max_dd", "max_drawdown", "max_dd", "dd"]
                                ret_cands = ["performance.twr_total_return", "performance.total_return", "performance.twr", "twr_total_return", "total_return",
                                            "equity.net_profit_ex_cashflows", "equity.net_profit", "net_profit_ex_cashflows", "net_profit", "profit"]
                                dd_use = dd_col if dd_col in df_full.columns else _pick_col(df_full, dd_cands)
                                ret_use = ret_col if ret_col in df_full.columns else _pick_col(df_full, ret_cands)

                                _b1 = df_full[df_full["config_id"].astype(str) == str(_bid)].head(1)
                                if not _b1.empty:
                                    # Drawdown x
                                    if dd_use and dd_use in _b1.columns:
                                        bx = _drawdown_to_frac(pd.to_numeric(_b1[dd_use], errors="coerce")).iloc[0]

                                    # Return y (keep axis semantics aligned to the chart as best we can)
                                    chart_is_pct = True
                                    try:
                                        chart_is_pct = ("twr" in str(ret_col).lower()) or ("return" in str(ret_col).lower())
                                    except Exception:
                                        chart_is_pct = True

                                    if chart_is_pct:
                                        # Prefer a return-like column if present; otherwise compute from profit / starting_equity.
                                        by = None
                                        if ret_use and ret_use in _b1.columns and (("twr" in str(ret_use).lower()) or ("return" in str(ret_use).lower())):
                                            by = pd.to_numeric(_b1[ret_use], errors="coerce").iloc[0]

                                        if by is None or pd.isna(by):
                                            # Profit fallback
                                            pcol = _pick_col(df_full, ["equity.net_profit_ex_cashflows", "equity.net_profit", "net_profit_ex_cashflows", "net_profit", "profit"])
                                            start_eq = None
                                            try:
                                                start_eq = float(meta.get("starting_equity") or 0)
                                            except Exception:
                                                start_eq = None
                                            if pcol and (pcol in _b1.columns) and start_eq and start_eq > 0:
                                                _p = pd.to_numeric(_b1[pcol], errors="coerce").iloc[0]
                                                if pd.notna(_p):
                                                    by = float(_p) / float(start_eq)
                                    else:
                                        # Currency-like axis; prefer profit columns.
                                        by = None
                                        if ret_use and ret_use in _b1.columns and (("twr" not in str(ret_use).lower()) and ("return" not in str(ret_use).lower())):
                                            by = pd.to_numeric(_b1[ret_use], errors="coerce").iloc[0]
                                        if by is None or pd.isna(by):
                                            pcol = _pick_col(df_full, ["equity.net_profit_ex_cashflows", "equity.net_profit", "net_profit_ex_cashflows", "net_profit", "profit"])
                                            if pcol and pcol in _b1.columns:
                                                by = pd.to_numeric(_b1[pcol], errors="coerce").iloc[0]

                        if pd.notna(bx) and pd.notna(by) and math.isfinite(float(bx)) and math.isfinite(float(by)):
                            hover_ret = "%{y:.2%}" if not is_currency else "$%{y:,.0f}"
                            fig_rr.add_trace(
                                go.Scatter(
                                    x=[float(bx)],
                                    y=[float(by)],
                                    mode="markers",
                                    name="Your strategy",
                                    marker=dict(
                                        size=18,
                                        symbol="star",
                                        color="rgba(17,24,39,0.95)",
                                        line=dict(width=2, color="rgba(255,255,255,0.95)"),
                                    ),
                                    customdata=[[str(_bid)]],
                                    hovertemplate=("Your strategy (baseline)<br>config=%{customdata[0]}<br>max DD=%{x:.2%}<br>return=" + hover_ret + "<extra></extra>"),
                                    showlegend=True,
                                )
                            )
                except Exception:
                    pass
                # Interactive selection (click/box-select) for inspection
                if "ui.rr_pick" not in st.session_state:
                    st.session_state["ui.rr_pick"] = None
                if "ui.rr_pick_x" not in st.session_state:
                    st.session_state["ui.rr_pick_x"] = None
                if "ui.rr_pick_y" not in st.session_state:
                    st.session_state["ui.rr_pick_y"] = None
                if "ui.rr_pick_cd" not in st.session_state:
                    st.session_state["ui.rr_pick_cd"] = None

                try:
                    fig_rr.update_layout(clickmode="event+select")
                except Exception:
                    pass

                _rr_event = None
                try:
                    _kwargs: Dict[str, Any] = {"config": PLOTLY_CONFIG, "key": "risk_return.map", "on_select": "rerun"}
                    if _PLOTLY_HAS_WIDTH:
                        _kwargs["width"] = "stretch"
                    elif _PLOTLY_HAS_UCW:
                        _kwargs["use_container_width"] = True
                    _kwargs.setdefault("theme", "streamlit" if USE_STREAMLIT_PLOTLY_THEME else None)
                    _rr_event = st.plotly_chart(fig_rr, **_kwargs)
                except TypeError:
                    # Older Streamlit: no event capture
                    _plotly(fig_rr, key="risk_return.map")
                except Exception:
                    _plotly(fig_rr, key="risk_return.map")

                # Capture selection -> store last-picked config id (no auto-open; user confirms via button)
                try:
                    _sel = getattr(_rr_event, "selection", None)
                    if isinstance(_sel, dict):
                        _pts = _sel.get("points") or []
                        if _pts:
                            _pt0 = _pts[0] or {}
                            _cd = _pt0.get("customdata")
                            _cid = None
                            if isinstance(_cd, (list, tuple, np.ndarray)):
                                _cid = _cd[0] if len(_cd) > 0 else None
                            else:
                                _cid = _cd
                            if _cid:
                                st.session_state["ui.rr_pick"] = str(_cid)
                                st.session_state["ui.rr_pick_x"] = _pt0.get("x")
                                st.session_state["ui.rr_pick_y"] = _pt0.get("y")
                                st.session_state["ui.rr_pick_cd"] = _cd
                except Exception:
                    pass

                # Mini evidence panel (under the map)
                rr_pick = st.session_state.get("ui.rr_pick")
                if rr_pick:
                    with st.container():
                        st.markdown(f"**Selected from map:** `{rr_pick}`")
                        st.caption("Tip: click a point to select it, or use the box/lasso tools in the chart toolbar. Hover is view-only.")

                        # Prefer details from customdata if available; otherwise look up row.
                        cd = st.session_state.get("ui.rr_pick_cd")
                        label = verdict = stability_pct = reason = None
                        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) >= 5:
                            label = str(cd[1]) if cd[1] is not None else ""
                            verdict = str(cd[2]) if cd[2] is not None else ""
                            stability_pct = str(cd[3]) if cd[3] is not None else ""
                            reason = str(cd[4]) if cd[4] is not None else ""
                        else:
                            _row = _lookup_row_by_config_id(
                                str(rr_pick),
                                hi2 if "hi2" in locals() else None,
                                hi if "hi" in locals() else None,
                                df_vis if "df_vis" in locals() else None,
                                df_show if "df_show" in locals() else None,
                                df_all if "df_all" in locals() else None,
                                df_req if "df_req" in locals() else None,
                                df2 if "df2" in locals() else None,
                                frames.get("full_all") if isinstance(frames, dict) else None,
                                frames.get("sweep_all") if isinstance(frames, dict) else None,
                                frames.get("sweep_passed") if isinstance(frames, dict) else None,
                            )
                            if _row is not None:
                                _d0 = _row.to_dict()
                                # Label
                                _lcol = _pick_col(pd.DataFrame([_d0]), ["config.label", "label", "config_label"])
                                if _lcol and _lcol in _d0:
                                    label = str(_d0.get(_lcol) or "")
                                # Verdict
                                if "grand.verdict" in _d0:
                                    verdict = str(_d0.get("grand.verdict") or "").upper()
                                # Stability percentile (best-effort)
                                try:
                                    _sc = None
                                    for _c in ["score.grand_robust", "robustness_score"]:
                                        if _c in _d0:
                                            _sc = _c
                                            break
                                    if _sc:
                                        stability_pct = _fmt_num(_d0.get(_sc))
                                except Exception:
                                    pass
                                # Reason snippet (best-effort; same logic as hover)
                                try:
                                    reason = _reason_snippet(_d0)
                                except Exception:
                                    reason = None

                        # Exact x/y used in the plot
                        x_dd = st.session_state.get("ui.rr_pick_x")
                        y_ret = st.session_state.get("ui.rr_pick_y")

                        c1, c2, c3, c4 = st.columns([0.24, 0.22, 0.22, 0.32])
                        with c1:
                            st.write("**Verdict**")
                            st.write(verdict or "—")
                            if label:
                                st.caption(f"Label: {label}")
                        with c2:
                            st.metric("Max DD", ("—" if x_dd is None or pd.isna(x_dd) else f"{float(x_dd)*100:.2f}%"))
                        with c3:
                            if is_currency:
                                st.metric("Return", ("—" if y_ret is None or pd.isna(y_ret) else f"${float(y_ret):,.0f}"))
                            else:
                                st.metric("Return", ("—" if y_ret is None or pd.isna(y_ret) else f"{float(y_ret)*100:.2f}%"))
                        with c4:
                            st.write("**Stability**")
                            st.write(stability_pct or "—")
                            if reason:
                                st.caption(reason)

                        bL, bR = st.columns([0.35, 0.65])
                        with bL:
                            if st.button("Open in Evidence ↓", key=f"rr.inspect.{rr_pick}", type="primary"):
                                st.session_state["ui.evidence_override_pick"] = str(rr_pick)
                                st.session_state["cockpit.pick"] = str(rr_pick)
                                st.session_state["ui.open_evidence"] = True
                                st.session_state["ui.jump_tab"] = "Batch scan"
                                st.rerun()
                        with bR:
                            st.caption("Opens the Evidence drawer below for this config. You’ll need to scroll down to reach it.")

                        if st.button("Clear selection", key="rr.clear"):
                            st.session_state["ui.rr_pick"] = None
                            st.session_state["ui.rr_pick_x"] = None
                            st.session_state["ui.rr_pick_y"] = None
                            st.session_state["ui.rr_pick_cd"] = None
                            st.rerun()

                # Stability distribution

                score_col2 = "score.grand_robust" if "score.grand_robust" in df_req.columns else None
                if score_col2:
                    s = pd.to_numeric(df_req[score_col2], errors="coerce").dropna()
                    if len(s) > 0:
                        _arr = s.to_numpy(dtype=float)
                        n = int(len(_arr))
                        med = float(np.nanmedian(_arr))
                        p80 = float(np.nanpercentile(_arr, 80))
                        p90 = float(np.nanpercentile(_arr, 90)) if n >= 10 else None

                        cutoff = None
                        try:
                            if "score.grand_robust" in df_show.columns and len(df_show) > 0:
                                _cut = pd.to_numeric(df_show["score.grand_robust"], errors="coerce").dropna()
                                if len(_cut) > 0:
                                    cutoff = float(_cut.min())
                        except Exception:
                            cutoff = None

                        st.markdown("#### Stability Score: how repeatable is this plan?")
                        st.markdown("Stability = **repeatability** across start dates & time windows (**less fragile = higher**).")
                        with st.expander("What is this score?", expanded=False):
                            st.markdown(
                                "- Combines **Batch**, **Rolling Starts**, and **Walkforward** checks.\n"
                                "- Higher is better: steadier behavior across time, fewer deep drawdowns, fewer ugly windows.\n"
                                "- Think **crash-test rating** for strategies — not horsepower."
                            )

                        with st.expander("How it’s calculated", expanded=False):
                            # Layer A — what it is (one sentence)
                            st.markdown(
                                "Stability is a **weighted** score: **worst-case performance minus pain penalties**, across **Walkforward + Rolling Starts + Batch**."
                            )

                            # Layer B — recipe (weights + components)
                            _lens_here = _get_active_grand_lens(use_profile_lens=bool(st.session_state.get("grand.use_profile_lens", True)))
                            _wdf = pd.DataFrame(
                                [
                                    {"Component": "Walkforward", "Weight": f"{int(round(100*_lens_here['wf_w']))}%", "Uses": "P10 return; P90 drawdown + P90 underwater"},
                                    {"Component": "Rolling Starts", "Weight": f"{int(round(100*_lens_here['rs_w']))}%", "Uses": "P10 return; P90 drawdown + P90 underwater"},
                                    {"Component": "Batch", "Weight": f"{int(round(100*_lens_here['bt_w']))}%", "Uses": "Total return; Max drawdown"},
                                ]
                            )
                            st.table(_wdf)
                            st.caption(f"Penalties: drawdown ×{_lens_here['dd_k']:.2g}, underwater ×{_lens_here['uw_k']:.2g}, batch drawdown ×{_lens_here['bt_dd_k']:.2g}. Missing evidence penalties may apply.")


                            # Layer C — your plan breakdown (dynamic)
                            _baseline_id = st.session_state.get("baseline_config_id")
                            _selected_id = st.session_state.get("cockpit.pick")

                            _opts = ["Baseline"]
                            if _selected_id and _baseline_id and (str(_selected_id) != str(_baseline_id)):
                                _opts = ["Baseline", "Selected"]
                                _which = st.radio(
                                    "Show breakdown for",
                                    options=_opts,
                                    index=0,
                                    horizontal=True,
                                    key="ui.stability.breakdown.which",
                                )
                            elif _selected_id and (not _baseline_id):
                                _opts = ["Selected"]
                                _which = "Selected"
                            else:
                                _which = "Baseline"

                            _target_id = _baseline_id if (_which == "Baseline") else _selected_id
                            if not _target_id:
                                st.caption("No baseline/selected config available yet.")
                            else:
                                _row_for = _lookup_row_by_config_id(
                                    str(_target_id),
                                    df2 if ("df2" in locals()) else None,
                                    df_show if ("df_show" in locals()) else None,
                                    df if ("df" in locals()) else None,
                                    frames.get("full_all") if isinstance(frames, dict) else None,
                                    frames.get("sweep_all") if isinstance(frames, dict) else None,
                                )
                                if _row_for is None:
                                    st.warning("Couldn't locate that config_id in the current tables.")
                                else:
                                    _bd = _stability_breakdown(_row_for.to_dict(), lens=_get_active_grand_lens(use_profile_lens=bool(st.session_state.get('grand.use_profile_lens', True))))

                                    def _f_pct(v: Any, d: int = 2) -> str:
                                        try:
                                            x = float(v)
                                            if not math.isfinite(x):
                                                return "n/a"
                                            return _fmt_pct(x, digits=d)
                                        except Exception:
                                            return "n/a"

                                    def _f_days(v: Any) -> str:
                                        try:
                                            x = float(v)
                                            if not math.isfinite(x):
                                                return "n/a"
                                            yrs = x / 365.0
                                            return f"{x:.0f} days ({yrs:.2f}y)"
                                        except Exception:
                                            return "n/a"

                                    wf = _bd.get("wf", {})
                                    rs = _bd.get("rs", {})
                                    bt = _bd.get("bt", {})

                                    wf_ret = wf.get("inputs", {}).get("return_p10", {}).get("value", float("nan"))
                                    wf_dd = wf.get("inputs", {}).get("dd_p90", {}).get("value", float("nan"))
                                    wf_uw = wf.get("inputs", {}).get("uw_days_p90", {}).get("value", float("nan"))

                                    rs_ret = rs.get("inputs", {}).get("twr_p10", {}).get("value", float("nan"))
                                    rs_dd = rs.get("inputs", {}).get("dd_p90", {}).get("value", float("nan"))
                                    rs_uw = rs.get("inputs", {}).get("uw_days_p90", {}).get("value", float("nan"))

                                    bt_ret = bt.get("inputs", {}).get("total_return", {}).get("value", float("nan"))
                                    bt_dd = bt.get("inputs", {}).get("max_drawdown", {}).get("value", float("nan"))

                                    _bd_rows = [
                                        {
                                            "Component": "Walkforward",
                                            "Weight": f"{int(round(100*float(wf.get('weight',0.0) or 0.0)))}%",
                                            "Return (P10)": _f_pct(wf_ret),
                                            "Drawdown (P90)": _f_pct(wf_dd),
                                            "Underwater (P90)": _f_days(wf_uw),
                                            "Component score": _fmt_num(wf.get("raw", float("nan"))),
                                            "Weighted": _fmt_num(wf.get("weighted", float("nan"))),
                                        },
                                        {
                                            "Component": "Rolling Starts",
                                            "Weight": f"{int(round(100*float(rs.get('weight',0.0) or 0.0)))}%",
                                            "Return (P10)": _f_pct(rs_ret),
                                            "Drawdown (P90)": _f_pct(rs_dd),
                                            "Underwater (P90)": _f_days(rs_uw),
                                            "Component score": _fmt_num(rs.get("raw", float("nan"))),
                                            "Weighted": _fmt_num(rs.get("weighted", float("nan"))),
                                        },
                                        {
                                            "Component": "Batch",
                                            "Weight": f"{int(round(100*float(bt.get('weight',0.0) or 0.0)))}%",
                                            "Return (total)": _f_pct(bt_ret),
                                            "Drawdown (max)": _f_pct(bt_dd),
                                            "Underwater (P90)": "—",
                                            "Component score": _fmt_num(bt.get("raw", float("nan"))),
                                            "Weighted": _fmt_num(bt.get("weighted", float("nan"))),
                                        },
                                    ]
                                    st.markdown("**Your plan breakdown**")
                                    st.dataframe(pd.DataFrame(_bd_rows), use_container_width=True, hide_index=True)

                                    # Per-component formula lines (short, code-style)
                                    st.caption("Per-component formulas (short):")
                                    st.code(
                                        f"""{wf.get('formula_str','wf = …')}
{rs.get('formula_str','rs = …')}
{bt.get('formula_str','bt = …')}""",
                                        language="text",
                                    )

                                    # Final arithmetic line
                                    pen_total = float(_bd.get("penalty_total", 0.0) or 0.0)
                                    st.code(
                                        f"""Score = {float(wf.get('weight',0.0)):.2f}*WF + {float(rs.get('weight',0.0)):.2f}*RS + {float(bt.get('weight',0.0)):.2f}*Batch − penalties
      = {float(wf.get('weight',0.0)):.2f}*{float(wf.get('raw', 0.0)):.4f} + {float(rs.get('weight',0.0)):.2f}*{float(rs.get('raw', 0.0)):.4f} + {float(bt.get('weight',0.0)):.2f}*{float(bt.get('raw', 0.0)):.4f} − {pen_total:.4f}
      = {float(_bd.get('total', 0.0)):.4f}""",
                                        language="text",
                                    )

                                    # Missing-data penalties
                                    _pens = _bd.get("penalties", []) or []
                                    if _pens:
                                        st.caption("Penalties applied (missing evidence):")
                                        for p in _pens:
                                            st.write(f"- **{p.get('component','?')}**: −{_fmt_num(p.get('amount', 0.0))} ({p.get('reason','')})")
                                    else:
                                        st.caption("No missing-data penalties applied.")

                                    # Optional: raw fields toggle
                                    _show_raw = st.checkbox("Show raw fields", value=False, key="ui.stability.breakdown.show_raw")
                                    if _show_raw:
                                        def _raw_line(label: str, item: Dict[str, Any]) -> str:
                                            c = item.get("col")
                                            v = item.get("value")
                                            return f"{label}: {c} = {v}"

                                        raw_lines = []
                                        for k, comp in [("WF", wf), ("RS", rs), ("Batch", bt)]:
                                            inp = comp.get("inputs", {}) or {}
                                            for kk, it in inp.items():
                                                raw_lines.append(_raw_line(f"{k}.{kk}", it if isinstance(it, dict) else {}))
                                        st.code("\n".join([l for l in raw_lines if l]), language="text")

                                    # Optional: action path
                                    st.caption("To increase Stability, you usually need:")
                                    st.markdown(
                                        "- higher **P10 returns** across windows\n"
                                        "- lower **P90 drawdowns**\n"
                                        "- shorter **underwater time**"
                                    )

                        # Focus the view on where the survivor population actually sits (avoid cutoff stretching the axis).
                        try:
                            x_p2 = float(np.nanpercentile(_arr, 5))
                            x_p995 = float(np.nanpercentile(_arr, 99))
                            if (not math.isfinite(x_p2)) or (not math.isfinite(x_p995)) or (x_p995 <= x_p2):
                                raise ValueError('bad percentiles')
                        except Exception:
                            x_p2 = float(np.nanmin(_arr))
                            x_p995 = float(np.nanmax(_arr))
                        x_pad = (x_p995 - x_p2) * 0.06 if (x_p995 > x_p2) else 1.0
                        x0 = x_p2 - x_pad
                        x1 = x_p995 + x_pad
                        cutoff_in_range = (cutoff is not None and math.isfinite(float(cutoff)) and (float(cutoff) >= x0) and (float(cutoff) <= x1))
                        cutoff_relevant = bool(cutoff_in_range)


                        # Baseline score (user's original config)
                        baseline_score = None
                        baseline_in_req = False
                        baseline_offscale = False
                        baseline_marker_x = None
                        try:
                            _bid = st.session_state.get("baseline_config_id")
                            if _bid:
                                # Prefer requirement-passed table if it contains the score column
                                _row_req = _lookup_row_by_config_id(_bid, df2) if ("df2" in locals()) else None
                                if _row_req is not None and score_col2 and (score_col2 in _row_req.index):
                                    _v = pd.to_numeric(pd.Series([_row_req[score_col2]]), errors="coerce").iloc[0]
                                    if pd.notna(_v) and math.isfinite(float(_v)):
                                        baseline_score = float(_v)
                                        baseline_in_req = True

                                # Fallback: find the baseline anywhere we can, even if filtered out of survivors
                                if baseline_score is None or (not math.isfinite(float(baseline_score))):
                                    _row_any = _lookup_row_by_config_id(
                                        _bid,
                                        df,
                                        frames.get("full_all") if isinstance(frames, dict) else None,
                                        frames.get("sweep_all") if isinstance(frames, dict) else None,
                                    )
                                    if _row_any is not None:
                                        if score_col2 and (score_col2 in _row_any.index):
                                            _v2 = pd.to_numeric(pd.Series([_row_any[score_col2]]), errors="coerce").iloc[0]
                                            if pd.notna(_v2) and math.isfinite(float(_v2)):
                                                baseline_score = float(_v2)
                                        if baseline_score is None or (not math.isfinite(float(baseline_score))):
                                            try:
                                                baseline_score = float(_grand_score_row(_row_any.to_dict(), lens=_get_active_grand_lens(use_profile_lens=bool(st.session_state.get('grand.use_profile_lens', True)))))
                                            except Exception:
                                                baseline_score = None
                        except Exception:
                            baseline_score = None

                        baseline_in_range = (
                            baseline_score is not None
                            and math.isfinite(float(baseline_score))
                            and (float(baseline_score) >= x0)
                            and (float(baseline_score) <= x1)
                        )
                        if baseline_score is not None and math.isfinite(float(baseline_score)):
                            baseline_marker_x = float(min(max(float(baseline_score), float(x0)), float(x1)))
                            baseline_offscale = not baseline_in_range
                        baseline_pct = float('nan')
                        baseline_pct_ok = False
                        baseline_zone = ''
                        try:
                            if baseline_score is not None and math.isfinite(float(baseline_score)):
                                baseline_pct = _pct_rank(_arr, float(baseline_score))
                            baseline_pct_ok = bool(baseline_pct is not None and math.isfinite(float(baseline_pct)))
                            baseline_zone = _stability_zone_label(float(baseline_pct)) if baseline_pct_ok else ''
                        except Exception:
                            baseline_pct = float('nan')
                            baseline_pct_ok = False
                            baseline_zone = ''
                        # Baseline gap vs typical (in standard deviations; interpretable distance)
                        baseline_z = None
                        try:
                            _std0 = float(np.nanstd(_arr))
                            if baseline_score is not None and math.isfinite(float(baseline_score)) and math.isfinite(_std0) and _std0 > 0:
                                baseline_z = (float(baseline_score) - float(med)) / _std0
                        except Exception:
                            baseline_z = None

                        # Summary (keep it tight)
                        tail_note = ""
                        try:
                            _minv = float(np.nanmin(_arr))
                            if math.isfinite(_minv) and (_minv < x0):
                                tail_note = f" · Tail off-scale (min {_fmt_num(_minv, digits=3)})"
                        except Exception:
                            tail_note = ""
                        st.caption(f"N={n:,} · View: P5–P99{tail_note}")
                        # Panel A: 'You vs survivors' percentile bar (fast comprehension)
                        if baseline_pct_ok:
                            st.markdown("##### Your position vs survivors")
                            delta_to_strong = max(0.0, 80.0 - float(baseline_pct))
                            _line = f"**You:** P{float(baseline_pct):.0f}"
                            if baseline_zone:
                                _line += f" ({baseline_zone})"
                            if baseline_z is not None and math.isfinite(float(baseline_z)):
                                _line += f" · vs typical: {float(baseline_z):+.2f}σ"
                            if not baseline_in_req:
                                _line += " · **filtered out**"
                            if baseline_offscale:
                                _line += " · off-scale"
                            if delta_to_strong > 0:
                                _line += f" · Need **+{delta_to_strong:.0f}** to reach Strong"
                            else:
                                _line += " · **Elite**" if float(baseline_pct) >= 90.0 else " · **Strong**"
                            st.markdown(_line)
                            fig_pct = _stability_percentile_bar_fig(float(baseline_pct))
                            if fig_pct is not None:
                                _plotly(fig_pct, key="stability_pct_bar")
                        with st.expander("Show breakpoint values (engine units)", expanded=False):
                            st.caption("These are the Stability *index* values at P50/P80/P90 for this run. Use percentiles above to interpret.")
                            if cutoff_relevant and (cutoff is not None and math.isfinite(float(cutoff))):
                                sc1, sc2, sc3, sc4 = st.columns(4)
                                sc1.metric("Typical (P50)", _fmt_num(med, digits=3))
                                sc2.metric("Strong (P80)", _fmt_num(p80, digits=3))
                                sc3.metric("Elite (P90)", _fmt_num(p90, digits=3) if (p90 is not None and math.isfinite(float(p90))) else "—")
                                sc4.metric("Cutoff", _fmt_num(cutoff, digits=3))
                            else:
                                sc1, sc2, sc3 = st.columns(3)
                                sc1.metric("Typical (P50)", _fmt_num(med, digits=3))
                                sc2.metric("Strong (P80)", _fmt_num(p80, digits=3))
                                sc3.metric("Elite (P90)", _fmt_num(p90, digits=3) if (p90 is not None and math.isfinite(float(p90))) else "—")

                        # Quick verdict (compact)
                        try:
                            _span = float(x_p995 - x_p2) if (math.isfinite(float(x_p995)) and math.isfinite(float(x_p2)) and (x_p995 > x_p2)) else float(x1 - x0)
                            _sep = float(p80 - med)
                            _ratio = (abs(_sep) / _span) if (_span and math.isfinite(_span) and _span > 0) else 0.0
                            if _ratio < 0.08:
                                _sep_lbl = 'small'
                            elif _ratio < 0.18:
                                _sep_lbl = 'moderate'
                            else:
                                _sep_lbl = 'large'
                        except Exception:
                            _sep_lbl = '—'
                        _cut_lbl = ''
                        if cutoff_relevant and (cutoff is not None and math.isfinite(float(cutoff))):
                            if float(cutoff) >= float(p80):
                                _cut_lbl = 'selective (strong+)'
                            else:
                                _cut_lbl = 'permissive'
                        _vline = f"Separation: **{_sep_lbl}**"
                        if _cut_lbl:
                            _vline += f" · Cutoff: {_cut_lbl}"
                        st.caption(_vline)
                        # Density curve + simple zones (cleaner than a histogram)
                        try:
                            _arr_f = _arr[np.isfinite(_arr)]
                            n_eff = int(len(_arr_f))
                            if n_eff < 3:
                                raise ValueError('not enough points for KDE')
                        
                            xs = np.linspace(float(x0), float(x1), 420)
                        
                            # Simple Gaussian KDE (no SciPy dependency)
                            std = float(np.nanstd(_arr_f))
                            try:
                                iqr = float(np.nanpercentile(_arr_f, 75) - np.nanpercentile(_arr_f, 25))
                            except Exception:
                                iqr = 0.0
                            sigma = std
                            if iqr and math.isfinite(iqr) and iqr > 0:
                                sigma = min(std, iqr / 1.349) if (std and math.isfinite(std) and std > 0) else (iqr / 1.349)
                            if (not sigma) or (not math.isfinite(sigma)) or sigma <= 0:
                                sigma = max(1e-9, float((x1 - x0) / 10.0))
                        
                            h = 1.06 * sigma * (n_eff ** (-1.0 / 5.0))
                            h_floor = float((x1 - x0) / 250.0) if (x1 > x0) else 1e-6
                            if (not math.isfinite(h)) or h <= 0:
                                h = h_floor
                            h = max(h, h_floor)
                        
                            u = (xs[:, None] - _arr_f[None, :]) / h
                            ys = np.exp(-0.5 * (u ** 2)).sum(axis=1) / (n_eff * h * math.sqrt(2.0 * math.pi))
                            y_max = float(np.nanmax(ys)) if len(ys) else 1.0

                            # Percentile tick marks (top axis) for interpretability
                            pct_tick_vals: List[float] = []
                            pct_tick_text: List[str] = []
                            try:
                                for _p, _lbl in [(10, "P10"), (50, "P50"), (80, "P80"), (90, "P90")]:
                                    _vv = float(np.nanpercentile(_arr_f, _p))
                                    if math.isfinite(_vv) and (float(x0) <= _vv <= float(x1)):
                                        pct_tick_vals.append(float(_vv))
                                        pct_tick_text.append(_lbl)
                            except Exception:
                                pct_tick_vals, pct_tick_text = [], []
                        
                            fig_den = go.Figure()
                            fig_den.add_trace(
                                go.Scatter(
                                    x=xs,
                                    y=ys,
                                    mode="lines",
                                    line=dict(color=ACCENT_BLUE, width=2),
                                    fill="tozeroy",
                                    hovertemplate="Survivor density<extra></extra>",
                                    showlegend=False,
                                )
                            )
                        
                            # Zones: right tail is where stronger candidates live
                            try:
                                fig_den.add_vrect(x0=float(p80), x1=float(x1), fillcolor=ACCENT_BLUE, opacity=0.08, line_width=0, layer="below")
                            except Exception:
                                pass
                            if p90 is not None and math.isfinite(float(p90)):
                                try:
                                    fig_den.add_vrect(x0=float(p90), x1=float(x1), fillcolor=ACCENT_BLUE, opacity=0.14, line_width=0, layer="below")
                                except Exception:
                                    pass
                        
                            def _add_marker_line(x: float, label: str, dash: str, color: str, hover_title: str, *, y_annot: float) -> None:
                                try:
                                    fig_den.add_trace(
                                        go.Scatter(
                                            x=[x, x],
                                            y=[0.0, y_max * 1.05],
                                            mode="lines",
                                            line=dict(color=color, width=2, dash=dash),
                                            hovertemplate=f"{hover_title}<br>Score: {_fmt_num(x, digits=3)}<extra></extra>",
                                            showlegend=False,
                                        )
                                    )
                                    fig_den.add_annotation(
                                        x=x, xref="x",
                                        y=y_annot, yref="paper",
                                        text=label,
                                        showarrow=False,
                                        xanchor="center",
                                        yanchor="bottom",
                                        font=dict(size=12, color=color),
                                    )
                                except Exception:
                                    pass
                        
                            _add_marker_line(float(med), "Typical", "solid", "#111827", "Typical survivor (P50)", y_annot=1.02)
                            _add_marker_line(float(p80), "Strong zone", "dash", ACCENT_BLUE, "Strong threshold (P80)", y_annot=1.08)
                            if p90 is not None and math.isfinite(float(p90)):
                                _add_marker_line(float(p90), "Elite zone", "dot", ACCENT_BLUE, "Elite threshold (P90)", y_annot=1.14)
                            if cutoff_in_range:
                                _add_marker_line(float(cutoff), "Cutoff", "dashdot", WARN_COLOR, "Visibility cutoff (UI)", y_annot=1.02)
                            if baseline_marker_x is not None and math.isfinite(float(baseline_marker_x)):
                                try:
                                    _x_you = float(baseline_marker_x)
                                    _you_text = ("You"
                                                 + (f" (P{float(baseline_pct):.0f})" if baseline_pct_ok else "")
                                                 + (f" · {float(baseline_z):+.2f}σ" if (baseline_z is not None and math.isfinite(float(baseline_z))) else "")
                                                 + (" · filtered out" if (not baseline_in_req) else "")
                                                 + (" · off-scale" if baseline_offscale else "")
                                                 )
                                    # Strong emphasis: draw last + thicker line + label bubble anchored to the line.
                                    fig_den.add_vline(
                                        x=_x_you,
                                        line_width=4,
                                        line_dash="solid",
                                        line_color="#111827",
                                    )
                                    fig_den.add_annotation(
                                        x=_x_you, xref="x",
                                        y=1.20, yref="paper",
                                        text=_you_text,
                                        showarrow=False,
                                        xanchor="center",
                                        yanchor="bottom",
                                        font=dict(size=11, color="rgba(255,255,255,0.98)"),
                                        bgcolor="rgba(17,24,39,0.92)",
                                        bordercolor="rgba(255,255,255,0.95)",
                                        borderwidth=1,
                                        borderpad=4,
                                    )
                                except Exception:
                                    _add_marker_line(
                                        float(baseline_marker_x),
                                        "You" if not baseline_offscale else "You (off)",
                                        "solid",
                                        "#111827",
                                        "You",
                                        y_annot=1.20,
                                    )

                            _style_fig(fig_den, title=None)
                            fig_den.update_layout(margin=dict(l=20, r=20, t=110, b=20))
                            fig_den.update_yaxes(title=None, showticklabels=False, showgrid=False, zeroline=False)

                            # Bottom axis: raw Stability index (engine units). Keep ticks sparse + rounded.
                            fig_den.update_layout(
                                xaxis=dict(
                                    title="Stability index (higher = more repeatable)",
                                    range=[float(x0), float(x1)],
                                    nticks=4,
                                    tickformat=".3f",
                                    ticks="outside",
                                )
                            )
                            # Top axis: percentile landmarks for fast interpretation (P10/P50/P80/P90)
                            if isinstance(pct_tick_vals, list) and len(pct_tick_vals) > 0:
                                fig_den.update_layout(
                                    xaxis2=dict(
                                        overlaying="x",
                                        side="top",
                                        tickmode="array",
                                        tickvals=pct_tick_vals,
                                        ticktext=pct_tick_text,
                                        showgrid=False,
                                        zeroline=False,
                                        ticks="outside",
                                        tickfont=dict(size=10, color="#6b7280"),
                                    )
                                )
                        
                        
                            st.markdown("##### Where survivors land on Stability")
                            _plotly(fig_den)
                            st.caption("Right = more repeatable. Higher curve = more survivor configs at that score.")
                        except Exception:
                            # Fallback: keep a simple histogram without stretching to cutoff
                            nbins = int(max(20, min(50, math.sqrt(n) * 3)))
                            fig_hist = go.Figure(
                                go.Histogram(x=_arr, nbinsx=nbins, marker=dict(color=ACCENT_BLUE), opacity=0.85, showlegend=False)
                            )
                            _style_fig(fig_hist, title=None)
                            fig_hist.update_layout(margin=dict(l=20, r=20, t=120, b=20))
                            fig_hist.update_xaxes(title="Stability index (higher = more repeatable)", range=[float(x0), float(x1)], nticks=6, tickformat=".3f")
                            fig_hist.update_yaxes(title="Count")
                            _plotly(fig_hist)
                # Common issues bar (clean labels + full detail in hover/table)
                try:
                    if hi2 is not None and "_reason" in hi2.columns:
                        _rr = hi2[["_reason"]].copy()
                        _rr["_reason"] = _rr["_reason"].astype(str).str.strip()
                        _rr = _rr[_rr["_reason"] != ""]
                        if len(_rr) > 0:
                            _sample_n = int(len(_rr))

                            def _parse_reason(_s: str) -> dict:
                                s = str(_s or "").strip()
                                stage = "Other"
                                body = s
                                m0 = re.match(r"^\s*([^:]+)\s*:\s*(.*)$", s)
                                if m0:
                                    stage = str(m0.group(1)).strip()
                                    body = str(m0.group(2)).strip()

                                # Pattern: '<metric> is <obs>%? but your limit is >=|<= <lim>%?'
                                m = re.search(
                                    r"(?P<metric>.+?)\s+is\s+(?P<obs>-?\d+(?:\.\d+)?)(?P<obs_pct>%?)\s+but\s+your\s+limit\s+is\s+(?P<op>>=|<=)\s+(?P<lim>-?\d+(?:\.\d+)?)(?P<lim_pct>%?)",
                                    body,
                                    flags=re.IGNORECASE,
                                )

                                metric = None
                                obs = None
                                lim = None
                                op = None
                                is_pct = False
                                issue = "outside limit"

                                if m:
                                    metric = str(m.group("metric")).strip().rstrip(".")
                                    op = str(m.group("op")).strip()
                                    try:
                                        obs = float(m.group("obs"))
                                        lim = float(m.group("lim"))
                                    except Exception:
                                        obs, lim = None, None
                                    is_pct = bool(m.group("obs_pct") or m.group("lim_pct") or ("%" in body))

                                    if obs is not None and lim is not None:
                                        if op == ">=" and obs < lim:
                                            issue = "too low"
                                        elif op == "<=" and obs > lim:
                                            issue = "too high"

                                # Human label for the chart (casual-friendly)
                                def _human_metric(_m: str) -> str:
                                    m = re.sub(r"\s+", " ", str(_m or "").strip())
                                    ml = m.lower()
                                    if "turnover" in ml:
                                        return "Trading frequency"
                                    if "invested fraction" in ml:
                                        return "Typical invested %"
                                    if "return_p50" in ml or (ml.startswith("return") and "p50" in ml):
                                        return "Median return"
                                    if "underwater time" in ml:
                                        return "Bad-streak length (p90)" if "p90" in ml else "Underwater time"
                                    m = m.replace("Typical ", "").strip()
                                    return m
                                
                                if metric:
                                    metric_short = _human_metric(metric)
                                else:
                                    metric_short = _human_metric(body)
                                if len(metric_short) > 44:
                                    metric_short = metric_short[:42] + "…"
                                
                                # Convert limit direction into plain phrasing
                                _issue_phrase = issue
                                if issue == "outside limit" and op == ">=":
                                    _issue_phrase = "below minimum"
                                elif issue == "outside limit" and op == "<=":
                                    _issue_phrase = "above maximum"
                                # Friendlier phrasing for a couple of common blockers
                                if metric_short.lower().startswith("typical invested") and _issue_phrase in ("too low", "below minimum"):
                                    short_label = "Not invested enough"
                                else:
                                    short_label = f"{metric_short} {_issue_phrase}".strip()
                                return {
                                    "stage": stage,
                                    "metric": metric or body,
                                    "metric_short": metric_short,
                                    "op": op or "",
                                    "limit": lim,
                                    "observed": obs,
                                    "is_pct": bool(is_pct),
                                    "issue": issue,
                                    "short_label": short_label,
                                    "raw": s,
                                }

                            _parsed = _rr["_reason"].map(_parse_reason).tolist()
                            df_p = pd.DataFrame(_parsed)
                            if not df_p.empty:
                                # Group duplicate reasons (same issue, different observed values)
                                g = (
                                    df_p.groupby(["stage","short_label","metric","op","limit","is_pct","issue"], dropna=False)
                                    .agg(
                                        count=("raw","size"),
                                        obs_median=("observed","median"),
                                        obs_min=("observed","min"),
                                        obs_max=("observed","max"),
                                        example=("raw","first"),
                                    )
                                    .reset_index()
                                )
                                g["pct"] = g["count"] / max(1, _sample_n)

                                def _fmt(v, is_pct: bool) -> str:
                                    if v is None or (isinstance(v, float) and not math.isfinite(v)):
                                        return "—"
                                    return (f"{v:.2f}%" if is_pct else f"{v:.4g}")

                                g["Observed"] = g.apply(
                                    lambda r: (
                                        f"{_fmt(r['obs_median'], r['is_pct'])} ({_fmt(r['obs_min'], r['is_pct'])}–{_fmt(r['obs_max'], r['is_pct'])})"
                                        if r.get("obs_median") is not None else "—"
                                    ),
                                    axis=1,
                                )
                                g["Limit"] = g.apply(
                                    lambda r: (f"{r['op']} {_fmt(r['limit'], r['is_pct'])}" if r.get("op") else "—"),
                                    axis=1,
                                )
                                g["Delta"] = g.apply(
                                    lambda r: (
                                        _fmt((r['obs_median'] - r['limit']) if (r.get('obs_median') is not None and r.get('limit') is not None) else None, r['is_pct'])
                                    ),
                                    axis=1,
                                )
                                g["Share"] = (g["pct"] * 100.0).round(0).astype(int).astype(str) + "%"
                                g["Label"] = g.apply(lambda r: f"{int(r['count'])} ({int(round(r['pct']*100.0))}%)", axis=1)
                                g["Example"] = g["example"].astype(str).str.slice(0, 140)

                                # Keep the chart focused: hide one-off noise (details table still keeps everything)
                                _rare = (g["count"] < 2) & (g["pct"] < 0.02)
                                g_main = g[~_rare].copy()
                                if _rare.any():
                                    g_other = pd.DataFrame([{
                                        "stage": "Other",
                                        "short_label": "Other (rare)",
                                        "metric": "Other (rare)",
                                        "op": "",
                                        "limit": None,
                                        "is_pct": False,
                                        "issue": "rare",
                                        "count": int(g.loc[_rare, "count"].sum()),
                                        "obs_median": None,
                                        "obs_min": None,
                                        "obs_max": None,
                                        "example": str(g.loc[_rare, "example"].iloc[0]) if len(g.loc[_rare]) else "",
                                        "pct": float(g.loc[_rare, "pct"].sum()),
                                    }])
                                    # Populate display fields so the table/hover don't show "None"
                                    g_other["Observed"] = "—"
                                    g_other["Limit"] = "—"
                                    g_other["Delta"] = "—"
                                    g_other["Share"] = (g_other["pct"] * 100.0).round(0).astype(int).astype(str) + "%"
                                    g_other["Label"] = g_other.apply(lambda r: f"{int(r['count'])} ({int(round(r['pct']*100.0))}%)", axis=1)
                                    g_other["Example"] = g_other["example"].astype(str).str.slice(0, 140)
                                    g_main = pd.concat([g_main, g_other], ignore_index=True)
                                
                                # Show the most common blockers
                                g_top = g_main.sort_values(["count","pct"], ascending=False).head(8).copy()
                                g_plot = g_top.sort_values(["count","pct"], ascending=True).copy()

                                # Plot: clean labels + stage color; full detail in hover
                                fig_r = px.bar(
                                    g_plot,
                                    x="count",
                                    y="short_label",
                                    color="stage",
                                    category_orders={"stage": ["Batch", "RS", "WF", "Other"]},
                                    orientation="h",
                                    text="Label",
                                    hover_data={
                                        "stage": True,
                                        "short_label": False,
                                        "count": False,
                                        "pct": False,
                                        "Observed": True,
                                        "Limit": True,
                                        "Delta": True,
                                        "Example": True,
                                    },
                                )
                                _style_fig(fig_r, title=None)
                                fig_r.update_layout(margin=dict(l=20, r=20, t=40, b=20), legend_title_text="")
                                fig_r.update_traces(textposition="outside", cliponaxis=False)
                                fig_r.update_yaxes(title=None, categoryorder="array", categoryarray=g_plot["short_label"].tolist())
                                fig_r.update_xaxes(title="Count (in sample)")

                                st.markdown("#### What\'s blocking the visible shortlist?")
                                st.caption(f"Top reasons candidates fail your current filters (N={_sample_n:,}).")
                                _plotly(fig_r)

                                with st.expander("Details (numbers)", expanded=False):
                                    df_det = g_top[["stage","short_label","count","Share","Observed","Limit","Delta","Example"]].copy()
                                    df_det = df_det.rename(columns={
                                        "stage": "Stage",
                                        "short_label": "Issue",
                                        "count": "Count",
                                    })
                                    for _c in ["Share", "Observed", "Limit", "Delta", "Example"]:
                                        if _c in df_det.columns:
                                            df_det[_c] = df_det[_c].fillna("—")
                                    st.dataframe(df_det, width="stretch", height=260)
                except Exception:
                    pass
            else:
                st.caption("Risk/return map unavailable (missing drawdown/return columns).")

        st.divider()

    # =========================
    # (Run overview is shown above; shortlist below reflects your current lens)

    st.subheader("Your shortlist")
    st.caption(f"Showing **{len(df_show):,}** candidates (out of **{len(df):,}** survivors evaluated).")
    if df_show.empty:
        st.info("No candidates under current rules. Relax constraints or run missing tests.")
        st.stop()

    # Shortlist table (keep it readable: decision columns only)
    _label_col = _pick_col(df_show, ["config.label", "label", "config_label"])
    _score_col = "score.grand_robust" if "score.grand_robust" in df_show.columns else ("robustness_score" if "robustness_score" in df_show.columns else None)
    _ret_col = _pick_col(df_show, ["performance.twr_total_return", "return_p50", "equity.net_profit_ex_cashflows", "equity.net_profit"])
    _dd_col = _pick_col(df_show, ["performance.max_drawdown_equity", "dd_p90", "performance.max_drawdown", "equity.max_drawdown"])

    show_cols = ["config_id"]
    if _label_col:
        show_cols.append(_label_col)
    for c in ["grand.verdict", "rsq.verdict", "wfq.verdict"]:
        if c in df_show.columns:
            show_cols.append(c)
    for c in [_score_col, _ret_col, _dd_col]:
        if c and c in df_show.columns and c not in show_cols:
            show_cols.append(c)
    for c in ["pct_profitable_windows", "pct_windows_traded"]:
        if c in df_show.columns:
            show_cols.append(c)

    display_df = df_show.reindex(columns=show_cols).copy()

    # Friendly labels
    rename_map = {
        "config_id": "ID",
        "grand.verdict": "Grand",
        "rsq.verdict": "RS",
        "wfq.verdict": "WF",
        "pct_profitable_windows": "% Profitable windows",
        "pct_windows_traded": "% Windows traded",
        "score.grand_robust": "Stability score",
        "robustness_score": "Stability score",
        "performance.twr_total_return": "Total return",
        "return_p50": "Median return (RS)",
        "equity.net_profit_ex_cashflows": "Net profit (ex deposits)",
        "equity.net_profit": "Net profit",
        "performance.max_drawdown_equity": "Max drawdown",
        "dd_p90": "P90 drawdown (RS)",
        "performance.max_drawdown": "Max drawdown",
        "equity.max_drawdown": "Max drawdown",
        "config.label": "Label",
        "label": "Label",
        "config_label": "Label",
    }
    display_df = display_df.rename(columns={c: rename_map.get(c, c) for c in display_df.columns})

    st.dataframe(display_df, width="stretch", height=520)
    # Evidence drawer state
    if "ui.open_evidence" not in st.session_state:
        st.session_state["ui.open_evidence"] = False

    with st.expander("Cards view (optional)", expanded=False):
        st.caption("Quick scan of the top rows in your shortlist. Same filters + sorting. Use **Inspect →** to jump straight to Evidence.")

        # Card styling helpers (small, readable chips)
        st.markdown(
            """
        <style>
        .ff-pill { display:inline-block; padding:2px 8px; border-radius:999px; font-size:0.78rem; line-height:1.4;
                   border:1px solid rgba(49,51,63,0.18); margin-right:6px; margin-bottom:4px; }
        .ff-pill.big { font-size:0.84rem; font-weight:600; padding:3px 10px; }
        .ff-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }

        .ff-neutral { background: rgba(149,165,166,0.12); }
        .ff-pass { background: rgba(46,204,113,0.14); }
        .ff-warn { background: rgba(241,196,15,0.18); }
        .ff-fail { background: rgba(231,76,60,0.14); }

        .ff-dot.ff-neutral { background: rgba(149,165,166,0.85); }
        .ff-dot.ff-pass { background: rgba(46,204,113,0.90); }
        .ff-dot.ff-warn { background: rgba(241,196,15,0.90); }
        .ff-dot.ff-fail { background: rgba(231,76,60,0.90); }
        </style>
        """,
            unsafe_allow_html=True,
        )

        skim_mode = st.checkbox("Skim mode (recommended)", value=True, key="cards.skim_mode")


        # =========================
        # Top 10 candidate cards (quick inspect)
        # =========================
        top_n = int(min(10, len(df_show)))
        _top10 = df_show.head(top_n).copy()

        # Rolling Starts failure threshold comes from the current preference choice.
        # We interpret it as: "a start is a failure if its TWR return is below this tolerance".
        def _rs_failure_threshold_from_answers(ans: Dict[str, int]) -> float:
            try:
                q = next(q for q in rolling_questions() if q.id == "rs_worst_return")
                idx = int(ans.get(q.id, int(q.default_index)))
                idx = max(0, min(idx, len(q.choices) - 1))
                choice = q.choices[idx]
                for c in choice.constraints:
                    if str(c.metric_id) == "twr_p10":
                        return float(c.threshold)
            except Exception:
                pass
            # If "Don't filter on this" (or anything weird), default to a simple "below 0%" notion of failure.
            return 0.0

        _rs_fail_thr = _rs_failure_threshold_from_answers(rs_ans)
        _rs_detail = load_rs_detail(run_dir, rs_dir_effective) if rs_dir_effective else None
        _wf_detail = load_wf_results(wf_dir_effective) if wf_dir_effective else None

        # Per-config quantile caches for mini distribution strips (fast + stable across reruns)
        _dist_key = (
            "boxstrip_cache_v1",
            str(run_dir),
            str(rs_dir_effective) if rs_dir_effective else "",
            str(wf_dir_effective) if wf_dir_effective else "",
        )
        if "_boxstrip_cache" not in st.session_state or st.session_state["_boxstrip_cache"].get("key") != _dist_key:
            st.session_state["_boxstrip_cache"] = {"key": _dist_key, "rs": {}, "wf": {}}

        def _compute_boxstrip_quantiles(detail_df, value_col: str, cid: str):
            if detail_df is None or getattr(detail_df, "empty", True):
                return None
            if "config_id" not in detail_df.columns or value_col not in detail_df.columns:
                return None
            mask = detail_df["config_id"].astype(str) == str(cid)
            if not mask.any():
                return None
            s = pd.to_numeric(detail_df.loc[mask, value_col], errors="coerce").dropna()
            if s.empty:
                return None
            try:
                qs = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90]).to_dict()
                return {
                    "p10": float(qs.get(0.10)),
                    "p25": float(qs.get(0.25)),
                    "p50": float(qs.get(0.50)),
                    "p75": float(qs.get(0.75)),
                    "p90": float(qs.get(0.90)),
                    "n": int(len(s)),
                }
            except Exception:
                return None

        def _get_rs_boxstrip_q(cid: str):
            cache = st.session_state["_boxstrip_cache"]["rs"]
            cid = str(cid)
            if cid in cache:
                return cache[cid]
            q = _compute_boxstrip_quantiles(_rs_detail, "performance.twr_total_return", cid)
            cache[cid] = q
            return q

        def _get_wf_boxstrip_q(cid: str):
            cache = st.session_state["_boxstrip_cache"]["wf"]
            cid = str(cid)
            if cid in cache:
                return cache[cid]
            q = _compute_boxstrip_quantiles(_wf_detail, "window_return", cid)
            cache[cid] = q
            return q

        def _fmt_pct(x: Any, digits: int = 1) -> str:
            try:
                v = float(x)
                if not math.isfinite(v):
                    return "—"
                return f"{v * 100:.{digits}f}%"
            except Exception:
                return "—"

        def _fmt_num(x: Any, digits: int = 2) -> str:
            try:
                v = float(x)
                if not math.isfinite(v):
                    return "—"
                return f"{v:,.{digits}f}"
            except Exception:
                return "—"

        def _chip(v: str) -> str:
            v = str(v or "").upper().strip()
            if v == "PASS":
                return "✅ PASS"
            if v == "WARN":
                return "⚠️ WARN"
            if v in {"FAIL", "UNMEASURED"}:
                return "❌ FAIL"
            return v or "—"


        def _truncate(s: str, n: int = 36) -> str:
            s = str(s or "")
            if len(s) <= n:
                return s
            return s[: max(0, n - 1)] + "…"

        def _status_class(v: str) -> str:
            v = str(v or "").upper().strip()
            if v == "PASS":
                return "ff-pass"
            if v == "WARN":
                return "ff-warn"
            if v in {"FAIL", "UNMEASURED"}:
                return "ff-fail"
            return "ff-neutral"

        def _pill(text: str, status: str, big: bool = False, title: str = "") -> str:
            cls = _status_class(status)
            big_cls = " big" if big else ""
            tt = f' title="{title}"' if title else ""
            return f'<span class="ff-pill {cls}{big_cls}"{tt}>{text}</span>'

        def _stage_pill(stage: str, status: str, title: str = "") -> str:
            cls = _status_class(status)
            dot = f'<span class="ff-dot {cls}"></span>'
            return _pill(f"{dot}{stage}", status=status, big=False, title=title)

        def _issue_summary(reason: str) -> str:
            """Return a short, casual-user friendly blocker label."""
            r = str(reason or "").strip()
            if not r or r.lower().startswith("no violations"):
                return "No obvious blockers under current preferences."
            lo = r.lower()
            if "turnover" in lo or "trading frequency" in lo:
                return "Trading frequency too high"
            if "invested fraction" in lo or "invested %" in lo:
                return "Typical invested % too low"
            if "underwater time" in lo:
                return "Too much time underwater"
            if "return_p50" in lo or "median return" in lo:
                return "Median return too low"
            if "max drawdown" in lo or "drawdown" in lo:
                return "Drawdown too high"
            if ":" in r:
                return r.split(":", 1)[1].strip()
            return r

        # Percentile for robustness score (within the visible table population).
        _score_col = None
        for c in ["score.grand_robust", "robustness_score"]:
            if c in df_show.columns:
                _score_col = c
                break
        _score_pct = {}
        if _score_col:
            try:
                _s = pd.to_numeric(df_show[_score_col], errors="coerce")
                _r = _s.rank(pct=True, ascending=True)
                _score_pct = {str(cid): float(pct) for cid, pct in zip(df_show["config_id"].astype(str), _r)}
            except Exception:
                _score_pct = {}

        def _top_reason_snippet(row_dict: Dict[str, Any]) -> str:
            # Pick the most severe (critical > warn > info) message across stages, if any.
            sev_rank = {"critical": 0, "warn": 1, "info": 2}
            best = None  # (sev_rank, stage, msg)
            try:
                out_b = evaluate_row_with_questions(row_dict, batch_questions(), batch_ans)
                for v in out_b.violations:
                    r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                    msg = str(v.get("message", "")).strip()
                    if msg and (best is None or r < best[0]):
                        best = (r, "Batch", msg)
            except Exception:
                pass

            if rs_sum is not None and not rs_sum.empty:
                try:
                    out_r = evaluate_row_with_questions(row_dict, rolling_questions(), rs_ans)
                    for v in out_r.violations:
                        r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                        msg = str(v.get("message", "")).strip()
                        if msg and (best is None or r < best[0]):
                            best = (r, "RS", msg)
                except Exception:
                    pass

            if wf_sum is not None and not wf_sum.empty:
                try:
                    out_w = evaluate_row_with_questions(row_dict, walkforward_questions(), wf_ans)
                    for v in out_w.violations:
                        r = sev_rank.get(str(v.get("severity", "info")).lower(), 9)
                        msg = str(v.get("message", "")).strip()
                        if msg and (best is None or r < best[0]):
                            best = (r, "WF", msg)
                except Exception:
                    pass

            if best is None:
                return "No violations under current preferences."
            msg = best[2]
            if len(msg) > 120:
                msg = msg[:117].rstrip() + "…"
            return f"{best[1]}: {msg}"

        def _rs_n_starts(config_id: str) -> Optional[int]:
            if _rs_detail is None or _rs_detail.empty or "config_id" not in _rs_detail.columns:
                return None
            try:
                return int((_rs_detail["config_id"].astype(str) == str(config_id)).sum())
            except Exception:
                return None

        def _wf_n_windows(config_id: str) -> Optional[int]:
            if _wf_detail is None or _wf_detail.empty or "config_id" not in _wf_detail.columns:
                return None
            try:
                return int((_wf_detail["config_id"].astype(str) == str(config_id)).sum())
            except Exception:
                return None

        def _rs_fail_rate(config_id: str) -> Optional[float]:
            if _rs_detail is None or _rs_detail.empty or "config_id" not in _rs_detail.columns:
                return None
            d = _rs_detail[_rs_detail["config_id"].astype(str) == str(config_id)].copy()
            if d.empty or "performance.twr_total_return" not in d.columns:
                return None
            vals = pd.to_numeric(d["performance.twr_total_return"], errors="coerce").dropna()
            if vals.empty:
                return None
            return float((vals < float(_rs_fail_thr)).mean())

        cards_cols = st.columns(2, gap="medium")
        for i, (_, r) in enumerate(_top10.iterrows()):
            cid = str(r.get("config_id", "")).strip()
            label = str(r.get("config.label", "")).strip() if "config.label" in _top10.columns else ""
            grand_v = str(r.get("grand.verdict", "")).strip()
            batch_v = str(r.get("batch.verdict", "")).strip()
            rs_v = str(r.get("rsq.verdict", "")).strip()
            wf_v = str(r.get("wfq.verdict", "")).strip()

            rr = r.to_dict()
            reason = _top_reason_snippet(rr)

            # Core stats
            batch_ret = rr.get("performance.twr_total_return", np.nan)
            batch_dd = rr.get("performance.max_drawdown_equity", np.nan)

            rs_p10 = rr.get("twr_p10", np.nan)

            rs_p50 = rr.get("twr_p50", np.nan)
            rs_p90 = rr.get("twr_p90", np.nan)
            rs_fail = _rs_fail_rate(cid)

            wf_p10 = rr.get("return_p10", np.nan)
            wf_p50 = rr.get("return_p50", np.nan)
            wf_p90 = rr.get("return_p90", np.nan)

            score_pct = _score_pct.get(cid)
            score_line = f"{int(round(score_pct * 100))}th pct" if (score_pct is not None and math.isfinite(score_pct)) else "—"

            # Small "dopamine" signal: a simple grade + a confidence/progress bar.
            _gv = str(grand_v or "").upper().strip()
            if _gv == "PASS":
                if score_pct is not None and score_pct >= 0.90:
                    grade = "S"
                elif score_pct is not None and score_pct >= 0.75:
                    grade = "A"
                else:
                    grade = "A-"
            elif _gv == "WARN":
                grade = "B"
            else:
                grade = "C"

            base = {"PASS": 0.75, "WARN": 0.55, "FAIL": 0.35}.get(_gv, 0.45)
            sp = float(score_pct) if (score_pct is not None and math.isfinite(score_pct)) else 0.50
            # Map percentile (0..1) into a gentle +/- adjustment.
            adj = 0.20 * ((sp - 0.50) * 2.0)  # -0.20 .. +0.20
            confidence = float(max(0.0, min(1.0, base + adj)))

            stage_checks = []
            if batch_v:
                stage_checks.append(str(batch_v).upper())
            if rs_sum is not None and not rs_sum.empty and rs_v:
                stage_checks.append(str(rs_v).upper())
            if wf_sum is not None and not wf_sum.empty and wf_v:
                stage_checks.append(str(wf_v).upper())
            checks_total = len(stage_checks)
            checks_passed = sum(1 for v in stage_checks if v == "PASS")
            checks_ratio = (checks_passed / checks_total) if checks_total else 0.0

            with cards_cols[i % 2]:
                with st.container():
                    # Header
                    hL, hR = st.columns([0.78, 0.22])
                    with hL:
                        st.markdown(f"**{_truncate(label or cid, 44)}**")
                        st.caption(f"`{_truncate(cid, 18)}`")
                    with hR:
                        st.markdown(f"**#{i + 1}**")
                        st.caption(f"Badge: **{grade}**")

                    # Verdict row (Grand + stage chips)
                    _grand_txt = f"Grand: {_chip(grand_v)}"
                    chips = [_pill(_grand_txt, grand_v, big=True, title="Overall verdict across all checks")]
                    if batch_v:
                        chips.append(_stage_pill("Batch", batch_v, title=f"Batch: {_chip(batch_v)}"))
                    if rs_sum is not None and not rs_sum.empty and rs_v:
                        chips.append(_stage_pill("Start-date", rs_v, title=f"Rolling Starts: {_chip(rs_v)}"))
                    if wf_sum is not None and not wf_sum.empty and wf_v:
                        chips.append(_stage_pill("Time-split", wf_v, title=f"Walkforward: {_chip(wf_v)}"))
                    st.markdown("".join(chips), unsafe_allow_html=True)

                    # Confidence checks (chips) + overall fit
                    cL, cR = st.columns([0.62, 0.38])
                    with cL:
                        st.caption(f"Confidence checks: {checks_passed}/{checks_total}" if checks_total else "Confidence checks: —")
                        st.progress(float(checks_ratio) if checks_total else 0.0)
                    with cR:
                        st.caption(f"Overall fit: {int(round(confidence * 100))}/100")
                        st.progress(float(confidence))

                    # Metrics (skim) + robustness distributions (top 10)
                    if skim_mode:
                        b1, b2, b3 = st.columns(3)
                        with b1:
                            st.caption("Repeatability")
                            st.metric("Stability", score_line)
                        with b2:
                            st.caption("Return behavior")
                            st.metric("Batch return", _fmt_pct(batch_ret))
                        with b3:
                            st.caption("Drawdown risk")
                            st.metric("Max drawdown", _fmt_pct(batch_dd))

                        st.caption("Robustness distributions")
                        dL, dR = st.columns(2, gap="small")

                        with dL:
                            if rs_sum is None or rs_sum.empty:
                                st.caption("Return % if started on different days")
                                st.caption("Not computed")
                            else:
                                q = _get_rs_boxstrip_q(cid) if "_get_rs_boxstrip_q" in locals() else None
                                if q is None:
                                    fig_rs = _dist_bar_fig(rs_p10, rs_p50, rs_p90, "Return % if started on different days", zero_line=True)
                                    st.plotly_chart(fig_rs, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"rs_bar_{cid}")
                                    try:
                                        _rs_spread = float(rs_p90) - float(rs_p10)
                                    except Exception:
                                        _rs_spread = None
                                    if _rs_spread is not None:
                                        st.caption(f"Spread: {_fmt_pct(_rs_spread, digits=1)}")
                                else:
                                    fig_rs = _dist_boxstrip_fig(q["p10"], q["p25"], q["p50"], q["p75"], q["p90"], "Return % if started on different days", zero_line=True)
                                    st.plotly_chart(fig_rs, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"rs_box_{cid}")
                                    spread = q["p90"] - q["p10"]
                                    st.caption(f"Typical zone: {_fmt_pct(q['p25'], digits=1)} → {_fmt_pct(q['p75'], digits=1)} • Spread: {_fmt_pct(spread, digits=1)} • N={q['n']} starts")

                        with dR:
                            if wf_sum is None or wf_sum.empty:
                                st.caption("Return % across different market periods")
                                st.caption("Not computed")
                            else:
                                q = _get_wf_boxstrip_q(cid) if "_get_wf_boxstrip_q" in locals() else None
                                if q is None:
                                    fig_wf = _dist_bar_fig(wf_p10, wf_p50, wf_p90, "Return % across different market periods", zero_line=True)
                                    st.plotly_chart(fig_wf, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"wf_bar_{cid}")
                                    try:
                                        _wf_spread = float(wf_p90) - float(wf_p10)
                                    except Exception:
                                        _wf_spread = None
                                    if _wf_spread is not None:
                                        st.caption(f"Spread: {_fmt_pct(_wf_spread, digits=1)}")
                                else:
                                    fig_wf = _dist_boxstrip_fig(q["p10"], q["p25"], q["p50"], q["p75"], q["p90"], "Return % across different market periods", zero_line=True)
                                    st.plotly_chart(fig_wf, use_container_width=True, theme=None, config={"displayModeBar": False}, key=f"wf_box_{cid}")
                                    spread = q["p90"] - q["p10"]
                                    st.caption(f"Typical zone: {_fmt_pct(q['p25'], digits=1)} → {_fmt_pct(q['p75'], digits=1)} • Spread: {_fmt_pct(spread, digits=1)} • N={q['n']} windows")

                        # Primary blocker (skim) + optional details
                    short_issue = _issue_summary(reason)
                    st.markdown("<div style='height:1px;background:rgba(49,51,63,0.10);margin:0.35rem 0 0.45rem 0;'></div>", unsafe_allow_html=True)
                    if short_issue.lower().startswith("no obvious blockers"):
                        st.markdown("**Primary blocker:** None found ✅")
                    else:
                        st.markdown(f"**Primary blocker:** {short_issue}")

                    footL, footR = st.columns([0.72, 0.28])
                    with footL:
                        with st.expander("Details", expanded=(not skim_mode)):
                            st.write("**Strategy**")
                            st.code(cid)
                            if label and label != cid:
                                st.caption(f"Label: {label}")

                            st.write("**Verdicts**")
                            st.write(f"Grand: {_chip(grand_v)}")
                            if batch_v:
                                st.write(f"Batch: {_chip(batch_v)}")
                            if rs_sum is not None and not rs_sum.empty and rs_v:
                                st.write(f"Rolling Starts: {_chip(rs_v)}")
                            if wf_sum is not None and not wf_sum.empty and wf_v:
                                st.write(f"Walkforward: {_chip(wf_v)}")

                            st.write("**Numbers**")
                            st.write(
                                {
                                    "Stability": score_line,
                                    "Batch return": _fmt_pct(batch_ret),
                                    "Max DD": _fmt_pct(batch_dd),
                                    "RS p10": _fmt_pct(rs_p10),
                                    "RS fail rate": ("—" if rs_fail is None else f"{rs_fail*100:.0f}%"),
                                    "WF p10/p50": ("—" if wf_sum is None or wf_sum.empty else f"{_fmt_pct(wf_p10)} / {_fmt_pct(wf_p50)}"),
                                    "Overall fit": f"{int(round(confidence*100))}/100",
                                }
                            )

                            st.write("**Raw reason**")
                            st.code(reason)

                    with footR:
                        if st.button("Inspect →", key=f"top10.inspect.{cid}", type="primary"):
                            st.session_state["cockpit.pick"] = str(cid)
                            st.session_state["ui.open_evidence"] = True
                            st.session_state["ui.jump_tab"] = "Batch scan"
                            st.rerun()


    st.divider()

    # Allow inspection of a config selected elsewhere (e.g., Risk/return map) even if it is not in the current shortlist.
    if "ui.evidence_override_pick" not in st.session_state:
        st.session_state["ui.evidence_override_pick"] = None
    _override_pick = st.session_state.get("ui.evidence_override_pick")

    _pick_opts = df_show["config_id"].astype(str).tolist()[:5000]
    if _override_pick and str(_override_pick) not in set(_pick_opts):
        _pick_opts = [str(_override_pick)] + _pick_opts

    pick = st.selectbox(
        "Select a strategy to inspect",
        options=_pick_opts,
        index=0,
        key="cockpit.pick",
    )
    if not pick:
        st.stop()



    # Open Evidence drawer (manual path if you didn't use an Inspect button)
    open_evidence = bool(st.session_state.get('ui.open_evidence', False))
    obL, obR = st.columns([0.22, 0.78])
    with obL:
        if st.button('Open Evidence →', key='evidence.open_outside', type='primary'):
            st.session_state['ui.open_evidence'] = True
            st.session_state['ui.jump_tab'] = 'Batch scan'
            st.rerun()
    with obR:
        st.caption('Opens the Evidence drawer below for the selected strategy. (Cards view **Inspect →** does this automatically.)')

    # ------------------------------------------------------------
    # Evidence drawer (collapsed by default; opens when you Inspect)
    # ------------------------------------------------------------
    open_evidence = bool(st.session_state.get("ui.open_evidence", False))
    _ev_label = "Evidence (open to inspect a candidate)"
    if open_evidence:
        try:
            _rv = df2[df2['config_id'].astype(str) == str(pick)].iloc[0].to_dict()
            _gv = str(_rv.get('grand.verdict', '')).upper().strip()
            _ev_label = f"Evidence — {pick}" + (f" ({_gv})" if _gv else "")
        except Exception:
            _ev_label = f"Evidence — {pick}"

    with st.expander(_ev_label, expanded=open_evidence):
        if not open_evidence:
            st.caption("Pick a strategy above and click **Open Evidence →** (or use **Inspect →** in Cards view) to open the autopsy drawer.")
        else:
            # Drawer controls
            _cL, _cR = st.columns([0.78, 0.22])
            with _cL:
                st.caption("Autopsy for the selected strategy. Close the drawer to keep browsing the shortlist.")
            with _cR:
                if st.button("Close", key="evidence.close"):
                    st.session_state["ui.open_evidence"] = False
                    st.rerun()


            row = df2[df2["config_id"].astype(str) == str(pick)].iloc[0].to_dict()

            cfg_map = {r.get("config_id"): r.get("normalized") for r in _load_jsonl(run_dir / "configs_resolved.jsonl")}
            cfg_norm = cfg_map.get(str(pick), {})

            art_dir = top_map.get(str(pick))
            if not (art_dir and art_dir.exists()):
                cache_dir = run_dir / "replay_cache" / str(pick)
                if cache_dir.exists():
                    art_dir = cache_dir

            st.divider()


            # Selected strategy summary (quick read before diving into tabs)
            with st.container():

                dd_col_s = _pick_col(df2, ["performance.max_drawdown_equity", "performance.max_drawdown", "equity.max_drawdown", "dd_p90"])
                ret_col_s = _pick_col(df2, ["performance.twr_total_return", "equity.net_profit_ex_cashflows", "equity.net_profit", "return_p50"])
                # Single source-of-truth context for the selected config (prevents mismatched stats)
                ctx = _resolve_selected_ctx(
                    run_dir,
                    pick,
                    df2=df2 if 'df2' in locals() else None,
                    top_map=top_map if 'top_map' in locals() else None,
                    rs_dir_effective=rs_dir_effective if 'rs_dir_effective' in locals() else None,
                    wf_dir_effective=wf_dir_effective if 'wf_dir_effective' in locals() else None,
                )
                row = ctx.get("row", {}) or {}
                cfg_norm = ctx.get("cfg_norm", {}) or {}
                art_dir = ctx.get("art_dir")
                trades_n = int(ctx.get("trades_n") or 0)

                has_batch = True  # Batch exists for any run
                has_rs = bool(ctx.get("has_rs"))
                has_wf = bool(ctx.get("has_wf"))
                has_receipts = bool(ctx.get("has_receipts", True))

                # Verdicts (shown as badges)
                v_grand = str(row.get("grand.verdict", "—"))
                v_batch = str(row.get("batch.verdict", row.get("batchq.verdict", row.get("batch.verdict", "—"))))
                v_rs = str(row.get("rs.verdict", row.get("rolling_starts.verdict", "—")))
                v_wf = str(row.get("wf.verdict", row.get("walkforward.verdict", "—")))

                hL, hR = st.columns([0.62, 0.38], gap="large", vertical_alignment="top")
                with hL:
                    st.markdown("#### Strategy dossier")
                    _ff_copy_id(str(pick), key=f"dossier.copy.{pick}")

                    # Context chips (optional quick context; keep minimal)
                    chips: List[str] = []
                    _market = str(cfg_norm.get("market") or cfg_norm.get("venue") or "").strip()
                    _grade = str(row.get("grade", row.get("rank.band", row.get("rank_band", ""))) or "").strip()

                    if _market:
                        chips.append(_market)
                    if _grade:
                        chips.append(f"Rank band: {_grade}")

                    if chips:
                        st.markdown(_ff_chip_row_html(chips), unsafe_allow_html=True)

                with hR:
                    st.markdown(
                        _ff_badge_stack_html([
                            ("Grand", v_grand, True),
                            ("Batch", v_batch, False),
                            ("Start-date", v_rs if has_rs else "—", False),
                            ("Time-split", v_wf if has_wf else "—", False),
                        ]),
                        unsafe_allow_html=True,
                    )

                # KPI strip (readouts tied to this selection)
                dd_s = row.get(dd_col_s) if dd_col_s else None
                ret_s = row.get(ret_col_s) if ret_col_s else None
                score_pct = row.get("stability.score_pct", row.get("stability_pct", None))

                rep_val = "—"
                if score_pct is not None:
                    try:
                        rep_val = f"Top {100 - int(float(score_pct)):.0f}%"
                    except Exception:
                        rep_val = "—"

                ret_val = _fmt_pct(ret_s, digits=1) if (ret_s is not None and math.isfinite(float(ret_s))) else "—"
                dd_val = _fmt_pct(dd_s, digits=1) if (dd_s is not None and math.isfinite(float(dd_s))) else "—"

                st.markdown(
                    _ff_kpi_strip_html([
                        ("Repeatability", rep_val, True),
                        ("Total return", ret_val, True),
                        ("Max drawdown", dd_val, False),
                        ("Trades", str(trades_n), False),
                    ]),
                    unsafe_allow_html=True,
                )

                st.markdown("##### Robustness checks")
                st.caption("Run remaining robustness checks for this run’s survivor set. Checks lock per run.")

                with st.container():
                    # Detect whether each robustness stage has already run for this run.
                    rs_done = False
                    wf_done = False
                    try:
                        _rs_dir = rs_latest if 'rs_latest' in globals() else None
                        if _rs_dir is None:
                            _rs_dir = locals().get("rs_dir_effective")
                        _rs_sum = load_rs_summary(run_dir, _rs_dir) if _rs_dir else None
                        rs_done = bool(_rs_sum is not None and (not _rs_sum.empty))
                    except Exception:
                        rs_done = False

                    try:
                        _wf_dir = wf_latest if 'wf_latest' in globals() else None
                        if _wf_dir is None:
                            _wf_dir = locals().get("wf_dir_effective")
                        _wf_sum = load_wf_summary(_wf_dir) if _wf_dir else None
                        wf_done = bool(_wf_sum is not None and (not _wf_sum.empty))
                    except Exception:
                        wf_done = False

                    # Needed for RS/WF commands (same survivor set the run was built on).
                    try:
                        _frames = load_batch_frames(run_dir)
                        _surv, _src = pick_survivors(_frames)
                        survivor_ids = _surv["config_id"].astype(str).tolist() if (_surv is not None and (not _surv.empty) and ("config_id" in _surv.columns)) else []
                    except Exception:
                        survivor_ids = []
                    N = int(len(survivor_ids))

                    meta = {}
                    try:
                        _mp = run_dir / "batch_meta.json"
                        if _mp.exists():
                            meta = _read_json(_mp)
                    except Exception:
                        meta = {}

                    bars_per_day = 1
                    try:
                        bars_per_day = int(_bars_per_day_from_run_meta(run_dir))
                    except Exception:
                        bars_per_day = 1

                    # Batch row (always ran for this run)
                    r0 = st.columns([0.70, 0.30], gap="small")
                    with r0[0]:
                        st.markdown(
                            _ff_badge_html("Batch", "ran") + " <span style='opacity:.72'>Batch scan already ran for this run.</span>",
                            unsafe_allow_html=True,
                        )
                    with r0[1]:
                        st.button("Locked", key=f"suite.batch.locked.{pick}", disabled=True, use_container_width=True)

                    # Start-date (Rolling Starts)
                    r1 = st.columns([0.70, 0.30], gap="small")
                    with r1[0]:
                        _rs_label = v_rs if (rs_done and str(v_rs).strip() and str(v_rs).strip() != "—") else ("ran" if rs_done else "Not run")
                        st.markdown(
                            _ff_badge_html("Start-date", _rs_label) + " <span style='opacity:.72'>Start-date sensitivity.</span>",
                            unsafe_allow_html=True,
                        )
                    with r1[1]:
                        if rs_done:
                            st.button("Locked", key=f"suite.rs.done.{pick}", disabled=True, use_container_width=True)
                        else:
                            if st.button("Run", key=f"suite.rs.run.{pick}", type="primary", disabled=(N == 0), use_container_width=True):
                                try:
                                    start_step = int(max(1, round(7 * bars_per_day)))
                                    min_bars = int(max(30, round(365 * bars_per_day)))
                                    rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}"
                                    if rs_out_dir.exists():
                                        rs_out_dir = rs_root / f"rs_step{start_step}_min{min_bars}_n{N}_{int(time.time())}"
                                    rs_progress = rs_out_dir / "progress" / "rolling_starts.jsonl"
                                    rs_progress.parent.mkdir(parents=True, exist_ok=True)

                                    cmd = [
                                        PY, "-m", "research.rolling_starts",
                                        "--from-run", str(run_dir),
                                        "--out", str(rs_out_dir),
                                        "--top-n", str(N),
                                        "--start-step", str(start_step),
                                        "--min-bars", str(min_bars),
                                        "--seed", "1",
                                        "--starting-equity", str(float(meta.get("starting_equity", 1000.0) or 1000.0)),
                                        "--jobs", "8",
                                        "--no-progress",
                                        "--progress-file", str(rs_progress),
                                        "--progress-every", "25",
                                    ]
                                    _run_cmd(cmd, cwd=REPO_ROOT, label="Rolling Starts", progress_path=rs_progress)
                                    st.success("Start-date test complete.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    # Time-split (Walkforward)
                    r2 = st.columns([0.70, 0.30], gap="small")
                    with r2[0]:
                        _wf_label = v_wf if (wf_done and str(v_wf).strip() and str(v_wf).strip() != "—") else ("ran" if wf_done else "Not run")
                        st.markdown(
                            _ff_badge_html("Time-split", _wf_label) + " <span style='opacity:.72'>Windowed time-split.</span>",
                            unsafe_allow_html=True,
                        )
                    with r2[1]:
                        if wf_done:
                            st.button("Locked", key=f"suite.wf.done.{pick}", disabled=True, use_container_width=True)
                        else:
                            if st.button("Run", key=f"suite.wf.run.{pick}", type="primary", disabled=(N == 0), use_container_width=True):
                                try:
                                    window_days = 365
                                    step_days = 30
                                    expected_window_bars = int(max(1, round(window_days * bars_per_day)))
                                    min_bars_effective = int(expected_window_bars)
                                    jobs = int(max(1, min(8, (os.cpu_count() or 4))))

                                    wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}"
                                    if wf_out_dir.exists():
                                        wf_out_dir = wf_root / f"wf_win{window_days}_step{step_days}_min{min_bars_effective}_n{N}_{int(time.time())}"
                                    wf_progress = wf_out_dir / "progress" / "walkforward.jsonl"
                                    wf_progress.parent.mkdir(parents=True, exist_ok=True)

                                    cmd = [
                                        PY, "-m", "engine.walkforward",
                                        "--from-run", str(run_dir),
                                        "--top-n", str(N),
                                        "--window-days", str(window_days),
                                        "--step-days", str(step_days),
                                        "--min-bars", str(min_bars_effective),
                                        "--jobs", str(jobs),
                                        "--out", str(wf_out_dir),
                                        "--sort-by", "gates.passed",
                                        "--sort-desc",
                                        "--no-progress",
                                        "--progress-file", str(wf_progress),
                                        "--progress-every", "25",
                                    ]
                                    _run_cmd(cmd, cwd=REPO_ROOT, label="Walkforward", progress_path=wf_progress)
                                    st.success("Time-split test complete.")
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                    st.caption("Checks are locked per run. Start a new run to rerun.")

                    if (rs_done and wf_done):
                        st.caption("All robustness checks are complete for this run.")

                st.divider()
                st.markdown("##### Build sheet")
                st.caption("A mechanics-first summary of what this strategy does (not advice).")

                # Locate artifacts for selected config (replay cache first, then top-k artifacts)
                replay_dir = run_dir / "replay_cache" / str(pick)
                try:
                    art_dir = replay_dir if (replay_dir / "equity_curve.csv").exists() else top_map.get(str(pick), replay_dir)
                except Exception:
                    art_dir = replay_dir

                # Replay artifacts / receipts are stored under replay_cache/<config_id>.
                # We always compute paths even if the directory doesn't exist yet, so we can
                # show a single, consistent "Generate replay artifacts" button when needed.
                eq_path = art_dir / "equity_curve.csv"
                cfg_path = art_dir / "config.json"
                met_path = art_dir / "metrics.json"
                tr_path = art_dir / "trades.csv"
                fi_path = art_dir / "fills.csv"
                ev_path = art_dir / "events.csv"

                replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"
                can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()

                # Replay artifacts controls (single, canonical)


                st.session_state["ui.replay.primary_controls_for"] = str(pick)


                _render_replay_artifacts_controls(


                    run_dir=run_dir,


                    pick=str(pick),


                    replay_dir=replay_dir,


                    has_core_artifacts=bool(eq_path.exists()),


                    can_replay=bool(can_replay),


                    key_prefix="replay.primary",


                    show_when_ready=True,


                )

                if eq_path.exists():


                    # ---------------------------
                    # Strategy build sheet (SPOT)
                    # ---------------------------
                    cfg_obj = _read_json(cfg_path)
                    met_obj = _read_json(met_path)

                    eq_df = _load_csv(eq_path)
                    if eq_df is None:
                        eq_df = pd.DataFrame()
                    tr_df = _load_csv(tr_path)
                    if tr_df is None:
                        tr_df = pd.DataFrame()

                    # Events are optional; if missing we still render the build.
                    ev_path = art_dir / "events.csv"
                    ev_df = _load_csv(ev_path) if ev_path.exists() else pd.DataFrame()
                    # If events.csv is missing, derive a minimal events tape from fills.csv so the price overlay still works.
                    if (ev_df is None or ev_df.empty) and fi_path.exists():
                        try:
                            _fi = _load_csv(fi_path)
                            if _fi is not None and not _fi.empty:
                                _dtc = _pick_col(_fi, ["dt","fill_dt","timestamp","time","ts"])
                                if _dtc is not None:
                                    _fi["_dt"] = pd.to_datetime(_fi[_dtc], errors="coerce", utc=True)
                                    _fi = _fi.dropna(subset=["_dt"]).sort_values("_dt")
                                    _pricec = _pick_col(_fi, ["price","fill_price","px"])
                                    _qtyc = _pick_col(_fi, ["qty","filled_qty","q"])
                                    pos = 0.0
                                    eps = 1e-12
                                    rows = []
                                    for _, rr in _fi.iterrows():
                                        side = str(rr.get("side") or "").strip().lower()
                                        if side not in {"buy","sell"}:
                                            continue
                                        try:
                                            qty = float(rr.get(_qtyc)) if _qtyc else float("nan")
                                        except Exception:
                                            qty = float("nan")
                                        try:
                                            px = float(rr.get(_pricec)) if _pricec else float("nan")
                                        except Exception:
                                            px = float("nan")
                                        before = pos
                                        if side == "buy":
                                            pos = pos + (qty if qty == qty else 0.0)
                                            ev = "ENTRY" if before <= eps else "ADD"
                                        else:
                                            pos = pos - (qty if qty == qty else 0.0)
                                            ot = str(rr.get("order_type") or "").lower()
                                            if "stop" in ot:
                                                ev = "STOP"
                                            else:
                                                ev = "TP" if pos > eps else "EXIT"
                                        rows.append({"dt": rr["_dt"], "event": ev, "side": side, "price": (px if px == px else None), "qty": (qty if qty == qty else None), "reason": (rr.get("order_type") or ""), "detail": ""})
                                    ev_df = pd.DataFrame(rows)
                        except Exception:
                            pass

                    rr_sel = row if isinstance(row, dict) else {}
                    cid_sel = str(pick)
                    label_sel = str(rr_sel.get("label") or rr_sel.get("strategy_label") or rr_sel.get("config_label") or "").strip()
                    if not label_sel:
                        label_sel = cid_sel

                    # --- Config params (spot DCA/swing) ---
                    # Prefer replay artifact config.json; fall back to resolved normalized config if missing.
                    if not isinstance(cfg_obj, dict) or not cfg_obj:
                        cfg_obj = cfg_norm if isinstance(cfg_norm, dict) else {}

                    # Support both wrapped config {"strategy_name","side","params":{...}} and older params-only dict.
                    if isinstance(cfg_obj, dict) and isinstance(cfg_obj.get("params"), dict):
                        params = dict(cfg_obj.get("params") or {})
                    else:
                        params = dict(cfg_obj) if isinstance(cfg_obj, dict) else {}

                    strategy_name = str((cfg_obj.get("strategy_name") if isinstance(cfg_obj, dict) else None) or rr_sel.get("strategy_name") or "strategy").strip()
                    side = str((cfg_obj.get("side") if isinstance(cfg_obj, dict) else None) or rr_sel.get("side") or "long").strip().lower()

                    def _p(key: str, default: Any) -> Any:
                        v = params.get(key, None)
                        return default if v is None else v

                    # Defaults mirror dca_swing.py behavior.
                    deposit_freq = str(_p("deposit_freq", "none") or "none")
                    deposit_amt = float(_p("deposit_amount_usd", 0.0) or 0.0)

                    buy_freq = str(_p("buy_freq", "weekly") or "weekly")
                    buy_amt = float(_p("buy_amount_usd", 0.0) or 0.0)

                    buy_mode = str(_p("buy_mode", "scheduled") or "scheduled").strip().lower()
                    max_buys_per_gate = int(_p("max_buys_per_gate", 0) or 0)

                    buy_filter = str(_p("buy_filter", "none") or "none")
                    entry_logic = params.get("entry_logic") if isinstance(params.get("entry_logic"), dict) else None
                    n_clauses = len((entry_logic or {}).get("clauses") or []) if entry_logic else 0
                    n_regime = len((entry_logic or {}).get("regime") or []) if entry_logic else 0

                    max_alloc_pct = float(_p("max_alloc_pct", 1.0) or 1.0)
                    sl_pct = float(_p("sl_pct", 0.0) or 0.0)
                    trail_pct = float(_p("trail_pct", 0.0) or 0.0)
                    max_hold_bars = int(_p("max_hold_bars", 0) or 0)

                    tp_pct = float(_p("tp_pct", 0.0) or 0.0)
                    tp_sell_fraction = float(_p("tp_sell_fraction", 0.0) or 0.0)
                    reserve_frac_of_proceeds = float(_p("reserve_frac_of_proceeds", _p("reserve_frac", 0.0)) or 0.0)

                    # --- Core stats from selected row (already in artifacts) ---
                    batch_ret = rr_sel.get("performance.twr_total_return", np.nan)
                    batch_dd = rr_sel.get("performance.max_drawdown_equity", np.nan)

                    rs_p10 = rr_sel.get("twr_p10", np.nan)
                    rs_p50 = rr_sel.get("twr_p50", np.nan)
                    rs_fail = _rs_fail_rate(cid_sel)

                    wf_p10 = rr_sel.get("return_p10", np.nan)
                    wf_p50 = rr_sel.get("return_p50", np.nan)
                    wf_neg = rr_sel.get("pct_windows_negative", np.nan)

                    # Stability percentile is computed over visible population earlier (same as the cards)
                    score_pct = _score_pct.get(cid_sel) if isinstance(_score_pct, dict) else None

                    def _clamp01(x: Any) -> float:
                        try:
                            v = float(x)
                            if not math.isfinite(v):
                                return 0.0
                            return float(max(0.0, min(1.0, v)))
                        except Exception:
                            return 0.0

                    # --- Trade stats (derived from trades.csv only) ---
                    trade_count = int(len(tr_df)) if tr_df is not None else 0
                    pnl_col = _pick_col(tr_df, ["net_pnl", "pnl_after_fees", "pnl", "gross_pnl"]) if trade_count else None
                    win_rate = np.nan
                    pf = np.nan
                    if pnl_col:
                        pnl = pd.to_numeric(tr_df[pnl_col], errors="coerce").fillna(0.0).astype(float)
                        win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan
                        wins = float(pnl[pnl > 0].sum())
                        losses = float(pnl[pnl < 0].sum())
                        pf = (wins / abs(losses)) if losses < 0 else (float("inf") if wins > 0 else np.nan)

                    # Holding time
                    med_hold_days = np.nan
                    if trade_count and ("entry_dt" in tr_df.columns) and ("exit_dt" in tr_df.columns):
                        fmt = "%Y-%m-%d %H:%M:%S%z"
                        ent = pd.to_datetime(tr_df["entry_dt"], utc=True, errors="coerce", format=fmt, cache=True)
                        ex = pd.to_datetime(tr_df["exit_dt"], utc=True, errors="coerce", format=fmt, cache=True)
                        dur = (ex - ent).dt.total_seconds() / 86400.0
                        med_hold_days = float(dur.median()) if dur.notna().any() else np.nan

                    # Activity (trades / month) using equity curve date span if present
                    trades_per_month = np.nan
                    if eq_df is not None and not eq_df.empty:
                        xcol_tmp = _pick_col(eq_df, ["dt", "timestamp", "time", "date"])
                        if xcol_tmp:
                            dts = pd.to_datetime(eq_df[xcol_tmp], utc=True, errors="coerce")
                            dts = dts.dropna()
                            if len(dts) >= 2:
                                span_days = max((dts.max() - dts.min()).days, 1)
                                months = span_days / 30.44
                                trades_per_month = float(trade_count / months) if months > 0 else np.nan
                    try:
                        if not math.isfinite(float(trades_per_month)):
                            trades_per_month = float(trade_count)
                    except Exception:
                        trades_per_month = float(trade_count)

                    # DCA intensity from events tape if present
                    adds_per_entry = np.nan
                    entries = 0
                    adds = 0
                    if ev_df is not None and not ev_df.empty and "event" in ev_df.columns:
                        entries = int((ev_df["event"].astype(str) == "ENTRY").sum())
                        adds = int((ev_df["event"].astype(str) == "ADD").sum())
                        adds_per_entry = float(adds / max(entries, 1))

                    # --- Build “traits” (game-style bars; deterministic transforms) ---
                    activity_score = _clamp01(float(trades_per_month) / 20.0)  # 20 trades/mo ~ max
                    patience_score = _clamp01(float(med_hold_days) / 14.0) if math.isfinite(float(med_hold_days)) else 0.0
                    dca_score = _clamp01(float(adds_per_entry) / 3.0) if math.isfinite(float(adds_per_entry)) else 0.0
                    toughness_score = _clamp01(1.0 - (float(batch_dd) / 0.25)) if math.isfinite(float(batch_dd)) else 0.0
                    consistency_score = _clamp01((float(rs_p10) + 0.10) / 0.25) if math.isfinite(float(rs_p10)) else 0.0
                    if rs_fail is not None and math.isfinite(float(rs_fail)):
                        consistency_score = _clamp01(consistency_score * (1.0 - float(rs_fail)))
                    general_score = _clamp01((float(wf_p10) + 0.10) / 0.25) if math.isfinite(float(wf_p10)) else 0.0

                    # “Overall fit” mirrors the Top-10 cards: base by grand verdict + percentile adjustment.
                    grand_v = str(rr_sel.get("grand_verdict") or rr_sel.get("verdict") or rr_sel.get("g.verdict") or "").upper().strip()
                    base = 0.75 if grand_v == "PASS" else (0.55 if grand_v == "WARN" else 0.35)
                    sp = float(score_pct) if (score_pct is not None and math.isfinite(float(score_pct))) else 0.50
                    adj = 0.20 * ((sp - 0.50) * 2.0)
                    confidence = float(max(0.0, min(1.0, base + adj)))

                    # Stage “checks passed”
                    batch_v = str(rr_sel.get("batchq.verdict") or rr_sel.get("batch.verdict") or rr_sel.get("batch_verdict") or "").upper().strip()
                    rs_v = str(rr_sel.get("rsq.verdict") or rr_sel.get("rs.verdict") or rr_sel.get("rs_verdict") or "").upper().strip()
                    wf_v = str(rr_sel.get("wfq.verdict") or rr_sel.get("wf.verdict") or rr_sel.get("wf_verdict") or "").upper().strip()

                    stage_checks = []
                    if batch_v:
                        stage_checks.append(batch_v)
                    if rs_sum is not None and not rs_sum.empty and rs_v:
                        stage_checks.append(rs_v)
                    if wf_sum is not None and not wf_sum.empty and wf_v:
                        stage_checks.append(wf_v)
                    checks_total = len(stage_checks)
                    checks_passed = sum(1 for x in stage_checks if x == "PASS")
                    checks_ratio = (checks_passed / checks_total) if checks_total else 0.0

                    # Grade (same logic as cards)
                    if grand_v == "PASS":
                        if score_pct is not None and score_pct >= 0.90:
                            grade = "S"
                        elif score_pct is not None and score_pct >= 0.75:
                            grade = "A"
                        else:
                            grade = "A-"
                    elif grand_v == "WARN":
                        grade = "B"
                    else:
                        grade = "C"

                    # Top reason (receipt snippet)
                    top_reason = _top_reason_snippet(rr_sel) if "_top_reason_snippet" in globals() or True else ""
                    if not top_reason:
                        top_reason = "—"

                    with st.container():
                        st.caption("Diagnostics derived from saved historical backtest artifacts (spot only, no leverage). Not investment advice.")
                    
                        # Header
                        hL, hR = st.columns([0.78, 0.22])
                        with hL:
                            st.markdown(f"**{label_sel}**")
                            st.caption(f"`{cid_sel}` · `{strategy_name}` · side: `{side}` · market: **spot**")
                        with hR:
                            st.markdown(f"**#{int(rr_sel.get('rank', 0) or 0)}**" if rr_sel.get('rank') else "")
                            st.caption(f"Rank band: **{grade}** (relative to this run)")
                    
                        # Summary for this check (relative)
                        st.markdown(
                            _ff_score_strip_html([
                                ("Filters passed", f"{checks_passed}/{checks_total}" if checks_total else "—", float(checks_ratio or 0.0)),
                                ("Diagnostics score", f"{int(round((confidence or 0.0) * 100))}/100", float(confidence or 0.0)),
                            ]),
                            unsafe_allow_html=True,
                        )
                        # Three-panel layout: config | characteristics | diagnostics
                        c1, c2, c3 = st.columns([0.38, 0.34, 0.28])


                        with c1:
                            st.markdown("**Strategy workflow**")

                            # On/off helpers (explicit is better than implied)
                            dep_off = (str(deposit_freq).strip().lower() in {"none", "off", "0", ""} or float(deposit_amt) <= 0.0)
                            buy_off = (str(buy_freq).strip().lower() in {"none", "off", "0", ""} or float(buy_amt) <= 0.0)

                            # Step 1 — Cashflow
                            if dep_off:
                                cash_desc = "Cash additions: off."
                                cash_chips = ["off"]
                            else:
                                cash_desc = f"Adds cash {str(deposit_freq)}: +${float(deposit_amt):,.0f}."
                                cash_chips = [str(deposit_freq), f"${float(deposit_amt):,.0f}"]

                            # Step 2 — Entry & scaling
                            gate_chip = "no gate"
                            gate_desc = "no entry gate"
                            try:
                                if entry_logic and (n_clauses or n_regime):
                                    gate_chip = "entry rules"
                                    gate_desc = "only when entry rules pass"
                                elif buy_filter and str(buy_filter).lower() != "none":
                                    gate_chip = str(buy_filter)
                                    gate_desc = "only when filter allows"
                            except Exception:
                                pass

                            if buy_off:
                                entry_desc = "Buys: off."
                                entry_chips = ["off"]
                            else:
                                if str(buy_mode).strip().lower() == "signal":
                                    _lim = f" Up to {int(max_buys_per_gate)} buys per signal window." if int(max_buys_per_gate) > 0 else ""
                                    entry_desc = f"While gate is true: buys up to ${float(buy_amt):,.0f} every {str(buy_freq)} (cooldown) ({gate_desc}).{_lim}"
                                    entry_chips = [f"≤ {str(buy_freq)}", f"${float(buy_amt):,.0f}", gate_chip]
                                    if int(max_buys_per_gate) > 0:
                                        entry_chips.append(f"max {int(max_buys_per_gate)}")
                                else:
                                    entry_desc = f"Buys {str(buy_freq)}: ${float(buy_amt):,.0f} on schedule ({gate_desc})."
                                    entry_chips = [str(buy_freq), f"${float(buy_amt):,.0f}", gate_chip]

                            # Step 3 — Position limits
                            alloc_desc = f"Stops buying once invested allocation reaches {max_alloc_pct*100:.0f}% of equity."
                            alloc_chips = [f"max alloc {max_alloc_pct*100:.0f}%"]

                            # Step 4 — Risk controls
                            sl_on = float(sl_pct) > 0
                            trail_on = float(trail_pct) > 0
                            time_on = int(max_hold_bars) > 0

                            risk_bits = []
                            risk_chips = []
                            if sl_on:
                                risk_bits.append(f"Stop-loss {sl_pct*100:.1f}%")
                                risk_chips.append(f"SL {sl_pct*100:.1f}%")
                            else:
                                risk_chips.append("SL off")
                            if trail_on:
                                risk_bits.append(f"Trailing {trail_pct*100:.1f}% from peak")
                                risk_chips.append(f"Trail {trail_pct*100:.1f}%")
                            else:
                                risk_chips.append("Trail off")
                            if time_on:
                                risk_bits.append(f"Time stop {int(max_hold_bars)} bars")
                                risk_chips.append(f"Time {int(max_hold_bars)}")
                            else:
                                risk_chips.append("Time off")

                            risk_desc = "Risk controls: " + (", ".join(risk_bits) + "." if risk_bits else "none.")

                            # Step 5 — Exits
                            tp_on = float(tp_pct) > 0 and float(tp_sell_fraction) > 0
                            exit_chips = []
                            if tp_on:
                                exit_desc = f"Take profit at +{tp_pct*100:.1f}%: sells {tp_sell_fraction*100:.0f}% of position."
                                exit_chips.extend([f"TP {tp_pct*100:.1f}%", f"sell {tp_sell_fraction*100:.0f}%"])
                                if float(reserve_frac_of_proceeds or 0.0) > 0:
                                    exit_desc += f" Reserves {reserve_frac_of_proceeds*100:.0f}% of proceeds as cash."
                                    exit_chips.append(f"reserve {reserve_frac_of_proceeds*100:.0f}%")
                                else:
                                    exit_chips.append("reserve 0%")
                            else:
                                exit_desc = "Take profit: off."
                                exit_chips.append("TP off")

                            steps = [
                                {"title": "Cashflow", "desc": cash_desc, "chips": cash_chips},
                                {"title": "Entry & scaling", "desc": entry_desc, "chips": entry_chips},
                                {"title": "Position limits", "desc": alloc_desc, "chips": alloc_chips},
                                {"title": "Risk controls", "desc": risk_desc, "chips": risk_chips},
                                {"title": "Exits", "desc": exit_desc, "chips": exit_chips},
                            ]
                            st.markdown(_ff_workflow_html(steps), unsafe_allow_html=True)

                            # Decision logic (readable, not JSON)
                            try:
                                show_logic = bool(entry_logic) and (int(n_regime) > 0 or int(n_clauses) > 0)
                            except Exception:
                                show_logic = bool(entry_logic)

                            if show_logic:
                                with st.expander("Entry logic (details)", expanded=False):
                                    if str(buy_mode).strip().lower() == "signal":
                                        st.caption("Buys can fire on any day while the gate is true, but they are spaced by your cooldown (max buy frequency).")
                                    else:
                                        st.caption("Scheduled buy attempts are skipped unless the gate is satisfied.")
                                    reg = (entry_logic or {}).get("regime") or []
                                    clauses = (entry_logic or {}).get("clauses") or []

                                    if reg:
                                        st.markdown("**Regime (must all be true)**")
                                        for c in reg:
                                            if isinstance(c, dict):
                                                st.markdown(f"- `{_human_condition(c)}`")

                                    if not clauses:
                                        st.markdown("**Triggers**")
                                        st.caption("No trigger clauses: once regime is true, buys follow schedule.")
                                    else:
                                        st.markdown("**Triggers (any one group can fire)**")
                                        for i, cl in enumerate(clauses, 1):
                                            st.markdown(f"*Group {i}*")
                                            if not cl:
                                                st.markdown("- `(always)`")
                                            else:
                                                for c in cl:
                                                    if isinstance(c, dict):
                                                        st.markdown(f"- `{_human_condition(c)}`")

                                    if st.checkbox("Show raw entry_logic fields", value=False, key=f"wf_raw_logic_{cid_sel}"):
                                        st.code(json.dumps(entry_logic, indent=2, ensure_ascii=False), language="json")

                            # Raw config mapping (power users)
                            if st.checkbox("Show raw fields used", value=False, key=f"wf_raw_fields_{cid_sel}"):
                                raw = {
                                    "deposit_freq": deposit_freq,
                                    "deposit_amount_usd": float(deposit_amt),
                                    "buy_freq": buy_freq,
                                    "buy_amount_usd": float(buy_amt),
                                    "buy_filter": buy_filter,
                                    "entry_logic": entry_logic,
                                    "max_alloc_pct": float(max_alloc_pct),
                                    "sl_pct": float(sl_pct),
                                    "trail_pct": float(trail_pct),
                                    "max_hold_bars": int(max_hold_bars),
                                    "tp_pct": float(tp_pct),
                                    "tp_sell_fraction": float(tp_sell_fraction),
                                    "reserve_frac_of_proceeds": float(reserve_frac_of_proceeds or 0.0),
                                }
                                st.code(json.dumps(raw, indent=2, ensure_ascii=False), language="json")
                        with c2:
                            st.markdown("**Behavior (what it tends to do)**")
                            tL, tR = st.columns(2, gap="small")
                            with tL:
                                left = [
                                    ("Trade frequency", f"{trades_per_month:.2f}/mo" if math.isfinite(float(trades_per_month)) else "—", float(activity_score)),
                                    ("Median hold", f"{med_hold_days:.2f} days" if math.isfinite(float(med_hold_days)) else "—", float(patience_score)),
                                    ("Adds per entry", f"{adds_per_entry:.2f}" if math.isfinite(float(adds_per_entry)) else "—", float(dca_score)),
                                ]
                                st.markdown(_ff_readouts_html(left), unsafe_allow_html=True)
                            with tR:
                                right = [
                                    ("Drawdown", _fmt_pct(batch_dd) if math.isfinite(float(batch_dd)) else "—", float(toughness_score)),
                                    ("RS stability (p10)", _fmt_pct(rs_p10) if math.isfinite(float(rs_p10)) else "—", float(consistency_score)),
                                    ("WF stability (p10)", _fmt_pct(wf_p10) if math.isfinite(float(wf_p10)) else "—", float(general_score)),
                                ]
                                st.markdown(_ff_readouts_html(right), unsafe_allow_html=True)

                        with c3:
                            st.markdown("**Outcomes (in this run)**")
                            st.markdown(
                                _ff_grid2_html([
                                    ("Stability score", f"{int(round(score_pct * 100))}th pct" if (score_pct is not None and math.isfinite(float(score_pct))) else "—"),
                                    ("Max drawdown", _fmt_pct(batch_dd)),
                                    ("Batch return", _fmt_pct(batch_ret)),
                                    ("Trades", f"{trade_count}"),
                                ]),
                                unsafe_allow_html=True,
                            )

                            with st.expander("Stress test summary", expanded=False):
                                if math.isfinite(float(rs_p10)):
                                    fr = "—" if rs_fail is None else f"{rs_fail * 100:.0f}%"
                                    st.caption(f"RS: p10 {_fmt_pct(rs_p10)} · p50 {_fmt_pct(rs_p50)} · fail {fr} (thr {_fmt_pct(_rs_fail_thr, digits=0)})")
                                else:
                                    st.caption("RS: —")

                                if math.isfinite(float(wf_p10)) or math.isfinite(float(wf_p50)):
                                    neg_txt = f"{float(wf_neg) * 100:.0f}% neg" if math.isfinite(float(wf_neg)) else "neg: —"
                                    st.caption(f"WF: p10 {_fmt_pct(wf_p10)} · p50 {_fmt_pct(wf_p50)} · {neg_txt}")
                                else:
                                    st.caption("WF: missing")

                            # Trade outcome stats (compact)
                            wr_txt = f"{win_rate * 100:.0f}%" if math.isfinite(float(win_rate)) else "—"
                            pf_txt = (f"{pf:.2f}" if (pf is not None and math.isfinite(float(pf)) and pf != float('inf')) else ("∞" if pf == float('inf') else "—"))
                            hold_txt = f"{med_hold_days:.2f} d" if math.isfinite(float(med_hold_days)) else "—"
                            st.caption(f"Outcomes: win {wr_txt} · PF {pf_txt} · median hold {hold_txt}")
                        # Constraint highlight
                        st.markdown(_ff_callout_html("Constraint hit", str(top_reason)), unsafe_allow_html=True)
                    
                        with st.expander("Show full configuration", expanded=False):
                            st.json(params if isinstance(params, dict) else cfg_obj)
                    
                            
                st.divider()

                _tabs_base = ['Batch scan', 'Start-date test', 'Time-split test', 'Receipts', 'Exports']
                _tab_containers = st.tabs(_tabs_base)
                _tabs = list(_tabs_base)
                _tab = dict(zip(_tabs, _tab_containers))

                with _tab.get("Receipts", _tab_containers[0]):
                    st.caption("High-level autopsy for the selected strategy: what it does, what it earned, and the biggest failure modes.")

                    st.markdown("#### Receipts (why the verdict is what it is)")

                    st.markdown("##### What was tested (inputs)")
                    _norm = cfg_norm or {}
                    if _norm:
                        cA, cB = st.columns([1, 1], gap="small")
                        with cA:
                            st.caption("Mechanics snapshot (normalized)")
                            try:
                                st.code(json.dumps(_norm, indent=2, ensure_ascii=False)[:1800] + ("…\n" if len(json.dumps(_norm, indent=2, ensure_ascii=False)) > 1800 else ""))
                            except Exception:
                                st.json(_norm)
                        with cB:
                            st.caption("Quick notes")
                            st.markdown("- This is the **exact** normalized plan settings used for this config.\n- Receipts below explain *why* it passed/warned at each stage.")
                    else:
                        st.info("Normalized config is not available for this run. Receipts still work from computed metrics, but the exact settings snapshot can't be shown here.")
                        st.caption("Tip: this usually means `configs_resolved.jsonl` is missing or was not saved for this run.")


                    def _stage_receipt_block(title: str, q_fn, ans: Dict[str, int]) -> None:
                        out = evaluate_row_with_questions(row, q_fn(), ans)
                        badge = out.verdict
                        st.markdown(f"**{title}: `{badge}`**  —  {out.crits} crit, {out.warns} warn, {out.missing} missing")
                        if out.violations:
                            vdf = pd.DataFrame(out.violations)
                            keep = [c for c in ["severity", "metric", "value", "op", "threshold", "message"] if c in vdf.columns]
                            st.dataframe(vdf[keep], width="stretch", height=240)
                        elif out.missing_metrics:
                            st.caption("No violations, but some metrics were missing for this stage.")
                            st.code(", ".join(out.missing_metrics))
                        else:
                            st.caption("No violations.")

                    _stage_receipt_block("Batch", batch_questions, batch_ans)
                    if rs_sum is not None and not rs_sum.empty:
                        _stage_receipt_block("Rolling Starts", rolling_questions, rs_ans)
                    else:
                        st.info("Start-date test (Rolling Starts) was not run for this strategy in this run.")
                    if wf_sum is not None and not wf_sum.empty:
                        _stage_receipt_block("Walkforward", walkforward_questions, wf_ans)
                    else:
                        st.info("Time-split test (Walkforward) was not run for this strategy in this run.")

                    if cfg_norm:
                        with st.expander("Config (normalized)", expanded=False):
                            st.json(cfg_norm)

                with _tab.get("Batch scan", _tab_containers[0]):

                    if not has_batch:

                        st.info("Batch scan was not run for this candidate in this batch. Turn it on in Run setup (Rolling Starts / Walkforward) and re-run to see this section.")

                    else:
                        st.caption("Fast scan across the whole sample. Look for obvious deal-breakers (fee drag, drawdown, low trade activity).")

                        st.caption("Build sheet is shown above in the **Strategy dossier**.")
                        st.divider()
                        st.markdown("#### Batch replay artifacts")


                        # Price + event timeline (receipts on the tape)
                        ev_path = art_dir / "events.csv"
                        if ev_path.exists() or fi_path.exists():
                            st.markdown("##### Price + event timeline (entries/exits/TPs on the tape)")
                            # Prefer the in-memory events tape (derived from fills.csv earlier) so the overlay still works even if
                            # events.csv is missing in older caches.
                            ev = None
                            try:
                                if "ev_df" in locals() and isinstance(ev_df, pd.DataFrame) and (ev_df is not None) and (not ev_df.empty):
                                    ev = ev_df.copy()
                            except Exception:
                                ev = None
                            # If events are missing/empty, try deriving directly from fills.csv so the overlay can still render.
                            if ev is None:
                                ev = pd.DataFrame()
                            if (ev is None) or ev.empty:
                                try:
                                    fi_path_local = art_dir / "fills.csv"
                                    _fi = _load_csv(fi_path_local) if fi_path_local.exists() else None
                                    if _fi is not None and not _fi.empty:
                                        _dtc = _pick_col(_fi, ["dt","fill_dt","timestamp","time","ts"])
                                        _sidec = _pick_col(_fi, ["side","action"])
                                        _pricec = _pick_col(_fi, ["price","fill_price","px"])
                                        _qtyc = _pick_col(_fi, ["qty","filled_qty","q","base_qty","asset_qty"])
                                        if _dtc is not None and _sidec is not None:
                                            if _dtc == "ts":
                                                s = pd.to_numeric(_fi[_dtc], errors="coerce")
                                                mx = float(s.dropna().max()) if not s.dropna().empty else 0.0
                                                unit = "ms" if mx > 1e12 else "s"
                                                _fi["_dt"] = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
                                            else:
                                                _fi["_dt"] = pd.to_datetime(_fi[_dtc], errors="coerce", utc=True)
                                            _fi = _fi.dropna(subset=["_dt"]).sort_values("_dt")
                                            pos = 0.0
                                            eps = 1e-12
                                            rows = []
                                            for _, r in _fi.iterrows():
                                                side = str(r.get(_sidec) or "").strip().lower()
                                                is_buy = side in {"buy","b","long","entry","open"}
                                                is_sell = side in {"sell","s","exit","close"}
                                                if not is_buy and not is_sell:
                                                    continue
                                                # qty in fills is often signed; use abs for event sizing + pos math
                                                try:
                                                    q = float(r.get(_qtyc)) if _qtyc is not None else float("nan")
                                                except Exception:
                                                    q = float("nan")
                                                q = abs(q) if (q == q) else float("nan")
                                                try:
                                                    px = float(r.get(_pricec)) if _pricec is not None else float("nan")
                                                except Exception:
                                                    px = float("nan")
                                                before = pos
                                                if is_buy:
                                                    pos = pos + (q if (q == q) else 0.0)
                                                    ev_type = "ENTRY" if before <= eps else "ADD"
                                                else:
                                                    pos = max(0.0, pos - (q if (q == q) else 0.0))
                                                    ev_type = "TP" if pos > eps else "EXIT"
                                                rows.append({
                                                    "dt": r.get("_dt"),
                                                    "event": ev_type,
                                                    "side": "buy" if is_buy else "sell",
                                                    "price": None if not (px == px) else float(px),
                                                    "qty": None if not (q == q) else float(q),
                                                    "reason": str(r.get("reason") or r.get("tag") or "").strip(),
                                                    "detail": "",
                                                })
                                            if rows:
                                                ev = pd.DataFrame(rows)
                                except Exception:
                                    pass

                            has_events = (ev is not None) and (not ev.empty)
                            if has_events:
                                if "dt" in ev.columns:
                                    ev["dt"] = pd.to_datetime(ev["dt"], errors="coerce", utc=True)
                                ev = ev.dropna(subset=["dt"]).sort_values("dt")

                            # Enrich STOP/SL markers when possible using fills.csv order_type
                            try:
                                fi_path_local = art_dir / "fills.csv"
                                if fi_path_local.exists():
                                    _fi = _load_csv(fi_path_local)
                                    if _fi is not None and not _fi.empty:
                                        _fi_dt_col = _pick_col(_fi, ["dt","fill_dt","timestamp","time","ts"])
                                        if _fi_dt_col is not None:
                                            _fi["_dt"] = pd.to_datetime(_fi[_fi_dt_col], errors="coerce", utc=True)
                                            _fi = _fi.dropna(subset=["_dt"])
                                            if "order_type" in _fi.columns and "side" in _fi.columns:
                                                _stop = _fi[_fi["side"].astype(str).str.lower().eq("sell") & _fi["order_type"].astype(str).str.lower().str.contains("stop")].copy()
                                                if not _stop.empty:
                                                    if has_events and "event" in ev.columns:
                                                        _stop_ts = set(_stop["_dt"].dt.floor("S"))
                                                        _ev_ts = ev["dt"].dt.floor("S")
                                                        mask = _ev_ts.isin(_stop_ts) & ev.get("side", pd.Series(["" for _ in range(len(ev))])).astype(str).str.lower().eq("sell")
                                                        ev.loc[mask, "event"] = "STOP"
                                                    # Append any stop rows not already present at that second
                                                    have = set(ev["dt"].dt.floor("S")) if has_events and ("dt" in ev.columns) else set()
                                                    _new = _stop if not have else _stop[~_stop["_dt"].dt.floor("S").isin(have)]
                                                    if not _new.empty:
                                                        _tmp = pd.DataFrame({
                                                            "dt": _new["_dt"],
                                                            "event": "STOP",
                                                            "side": "sell",
                                                            "price": pd.to_numeric(_new.get("price"), errors="coerce"),
                                                            "qty": pd.to_numeric(_new.get("qty"), errors="coerce"),
                                                            "reason": "stop",
                                                            "detail": "",
                                                        })
                                                        ev = pd.concat([ev, _tmp], ignore_index=True).dropna(subset=["dt"]).sort_values("dt")
                                                        has_events = True
                            except Exception:
                                pass

                            # Load price series from df_feat (preferred) for the exact run's training tape
                            price = None
                            feat_path = run_dir / "df_feat.parquet"
                            if feat_path.exists():
                                try:
                                    price = pd.read_parquet(feat_path, columns=["dt", "close"])
                                except Exception:
                                    try:
                                        df_tmp = pd.read_parquet(feat_path)
                                        if "dt" in df_tmp.columns and "close" in df_tmp.columns:
                                            price = df_tmp[["dt", "close"]].copy()
                                    except Exception:
                                        price = None

                            if price is not None and not price.empty and "dt" in price.columns:
                                price["dt"] = pd.to_datetime(price["dt"], errors="coerce", utc=True)
                                price = price.dropna(subset=["dt"]).sort_values("dt")
                                # Focus the view around event range (with buffer) when events exist
                                if has_events:
                                    lo = ev["dt"].min() - pd.Timedelta(days=7)
                                    hi = ev["dt"].max() + pd.Timedelta(days=7)
                                    price = price[(price["dt"] >= lo) & (price["dt"] <= hi)]
                                # Downsample for speed
                                max_points = 3500
                                if len(price) > max_points:
                                    idxs = np.linspace(0, len(price) - 1, max_points).astype(int)
                                    price = price.iloc[idxs]
                                if go is not None:
                                    fig_ev = go.Figure()
                                    fig_ev.add_trace(go.Scatter(x=price["dt"], y=price["close"], mode="lines", name="Close"))
                                    if has_events:
                                        show_events = st.multiselect(
                                            "Show events",
                                            ["ENTRY", "ADD", "TP", "STOP", "EXIT"],
                                            default=["ENTRY", "TP", "EXIT"],
                                            key=f"ev_show_{pick}",
                                        )
                                        def _add_ev(etype: str, symbol: str, name: str):
                                            if etype not in show_events:
                                                return
                                            sub = ev[ev.get("event") == etype] if "event" in ev.columns else pd.DataFrame()
                                            if sub is None or sub.empty:
                                                return
                                            y = pd.to_numeric(sub.get("price"), errors="coerce")
                                            text = None
                                            if "reason" in sub.columns or "detail" in sub.columns:
                                                r = sub["reason"].fillna("").astype(str) if "reason" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
                                                d = sub["detail"].fillna("").astype(str) if "detail" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
                                                text = (r + "\n" + d).str.strip()
                                            fig_ev.add_trace(go.Scatter(x=sub["dt"], y=y, mode="markers", name=name, marker=dict(symbol=symbol, size=10),
                                                                       text=text, hovertemplate="%{x}<br>%{y}<br>%{text}<extra></extra>" if text is not None else "%{x}<br>%{y}<extra></extra>"))
                                        _add_ev("ENTRY", "triangle-up", "Entry")
                                        _add_ev("ADD", "circle", "Add (DCA)")
                                        _add_ev("TP", "diamond", "TP / Partial sell")
                                        _add_ev("STOP", "x", "Stop / SL")
                                        _add_ev("EXIT", "triangle-down", "Exit")
                                    else:
                                        st.caption("No events to overlay for this candidate (0 fills or missing event tape).")
                                        st.caption("Artifacts look incomplete? Use the **Replay artifacts** control in the Build sheet above (toggle **Refresh cache**).")

                                        if str(st.session_state.get("ui.replay.primary_controls_for", "")) != str(pick):

                                            replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"

                                            can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()

                                            _render_replay_artifacts_controls(

                                                run_dir=run_dir,

                                                pick=str(pick),

                                                replay_dir=replay_dir,

                                                has_core_artifacts=True,

                                                can_replay=bool(can_replay),

                                                key_prefix="replay.fallback.events",

                                                show_when_ready=True,

                                            )
                                    fig_ev.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Date", yaxis_title="Price",
                                                        legend=dict(orientation="h", yanchor="bottom", y=1.10, xanchor="left", x=0, font=dict(size=12)))
                                    _plotly(fig_ev, key=f"batch_ev_timeline_{pick}")
                                else:
                                    st.info("Plotly is not available; cannot render event timeline chart.")
                            else:
                                st.info("Price tape isn't available for overlay (df_feat.parquet missing or lacks dt/close).")
                            if ev_path.exists():
                                st.download_button("Download events.csv", data=ev_path.read_bytes(), file_name=f"{pick}_events.csv")
                            else:
                                st.download_button("Download events.csv", data=b"", file_name=f"{pick}_events.csv", disabled=True)
                        else:
                            st.caption("No events.csv found for this config yet (replay artifacts need regeneration).")
                    
                            # Allow regeneration even if cached artifacts exist (needed when new artifact types are added).
                            st.caption("events.csv is missing for this config. If you need it, use the **Replay artifacts** control in the Build sheet above (toggle **Refresh cache**).")

                            if str(st.session_state.get("ui.replay.primary_controls_for", "")) != str(pick):

                                replay_script = REPO_ROOT / "tools" / "generate_replay_artifacts.py"

                                can_replay = (run_dir / "configs_resolved.jsonl").exists() and replay_script.exists()

                                _render_replay_artifacts_controls(

                                    run_dir=run_dir,

                                    pick=str(pick),

                                    replay_dir=replay_dir,

                                    has_core_artifacts=True,

                                    can_replay=bool(can_replay),

                                    key_prefix="replay.fallback.events_csv",

                                    show_when_ready=True,

                                )
                    
                    
                    
                    
                        if eq_path.exists():
                            eq = _load_csv(eq_path)
                            if eq is not None and not eq.empty:
                                if "dt" in eq.columns:
                                    eq["dt"] = pd.to_datetime(eq["dt"], errors="coerce", utc=True)
                
                                # Equity vs contributions (+ optional profit) + drawdown
                                if go is not None and make_subplots is not None and "equity" in eq.columns:
                                    try:
                                        eq2 = eq.copy()
                                        eq2["equity"] = pd.to_numeric(eq2["equity"], errors="coerce")
                                        eq2 = eq2.dropna(subset=["equity"])
                                        if not eq2.empty:
                                            xcol = "dt" if "dt" in eq2.columns else None
                
                                            # Prefer columns precomputed in replay artifacts; fall back for legacy caches.
                                            cf = pd.to_numeric(eq2["cashflow"], errors="coerce").fillna(0.0) if "cashflow" in eq2.columns else pd.Series([0.0] * len(eq2), index=eq2.index)
                                            deposits = cf.clip(lower=0.0)
                
                                            if "contrib_total" not in eq2.columns:
                                                contrib0 = float(eq2["equity"].iloc[0])
                                                eq2["contrib_total"] = contrib0 + deposits.cumsum()
                                            else:
                                                eq2["contrib_total"] = pd.to_numeric(eq2["contrib_total"], errors="coerce")
                
                                            if "profit" not in eq2.columns:
                                                eq2["profit"] = eq2["equity"] - eq2["contrib_total"]
                                            else:
                                                eq2["profit"] = pd.to_numeric(eq2["profit"], errors="coerce")
                
                                            peak = eq2["equity"].cummax()
                                            eq2["drawdown"] = (eq2["equity"] / peak) - 1.0
                                            dd_abs = (eq2["equity"] - peak)
                
                                            # Quick human numbers (so deposits can't gaslight you).
                                            last_eq = float(eq2["equity"].iloc[-1])
                                            last_contrib = float(eq2["contrib_total"].iloc[-1])
                                            last_profit = float(eq2["profit"].iloc[-1])
                
                                                                            # Headline numbers (make the cash-in story unambiguous)
                                            initial_cap = float(eq2["contrib_total"].iloc[0]) if "contrib_total" in eq2.columns and len(eq2) else float(eq2["equity"].iloc[0])
                                            deposits_only = max(0.0, last_contrib - initial_cap)
                
                                            c1, c2, c3, c4, c5 = st.columns(5)
                                            with c1:
                                                st.metric("Initial capital", f"{initial_cap:,.2f}")
                                            with c2:
                                                st.metric("Cash in (initial + deposits)", f"{last_contrib:,.2f}")
                                            with c3:
                                                st.metric("Deposits only", f"{deposits_only:,.2f}")
                                            with c4:
                                                st.metric("Net liquidation value", f"{last_eq:,.2f}")
                                            with c5:
                                                st.metric("Profit (equity − cash in)", f"{last_profit:,.2f}")
                                # Unified hover makes the 3-line story legible.
                                            roi = np.where(eq2["contrib_total"].to_numpy() > 0, (eq2["profit"].to_numpy() / eq2["contrib_total"].to_numpy()), np.nan)
                
                                            fig2 = make_subplots(
                                                rows=2,
                                                cols=1,
                                                shared_xaxes=True,
                                                vertical_spacing=0.06,
                                                row_heights=[0.68, 0.32],
                                                subplot_titles=("Equity vs cash in", "Drawdown = drop from running equity peak"),
                                            )
                
                                            fig2.add_trace(
                                                go.Scatter(
                                                    x=eq2[xcol] if xcol else None,
                                                    y=eq2["equity"],
                                                    mode="lines",
                                                    name="Equity (NLV)",
                                                    customdata=np.stack([eq2["contrib_total"].to_numpy(), eq2["profit"].to_numpy(), cf.to_numpy()], axis=1),
                                                    hovertemplate="Equity (NLV): %{y:,.2f}<br>Cash in (to date): %{customdata[0]:,.2f}<br>Profit: %{customdata[1]:,.2f}<br>Cashflow this bar: %{customdata[2]:+,.2f}<extra></extra>",
                                                    line=dict(width=3, color=ACCENT_BLUE),
                                                ),
                                                row=1,
                                                col=1,
                                            )
                
                                            fig2.add_trace(
                                                go.Scatter(
                                                    x=eq2[xcol] if xcol else None,
                                                    y=eq2["contrib_total"],
                                                    mode="lines",
                                                    name="Cash in (to date)",
                                                    customdata=np.stack([cf.to_numpy()], axis=1),
                                                    hovertemplate="Cash in (to date): %{y:,.2f}<br>Cashflow this bar: %{customdata[0]:+,.2f}<extra></extra>",
                                                    line=dict(width=2),
                                                ),
                                                row=1,
                                                col=1,
                                            )
                
                                            fig2.add_trace(
                                                go.Scatter(
                                                    x=eq2[xcol] if xcol else None,
                                                    y=eq2["profit"],
                                                    mode="lines",
                                                    name="Profit (Eq − cash-in)",
                                                    customdata=np.stack([roi, cf.to_numpy()], axis=1),
                                                    hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<br>Cashflow this bar: %{customdata[1]:+,.2f}<extra></extra>",
                                                    line=dict(width=2, dash="dot"),
                                                ),
                                                row=1,
                                                col=1,
                                            )
                
                                            fig2.add_trace(
                                                go.Scatter(
                                                    x=eq2[xcol] if xcol else None,
                                                    y=eq2["drawdown"],
                                                    mode="lines",
                                                    name="Drawdown",
                                                    customdata=np.stack([peak.to_numpy(), dd_abs.to_numpy()], axis=1),
                                                    hovertemplate="Drawdown: %{y:.2%}<br>Peak equity: %{customdata[0]:,.2f}<br>Peak→now: %{customdata[1]:,.2f}<extra></extra>",
                                                    line=dict(width=2, color=FAIL_COLOR),
                                                    fill="tozeroy",
                                                    fillcolor="rgba(255,23,68,0.18)",
                                                ),
                                                row=2,
                                                col=1,
                                            )
                
                                            fig2.update_yaxes(tickformat=".0f", row=1, col=1)
                                            fig2.update_yaxes(tickformat=".0%", row=2, col=1)
                                            fig2.update_layout(hovermode="x unified")
                                            st.markdown("#### Equity, cash in, profit + drawdown")
                                            _style_fig(fig2, title=None)
                                            # Title is rendered by Streamlit header; keep Plotly's top margin for a clean legend.
                                            fig2.update_layout(
                                                title_text="",
                                                margin=dict(t=85),
                                                legend=dict(
                                                    orientation="h",
                                                    yanchor="top",
                                                    y=1.12,
                                                    xanchor="left",
                                                    x=0,
                                                ),
                                            )
                                            st.caption(
                                                "Drawdown = drop from the running equity peak. "
                                                "Cash in = initial capital + deposits (cashflow > 0). "
                                                "Initial capital = cash in at the first bar. "
                                                "Deposits only = cash in − initial capital. "
                                                "Profit = equity − cash in."
                                            )
                                            _plotly(fig2)
                                            with st.expander("Show raw equity curve (table)", expanded=False):
                                                # A simple, no-frills view for sanity checks.
                                                if go is not None:
                                                    fig_raw = go.Figure()
                                                    fig_raw.add_trace(
                                                        go.Scatter(
                                                            x=eq2[xcol] if xcol else None,
                                                            y=eq2["equity"],
                                                            mode="lines",
                                                            name="Equity",
                                                            customdata=np.stack([cf.to_numpy()], axis=1),
                                                            hovertemplate="Equity: %{y:,.2f}<br>Cashflow this bar: %{customdata[0]:+,.2f}<extra></extra>",
                                                            line=dict(width=2, color=ACCENT_BLUE),
                                                        )
                                                    )
                                                    fig_raw.add_trace(
                                                        go.Scatter(
                                                            x=eq2[xcol] if xcol else None,
                                                            y=eq2["contrib_total"],
                                                            mode="lines",
                                                            name="Cash in (to date)",
                                                            hovertemplate="Cash in (to date): %{y:,.2f}<extra></extra>",
                                                            line=dict(width=1),
                                                        )
                                                    )
                                                    fig_raw.add_trace(
                                                        go.Scatter(
                                                            x=eq2[xcol] if xcol else None,
                                                            y=eq2["profit"],
                                                            mode="lines",
                                                            name="Profit",
                                                            customdata=np.stack([roi], axis=1),
                                                            hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<extra></extra>",
                                                            line=dict(width=1, dash="dot"),
                                                        )
                                                    )
                                                    fig_raw.update_layout(hovermode="x unified")
                                                    _style_fig(fig_raw, title="Raw equity curve (no drawdown)")
                                                    _plotly(fig_raw)
                                                    # Profit-only view (separate scale) helps when profit is visually squished.
                                                    fig_profit = go.Figure()
                                                    fig_profit.add_trace(
                                                        go.Scatter(
                                                            x=eq2[xcol] if xcol else None,
                                                            y=eq2["profit"],
                                                            mode="lines",
                                                            name="Profit",
                                                            customdata=np.stack([roi, cf.to_numpy()], axis=1),
                                                            hovertemplate="Profit: %{y:,.2f}<br>ROI on cash in: %{customdata[0]:.2%}<br>Cashflow this bar: %{customdata[1]:+,.2f}<extra></extra>",
                                                            line=dict(width=2, dash="dot"),
                                                        )
                                                    )
                                                    # Mark deposit/withdraw bars for quick intuition.
                                                    if float(np.nanmax(np.abs(cf.to_numpy()))) > 0:
                                                        mask = cf != 0
                                                        x_ev = (eq2.loc[mask, xcol] if xcol else None)
                                                        y_ev = eq2.loc[mask, "profit"]
                                                        fig_profit.add_trace(
                                                            go.Scatter(
                                                                x=x_ev,
                                                                y=y_ev,
                                                                mode="markers",
                                                                name="Cashflow event",
                                                                customdata=np.stack([cf.loc[mask].to_numpy()], axis=1),
                                                                hovertemplate="Cashflow: %{customdata[0]:+,.2f}<extra></extra>",
                                                            )
                                                        )
                                                    fig_profit.update_layout(hovermode="x unified")
                                                    _style_fig(fig_profit, title="Profit only (separate scale)")
                                                    _plotly(fig_profit)
                
                                                # Table is the ultimate audit log.
                                                show_cols = [c for c in ["dt", "equity", "contrib_total", "profit", "drawdown", "cashflow"] if c in eq2.columns]
                                                st.dataframe(eq2[show_cols].tail(500), width="stretch")
                                    except Exception:
                                        pass
                        else:
                            st.info("No equity_curve.csv found in artifacts for this config.")
                        if cfg_path.exists():
                            with st.expander("Config (artifact config.json)", expanded=False):
                                st.json(_read_json(cfg_path))

                        # Trade outcomes (easy read)
                        with st.expander("Trade outcomes (Batch)", expanded=False):
                            if tr_path.exists():
                                tr = _load_csv(tr_path)
                                if tr is not None and not tr.empty:
                                    st.markdown("##### Trade outcomes (Batch)")
                                    pnl_col = _pick_col(tr, ["net_pnl", "gross_pnl", "pnl", "profit"])
                                    if pnl_col and pnl_col in tr.columns and go is not None:
                                        pnl = pd.to_numeric(tr[pnl_col], errors="coerce").dropna()
                                        if len(pnl) > 0:
                                            win_rate = float((pnl > 0).mean())
                                            avg = float(pnl.mean())
                                            med = float(pnl.median())
                                            m1, m2, m3 = st.columns(3)
                                            with m1:
                                                st.metric("Win rate", f"{win_rate*100:.1f}%")
                                            with m2:
                                                st.metric("Avg trade PnL", f"{avg:.2f}")
                                            with m3:
                                                st.metric("Median trade PnL", f"{med:.2f}")

                                            figp = go.Figure(go.Histogram(x=pnl, nbinsx=40, marker=dict(color=ACCENT_BLUE)))
                                            _style_fig(figp, title="Trade PnL distribution")
                                            figp.update_xaxes(title=f"{pnl_col}")
                                            figp.update_yaxes(title="Count")
                                            _plotly(figp)
                            else:
                                st.caption("No trades.csv found for this config.")

                        st.caption("Exit reasons are shown above on the price + event timeline (entries/exits/TPs).")

                        cdl1, cdl2, cdl3, cdl4 = st.columns(4)
                        with cdl1:
                            if met_path.exists():
                                st.download_button("Download metrics.json", data=met_path.read_bytes(), file_name=f"{pick}_metrics.json")
                        with cdl2:
                            if tr_path.exists():
                                st.download_button("Download trades.csv", data=tr_path.read_bytes(), file_name=f"{pick}_trades.csv")
                        with cdl3:
                            if fi_path.exists():
                                st.download_button("Download fills.csv", data=fi_path.read_bytes(), file_name=f"{pick}_fills.csv")
                        with cdl4:
                            if eq_path.exists():
                                st.download_button("Download equity_curve.csv", data=eq_path.read_bytes(), file_name=f"{pick}_equity_curve.csv")
                        if not (met_path.exists() or tr_path.exists() or fi_path.exists() or eq_path.exists() or ev_path.exists() or cfg_path.exists()):
                            st.info("No saved artifacts for this config yet.")
                            st.caption("Generate replay artifacts above to populate downloads and the event overlay.")

                with _tab.get("Start-date test", _tab_containers[0]):

                    if not has_rs:
                        st.info("Start-date test (rolling starts) was not run for this candidate in this batch. Turn it on in Run setup (Rolling Starts / Walkforward) and re-run to see this section.")

                    else:
                        _rsr = ctx.get("rs_sum_row") or {}
                        if _rsr:
                            with st.expander("Batch summary (quick)", expanded=False):
                                s1, s2, s3 = st.columns(3)
                                with s1:
                                    st.metric("Typical return %", _fmt_pct(_rsr.get("twr_p50", _rsr.get("p50", 0.0)), digits=1))
                                with s2:
                                    st.metric("Bad→Good spread", _fmt_pct((_rsr.get("twr_p90", _rsr.get("p90", 0.0)) - _rsr.get("twr_p10", _rsr.get("p10", 0.0))), digits=1))
                                with s3:
                                    _fail = _rsr.get("fail_rate", _rsr.get("disappoint_rate", None))
                                    if _fail is not None:
                                        st.metric("Disappoint rate", _fmt_pct(_fail, digits=1))
                                    else:
                                        st.caption("Disappoint rate: —")
                        st.caption("Rolling Starts: re-run the same strategy from many different start dates. You want robust outcomes across start dates (not one lucky start).")

                        st.markdown("#### Rolling Starts detail")
                        if rs_dir_effective and (rs_dir_effective / "rolling_starts_detail.csv").exists():
                            rs_det = load_rs_detail(run_dir, rs_dir_effective)
                            if rs_det is not None and not rs_det.empty and "config_id" in rs_det.columns:
                                pick_eff = str(pick).strip()
                                # Fallback mapping if user selected a non-canonical id (line_no / row index).
                                try:
                                    if ("config.id" in df2.columns) and (pick_eff.isdigit() or (pick_eff and not pick_eff.startswith("cfg_"))):
                                        if pick_eff.isdigit() and ("config.line_no" in df2.columns):
                                            li = int(pick_eff)
                                            m = df2[df2["config.line_no"].astype(int) == li]
                                            if len(m) == 1:
                                                pick_eff = str(m["config_id"].iloc[0]).strip()
                                        if pick_eff.isdigit():
                                            i = int(pick_eff)
                                            if 1 <= i <= len(df_show):
                                                pick_eff = str(df_show["config_id"].astype(str).iloc[i - 1]).strip()
                                except Exception:
                                    pass
                                pick_eff = _canon_cfg_id(pick_eff)
                                d = rs_det[rs_det["config_id"] == pick_eff].copy()
                                if d.empty:
                                    st.info("No rolling-starts detail rows for this config.")
                                else:
                                    ret_col = _pick_col(d, ["performance.twr_total_return", "twr_total_return", "performance.total_return", "total_return", "window_return", "return", "net_return", "roi"])
                                    if not ret_col or ret_col not in d.columns:
                                        st.info("Rolling-starts detail rows exist, but no return column was found.")
                                    else:
                                        r = pd.to_numeric(d[ret_col], errors="coerce").dropna()
                                        if len(r) == 0:
                                            st.info("Rolling-starts return column is empty after cleaning.")
                                        else:
                                            with st.container():
                                                _k = hashlib.sha1(pick_eff.encode("utf-8")).hexdigest()[:10]
                                                tol = st.slider(
                                                    "Disappoint cutoff (return)",
                                                    min_value=-0.50,
                                                    max_value=0.50,
                                                    value=-0.10,
                                                    step=0.01,
                                                    format="%.2f",
                                                    key=f"rs.tol.{_k}",
                                                )
                                                p10 = float(np.nanpercentile(r, 10))
                                                p50 = float(np.nanpercentile(r, 50))
                                                p90 = float(np.nanpercentile(r, 90))
                                                disappoint = float((r < float(tol)).mean())
                                                rng = float(p90 - p10)

                                                st.caption("Summary for this check")
                                                st.markdown(
                                                    _summary_strip_html(
                                                        [
                                                            ("Bad-case (p10)", _fmt_pct(p10), False),
                                                            ("Typical (median)", _fmt_pct(p50), True),
                                                            ("Good-case (p90)", _fmt_pct(p90), False),
                                                            (f"Disappoint rate (≤ { _fmt_pct(tol, digits=1) })", _fmt_pct(disappoint, digits=1), True),
                                                        ],
                                                        chips=[f"Range (p90–p10): { _fmt_pct(rng, digits=1) }"],
                                                    ),
                                                    unsafe_allow_html=True,
                                                )
                                                if go is not None:
                                                    try:
                                                        near = _rs_zone_width(tol)
                                                        st.caption("Bands: red = below cutoff, yellow = near-miss, green = comfortably above cutoff.")
                                                        fig = _rs_violin_fig(d[ret_col], tol=tol, near=near)
                                                        _plotly(fig)
                                                        if "start_dt" in d.columns:
                                                            fig2 = _rs_timeline_fig(d, ret_col=ret_col, tol=tol, near=near)
                                                            _plotly(fig2)
                                                    except Exception:
                                                        pass
                                            show_cols = []
                                            for c in ["start_dt", "end_dt", ret_col, "trades", "bars", "seed"]:
                                                if c in d.columns:
                                                    show_cols.append(c)
                                            with st.expander("Show raw rolling-starts rows (advanced)", expanded=False):
                                                st.dataframe(d[show_cols].tail(500) if show_cols else d.tail(500), width="stretch")
                                                st.download_button(
                                                    "Download rolling_starts_detail.csv (full)",
                                                    data=(rs_dir_effective / "rolling_starts_detail.csv").read_bytes(),
                                                    file_name=f"{selected_run_name}_rolling_starts_detail.csv",
                                                )
                            else:
                                st.info("Rolling starts detail file exists but appears empty.")
                        else:
                            st.info("Rolling starts evidence not available for this run (run it from Build & Run).")
                with _tab.get("Time-split test", _tab_containers[0]):

                    if not has_wf:
                        st.info("Time-split test was not run for this candidate in this batch. Turn it on in Run setup (Rolling Starts / Walkforward) and re-run to see this section.")

                    else:
                        _wfr = ctx.get("wf_sum_row") or {}
                        if _wfr:
                            with st.expander("Batch summary (quick)", expanded=False):
                                s1, s2, s3 = st.columns(3)
                                with s1:
                                    st.metric("Typical return %", _fmt_pct(_wfr.get("return_p50", _wfr.get("p50", 0.0)), digits=1))
                                with s2:
                                    st.metric("Bad→Good spread", _fmt_pct((_wfr.get("return_p90", _wfr.get("p90", 0.0)) - _wfr.get("return_p10", _wfr.get("p10", 0.0))), digits=1))
                                with s3:
                                    _neg = _wfr.get("neg_rate", _wfr.get("fail_rate", None))
                                    if _neg is not None:
                                        st.metric("Negative windows", _fmt_pct(_neg, digits=1))
                                    else:
                                        st.caption("Negative windows: —")
                        st.caption("Walkforward: train/test through time windows. You want consistency across windows — not one regime doing all the work.")

                        st.markdown("#### Walkforward detail")
                        if wf_dir_effective and (wf_dir_effective / "wf_results.csv").exists():
                            wf_rows = load_wf_results(wf_dir_effective)
                            if wf_rows is not None and not wf_rows.empty and "config_id" in wf_rows.columns:
                                pick_eff = str(pick).strip()
                                try:
                                    if ("config.id" in df2.columns) and (pick_eff.isdigit() or (pick_eff and not pick_eff.startswith("cfg_"))):
                                        if pick_eff.isdigit() and ("config.line_no" in df2.columns):
                                            li = int(pick_eff)
                                            m = df2[df2["config.line_no"].astype(int) == li]
                                            if len(m) == 1:
                                                pick_eff = str(m["config_id"].iloc[0]).strip()
                                        if pick_eff.isdigit():
                                            i = int(pick_eff)
                                            if 1 <= i <= len(df_show):
                                                pick_eff = str(df_show["config_id"].astype(str).iloc[i - 1]).strip()
                                except Exception:
                                    pass
                                pick_eff = _canon_cfg_id(pick_eff)
                                d = wf_rows[wf_rows["config_id"] == pick_eff].copy()
                                if d.empty:
                                    st.info("No Walkforward detail rows for this config.")
                                else:
                                    ret_col = _pick_col(d, ["window_return", "test_return", "return", "ret"])
                                    if not ret_col or ret_col not in d.columns:
                                        st.info("Walkforward rows exist, but no window return column was found.")
                                    else:
                                        if "window_idx" not in d.columns:
                                            d["window_idx"] = np.arange(len(d))
                                        d["window_idx"] = pd.to_numeric(d["window_idx"], errors="coerce")
                                        d["window_return"] = pd.to_numeric(d[ret_col], errors="coerce")
                                        d = d.dropna(subset=["window_idx", "window_return"])
                                        d["window_idx"] = d["window_idx"].astype(int)
                                        with st.container():
                                            rvals = d["window_return"].astype(float)
                                            p50 = float(np.nanpercentile(rvals, 50))
                                            neg_rate = float((rvals < 0.0).mean())
                                            worst = float(np.nanmin(rvals.values))
                                            best = float(np.nanmax(rvals.values))

                                            st.caption("Summary for this check")
                                            st.markdown(
                                                _summary_strip_html(
                                                    [
                                                        ("Typical window (median)", _fmt_pct(p50), True),
                                                        ("% negative windows", _fmt_pct(neg_rate, digits=1), True),
                                                        ("Worst window (min)", _fmt_pct(worst), False),
                                                        ("Best window (max)", _fmt_pct(best), False),
                                                    ]
                                                ),
                                                unsafe_allow_html=True,
                                            )
                                            if go is not None:
                                                try:
                                                    d2 = d.sort_values("window_idx")
                                                    hover = None
                                                    if ("window_start_dt" in d2.columns) and ("window_end_dt" in d2.columns):
                                                        a = d2["window_start_dt"].astype(str).str.slice(0, 10)
                                                        b = d2["window_end_dt"].astype(str).str.slice(0, 10)
                                                        hover = a + " → " + b
                                                    fig_wf = go.Figure()
                                                    _vals = pd.to_numeric(d2["window_return"], errors="coerce").astype(float)
                                                    if _vals.notna().any():
                                                        _scale = float(np.nanpercentile(np.abs(_vals.values), 90) or 0.0)
                                                    else:
                                                        _scale = 0.0
                                                    _wf_tol = 0.0
                                                    _wf_margin = max(0.001, _scale * 0.15)

                                                    def _wf_band(v: float) -> str:
                                                        if pd.isna(v):
                                                            return "rgba(107,114,128,0.25)"
                                                        if v < _wf_tol - _wf_margin:
                                                            return "rgba(239,68,68,0.85)"   # red
                                                        if v < _wf_tol + _wf_margin:
                                                            return "rgba(234,179,8,0.85)"  # amber
                                                        return "rgba(34,197,94,0.85)"     # green

                                                    _colors = [_wf_band(v) for v in _vals.values.tolist()]
                                                    _hover = hover
                                                    if _hover is None:
                                                        _hover = [
                                                            (f"Window {int(w)}<br>Return: {v:.4%}" if pd.notna(v) else f"Window {int(w)}<br>Return: —")
                                                            for w, v in zip(d2["window_idx"].values.tolist(), _vals.values.tolist())
                                                        ]

                                                    fig_wf.add_trace(go.Bar(
                                                        x=d2["window_idx"],
                                                        y=_vals,
                                                        marker_color=_colors,
                                                        name="Window return",
                                                        hovertext=_hover,
                                                        hovertemplate="%{hovertext}<extra></extra>",
                                                    ))

                                                    # Rolling average overlay to show drift (keeps detail without clutter)
                                                    if len(d2) >= 2:
                                                        _ma = pd.Series(_vals).rolling(3, min_periods=1).mean()
                                                        fig_wf.add_trace(go.Scatter(
                                                            x=d2["window_idx"],
                                                            y=_ma,
                                                            mode="lines",
                                                            line=dict(color="rgba(17,24,39,0.35)", width=2),
                                                            name="3-window avg",
                                                            hovertemplate="3-window avg: %{y:.4%}<extra></extra>",
                                                        ))

                                                    # Background bands (bad / near-miss / good) so the chart reads at a glance
                                                    if _vals.notna().any():
                                                        _ymin = float(_vals.min())
                                                        _ymax = float(_vals.max())
                                                    else:
                                                        _ymin, _ymax = -0.01, 0.01
                                                    _pad = max((_ymax - _ymin) * 0.08, 0.002)
                                                    _y0 = _ymin - _pad
                                                    _y1 = _ymax + _pad
                                                    fig_wf.add_hrect(y0=_y0, y1=_wf_tol - _wf_margin, fillcolor="rgba(239,68,68,0.08)", line_width=0)
                                                    fig_wf.add_hrect(y0=_wf_tol - _wf_margin, y1=_wf_tol + _wf_margin, fillcolor="rgba(234,179,8,0.06)", line_width=0)
                                                    fig_wf.add_hrect(y0=_wf_tol + _wf_margin, y1=_y1, fillcolor="rgba(34,197,94,0.06)", line_width=0)

                                                    fig_wf.add_hline(y=_wf_tol, line_dash="dash", line_color="rgba(107,114,128,0.6)")
                                                    _style_fig(fig_wf, title="Walkforward: window returns")
                                                    fig_wf.update_yaxes(tickformat=".2%", title="Window return")
                                                    fig_wf.update_xaxes(title="Window index")
                                                    _plotly(fig_wf)
                                                except Exception:
                                                    pass
                                        show_cols = []
                                        for c in ["window_idx", "window_start_dt", "window_end_dt", ret_col, "train_start_dt", "train_end_dt", "test_start_dt", "test_end_dt"]:
                                            if c in wf_rows.columns:
                                                show_cols.append(c)
                                        st.dataframe(d.sort_values("window_idx")[show_cols].tail(500) if show_cols else d.sort_values("window_idx").tail(500), width="stretch")
                                        st.download_button(
                                            "Download wf_results.csv (full)",
                                            data=(wf_dir_effective / "wf_results.csv").read_bytes(),
                                            file_name=f"{selected_run_name}_wf_results.csv",
                                        )
                            else:
                                st.info("Walkforward results file exists but appears empty.")
                        else:
                            st.info("Walkforward evidence not available for this run (run it from Build & Run).")
                with _tab.get("Exports", _tab_containers[-1]):
                    st.caption("Export artifacts for sharing or replay. This app is tooling — exporting is how you keep work portable.")

                    st.markdown("#### Exports")

                    st.markdown("##### Artifacts for selected strategy")
                    if not (art_dir and art_dir.exists()):
                        st.info("No replay artifacts were saved for this strategy (no replay_cache/top artifacts folder found).")
                        st.caption("Tip: enable replay/artifact saving for shortlisted candidates, then rerun.")
                    else:
                        core = [
                            ("equity_curve.csv", "Equity curve (CSV)"),
                            ("trades.csv", "Trades (CSV)"),
                            ("fills.csv", "Fills (CSV)"),
                            ("metrics.json", "Metrics (JSON)"),
                            ("config.json", "Config (JSON)"),
                        ]
                        cols = st.columns(3)
                        for i, (fname, label) in enumerate(core):
                            fp = art_dir / fname
                            with cols[i % 3]:
                                if fp.exists():
                                    st.download_button(
                                        label,
                                        data=fp.read_bytes(),
                                        file_name=f"{selected_run_name}_{pick}_{fname}",
                                        key=f"export.core.{fname}.{pick}",
                                    )
                                else:
                                    st.caption(f"Missing: {fname}")

                    st.markdown("##### Run-level exports")
                    if run_dir and run_dir.exists():
                        run_files = [
                            ("results.csv", "Batch results (CSV)"),
                            ("results_full.csv", "Batch results full (CSV)"),
                            ("configs_resolved.jsonl", "Configs resolved (JSONL)"),
                        ]
                        cols2 = st.columns(3)
                        for i, (fname, label) in enumerate(run_files):
                            fp = run_dir / fname
                            with cols2[i % 3]:
                                if fp.exists():
                                    st.download_button(
                                        label,
                                        data=fp.read_bytes(),
                                        file_name=f"{selected_run_name}_{fname}",
                                        key=f"export.run.{fname}.{selected_run_name}",
                                    )
                                else:
                                    st.caption(f"Missing: {fname}")
                    else:
                        st.caption("Run folder not available in this context.")