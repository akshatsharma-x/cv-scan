"""
charts.py — Plotly figure builders for the dashboard.

Data-viz rules applied throughout:
  • Identity colours are the validated categorical slots 1–3
    (keyword = blue, semantic = orange, hybrid = aqua), light/dark steps.
  • Magnitude uses one sequential hue (blue); the low end recedes into the surface.
  • "This candidate vs. the pool" uses emphasis: the pool is neutral grey.
  • Thin marks: ≤ 24 px bars with 4 px rounded data-ends, 2 px lines, ≥ 8 px
    markers with a 2 px surface ring; hairline, solid, recessive grids.
  • A legend for ≥ 2 series, selective direct labels, hover on every mark.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ui import SERIES

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
SURFACE = {"dark": "#0b0d14", "light": "#f6f7fb"}
INK = {"dark": "#e7e9f2", "light": "#161927"}
INK2 = {"dark": "#a3a9be", "light": "#585e73"}
GRID = {"dark": "rgba(255,255,255,0.07)", "light": "rgba(22,25,39,0.08)"}
AXIS = {"dark": "#474b5c", "light": "#c2c5cf"}
NEUTRAL = {"dark": "#454b60", "light": "#c9ccd6"}
SEQUENTIAL = {
    "light": ["#e4effd", "#b6d5f8", "#84b9f3", "#519ced", "#257de0", "#1b64be", "#0e3d7e"],
    "dark": ["#122546", "#0f4a96", "#1b64be", "#257de0", "#519ced", "#84b9f3", "#cce3fd"],
}
METHOD_SERIES = {"TF-IDF": "keyword", "BERT": "semantic", "Hybrid": "hybrid",
                 "Keyword": "keyword", "Semantic": "semantic"}
PLOT_CONFIG = {"displayModeBar": False, "responsive": True}


def _m(dark: bool) -> str:
    return "dark" if dark else "light"


def color(kind: str, dark: bool) -> str:
    return SERIES[METHOD_SERIES.get(kind, kind)][_m(dark)]


def _scale(dark: bool):
    steps = SEQUENTIAL[_m(dark)]
    return [[i / (len(steps) - 1), c] for i, c in enumerate(steps)]


def _base(fig: go.Figure, dark: bool, height: int, legend: bool = True, **layout) -> go.Figure:
    m = _m(dark)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=18, t=40 if legend else 14, b=8),
        font=dict(family=FONT, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        barcornerradius=4,
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID[m], gridwidth=1, zeroline=False, linecolor=AXIS[m], ticks="", automargin=True)
    fig.update_yaxes(gridcolor=GRID[m], gridwidth=1, zeroline=False, linecolor=AXIS[m], ticks="", automargin=True)
    return fig


# ── Candidate analysis ─────────────────────────────────────────────────────
def matched_terms(matched: list[tuple[str, float]], dark: bool, top: int = 12) -> go.Figure:
    rows = matched[:top][::-1]
    terms = [t for t, _ in rows]
    shares = [s for _, s in rows]
    fig = go.Figure(
        go.Bar(
            x=shares, y=terms, orientation="h", marker=dict(color=color("keyword", dark)),
            text=[f"{s:.0%}" for s in shares], textposition="outside", cliponaxis=False,
            textfont=dict(color=INK2[_m(dark)]),
            hovertemplate="<b>%{x:.1%}</b> of the keyword score<br>%{y}<extra></extra>",
        )
    )
    _base(fig, dark, height=40 + 30 * max(len(rows), 1), legend=False, bargap=0.38)
    fig.update_xaxes(tickformat=".0%", range=[0, max(shares + [0.05]) * 1.22], showgrid=True)
    fig.update_yaxes(showgrid=False, tickfont=dict(size=12.5))
    return fig


def score_distribution(pool_scores, candidates, selected: str, series: str, dark: bool) -> go.Figure:
    """Histogram of the historical pool (neutral) with candidates highlighted."""
    m = _m(dark)
    fig = go.Figure(
        go.Histogram(
            x=np.asarray(pool_scores) * 100, xbins=dict(start=0, end=100, size=2.5),
            marker=dict(color=NEUTRAL[m]), name=f"Historical pool · {len(pool_scores):,} resumes",
            hovertemplate="%{y} resumes scored %{x}<extra></extra>",
        )
    )
    others = [(n, s) for n, s in candidates if n != selected]
    if others:
        fig.add_trace(
            go.Scatter(
                x=[s * 100 for _, s in others], y=[0] * len(others), mode="markers", name="Other candidates",
                marker=dict(symbol="triangle-up", size=12, color=INK2[m], line=dict(width=2, color=SURFACE[m])),
                text=[n for n, _ in others], hovertemplate="<b>%{x:.1f}%</b><br>%{text}<extra></extra>",
                cliponaxis=False,
            )
        )
    chosen = next((s for n, s in candidates if n == selected), None)
    if chosen is not None:
        top_pct = 100 - float((np.asarray(pool_scores) < chosen).mean() * 100)
        fig.add_vline(x=chosen * 100, line=dict(color=color(series, dark), width=2.5))
        fig.add_trace(
            go.Scatter(
                x=[chosen * 100], y=[0], mode="markers", name=selected if len(selected) < 34 else selected[:32] + "…",
                marker=dict(symbol="diamond", size=14, color=color(series, dark), line=dict(width=2, color=SURFACE[m])),
                hovertemplate=f"<b>%{{x:.1f}}%</b><br>{selected}<extra></extra>", cliponaxis=False,
            )
        )
        fig.add_annotation(
            x=chosen * 100, y=1, yref="paper", yanchor="bottom", showarrow=False,
            text=f"<b>{chosen * 100:.0f}%</b> · top {max(top_pct, 0.1):.1f}%", font=dict(color=INK[m], size=12),
            bgcolor=SURFACE[m], borderpad=3, xanchor="left" if chosen < 0.75 else "right", xshift=6 if chosen < 0.75 else -6,
        )
    _base(fig, dark, height=330, bargap=0.1)
    fig.update_xaxes(title="Score (%)", range=[0, 101], dtick=10)
    fig.update_yaxes(title="Resumes", rangemode="tozero")
    return fig


def _short(category: str) -> str:
    special = {"INFORMATION-TECHNOLOGY": "IT", "BUSINESS-DEVELOPMENT": "Business dev.", "PUBLIC-RELATIONS": "Public relations",
               "HR": "HR", "BPO": "BPO", "DIGITAL-MEDIA": "Digital media"}
    return special.get(category, category.replace("-", " ").capitalize())


def semantic_map(engine, run, selected_idx: int, dark: bool, dims: int = 2) -> go.Figure:
    m = _m(dark)
    proj = engine.pool_projection
    jd = engine.pca.transform(run.jd_embedding[None, :])[0]
    cand = engine.pca.transform(np.vstack([c.embedding for c in run.candidates]))
    names = [c.name for c in run.candidates]
    custom = np.stack([engine.pool["ID"].astype(str), engine.pool["Category"].map(_short), run.pool_score * 100], axis=1)
    var = engine.pca.explained_variance_ratio_ * 100
    pool_hover = "Resume %{customdata[0]}<br>%{customdata[1]}<br>Match <b>%{customdata[2]:.0f}%</b><extra>Historical pool</extra>"
    sel = np.zeros(len(names), bool)
    sel[selected_idx] = True

    if dims == 3:
        fig = go.Figure(
            go.Scatter3d(
                x=proj[:, 0], y=proj[:, 1], z=proj[:, 2], mode="markers", name="Historical pool",
                marker=dict(size=2.6, color=run.pool_score, colorscale=_scale(dark), cmin=0, cmax=1, opacity=0.85,
                            colorbar=dict(title="Match", tickformat=".0%", thickness=10, len=0.55, x=1.0)),
                customdata=custom, hovertemplate=pool_hover,
            )
        )
        fig.add_trace(go.Scatter3d(x=[jd[0]], y=[jd[1]], z=[jd[2]], mode="markers+text", name="Job description",
                                   text=["Job description"], textposition="top center",
                                   marker=dict(size=9, symbol="diamond", color=INK[m], line=dict(width=2, color=SURFACE[m])),
                                   hovertemplate="<b>Job description</b><extra></extra>"))
        fig.add_trace(go.Scatter3d(x=cand[:, 0], y=cand[:, 1], z=cand[:, 2], mode="markers+text", name="Candidates",
                                   text=[n if s else "" for n, s in zip(names, sel)], textposition="top center",
                                   marker=dict(size=[8 if s else 5 for s in sel], color=color("semantic", dark),
                                               line=dict(width=1, color=SURFACE[m])),
                                   hovertext=names, hovertemplate="<b>%{hovertext}</b><extra>Candidate</extra>"))
        axis = dict(showbackground=False, gridcolor=GRID[m], zeroline=False, showticklabels=False, title="")
        _base(fig, dark, height=560)
        fig.update_layout(scene=dict(xaxis=dict(axis, title=f"PC1 · {var[0]:.0f}%"), yaxis=dict(axis, title=f"PC2 · {var[1]:.0f}%"),
                                     zaxis=dict(axis, title=f"PC3 · {var[2]:.0f}%"), aspectmode="cube"),
                          margin=dict(l=0, r=0, t=40, b=0))
        return fig

    fig = go.Figure(
        go.Scattergl(
            x=proj[:, 0], y=proj[:, 1], mode="markers", name="Historical pool",
            marker=dict(size=6, color=run.pool_score, colorscale=_scale(dark), cmin=0, cmax=1,
                        colorbar=dict(title="Match", tickformat=".0%", thickness=10, len=0.6)),
            customdata=custom, hovertemplate=pool_hover,
        )
    )
    fig.add_trace(go.Scatter(x=[jd[0]], y=[jd[1]], mode="markers", name="Job description",
                             marker=dict(symbol="star", size=22, color=INK[m], line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="<b>Job description</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=cand[:, 0], y=cand[:, 1], mode="markers", name="Candidates", hovertext=names,
                             marker=dict(symbol="diamond", size=[19 if s else 13 for s in sel], color=color("semantic", dark),
                                         line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="<b>%{hovertext}</b><extra>Candidate</extra>"))
    fig.add_annotation(x=cand[selected_idx, 0], y=cand[selected_idx, 1], text=f"<b>{names[selected_idx]}</b>",
                       showarrow=True, arrowhead=0, arrowwidth=1, arrowcolor=INK2[m], ax=40, ay=-34,
                       font=dict(color=INK[m], size=12), bgcolor=SURFACE[m], borderpad=3)
    _base(fig, dark, height=500)
    fig.update_xaxes(title=f"PC1 · {var[0]:.0f}% of variance", showticklabels=False, showgrid=False)
    fig.update_yaxes(title=f"PC2 · {var[1]:.0f}% of variance", showticklabels=False, showgrid=False)
    return fig


# ── Evaluation ─────────────────────────────────────────────────────────────
def grouped_bars(values: dict[str, dict[str, float]], metrics: list[str], dark: bool, label: str = "Hybrid",
                 height: int = 360, y_range=(0, 1.05)) -> go.Figure:
    fig = go.Figure()
    for method, row in values.items():
        is_label = method == label
        fig.add_trace(
            go.Bar(
                x=metrics, y=[row[k] for k in metrics], name=method, width=0.16,
                marker=dict(color=color(method, dark)),
                text=[f"{row[k]:.2f}" for k in metrics] if is_label else None,
                textposition="outside", textfont=dict(color=INK2[_m(dark)], size=11), cliponaxis=False,
                hovertemplate=f"<b>%{{y:.3f}}</b> {method}<br>%{{x}}<extra></extra>",
            )
        )
    _base(fig, dark, height=height, barmode="group", bargap=0.3, bargroupgap=0.12)
    fig.update_yaxes(range=list(y_range), tickformat=".1f")
    fig.update_xaxes(showgrid=False)
    return fig


def learning_curve(per_split: dict, metric: str, dark: bool) -> go.Figure:
    m = _m(dark)
    fig = go.Figure()
    for method, splits in per_split.items():
        xs = list(splits.keys())
        fig.add_trace(
            go.Scatter(
                x=xs, y=[splits[s][metric] for s in xs], name=method, mode="lines+markers",
                line=dict(color=color(method, dark), width=2),
                marker=dict(size=9, color=color(method, dark), line=dict(width=2, color=SURFACE[m])),
                hovertemplate=f"{method}: <b>%{{y:.3f}}</b><extra></extra>",
            )
        )
    _base(fig, dark, height=340, hovermode="x unified")
    fig.update_xaxes(title="Training split", showgrid=False, type="category")
    fig.update_yaxes(title=metric, tickformat=".2f")
    return fig


def confusion_heatmap(cm: list[list[int]], classes: list[str], dark: bool) -> go.Figure:
    cm = np.asarray(cm, dtype=float)
    row_share = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm), where=cm.sum(axis=1, keepdims=True) > 0)
    labels = [_short(c) for c in classes]
    fig = go.Figure(
        go.Heatmap(
            z=row_share, x=labels, y=labels, customdata=cm.astype(int), colorscale=_scale(dark), zmin=0, zmax=1,
            xgap=2, ygap=2, colorbar=dict(title="Share of row", tickformat=".0%", thickness=10, len=0.7),
            hovertemplate="Actual <b>%{y}</b> → predicted <b>%{x}</b><br>%{customdata} resumes · %{z:.0%} of row<extra></extra>",
        )
    )
    _base(fig, dark, height=640, legend=False)
    fig.update_xaxes(title="Predicted category", tickangle=-45, showgrid=False, tickfont=dict(size=11))
    fig.update_yaxes(title="Actual category", autorange="reversed", showgrid=False, tickfont=dict(size=11))
    return fig


def c_sweep(sweep: dict, method: str, dark: bool, chosen_c: float = 1.0) -> go.Figure:
    m = _m(dark)
    Cs = sweep["C"]
    val = np.asarray(sweep[method]["val"]["mean"])
    std = np.asarray(sweep[method]["val"]["std"])
    train = np.asarray(sweep[method]["train"]["mean"])
    c = color(method, dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Cs + Cs[::-1], y=list(val + std) + list((val - std)[::-1]), fill="toself", mode="lines",
                             fillcolor=_rgba(c, 0.14), line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=Cs, y=train, name="Training accuracy", mode="lines+markers",
                             line=dict(color=NEUTRAL[m] if dark else "#9aa0b0", width=2),
                             marker=dict(size=8, color=NEUTRAL[m] if dark else "#9aa0b0", line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="Training: <b>%{y:.3f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=Cs, y=val, name="Validation accuracy (5-fold CV)", mode="lines+markers",
                             line=dict(color=c, width=2), marker=dict(size=9, color=c, line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="Validation: <b>%{y:.3f}</b><extra></extra>"))
    fig.add_vline(x=chosen_c, line=dict(color=INK2[m], width=1))
    fig.add_annotation(x=np.log10(chosen_c), y=0.02, yref="paper", yanchor="bottom", xanchor="left", xshift=5,
                       text=f"C used = {chosen_c:g}", showarrow=False, font=dict(size=11, color=INK2[m]))
    _base(fig, dark, height=360, hovermode="x unified")
    fig.update_xaxes(type="log", title="SVM regularisation C (log scale)", tickvals=Cs, ticktext=[f"{v:g}" for v in Cs], showgrid=False)
    fig.update_yaxes(title="Accuracy", tickformat=".0%", range=[max(0, float(min(val.min(), train.min())) - 0.06), 1.02])
    return fig


def weight_sweep(sweep: dict, dark: bool, current: float) -> go.Figure:
    m = _m(dark)
    w = sweep["weights"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w, y=sweep["P@5"], name="Precision@5", mode="lines+markers",
                             line=dict(color="#9aa0b0" if not dark else NEUTRAL[m], width=2),
                             marker=dict(size=8, color="#9aa0b0" if not dark else NEUTRAL[m], line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="P@5: <b>%{y:.3f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=w, y=sweep["nDCG@10"], name="nDCG@10", mode="lines+markers",
                             line=dict(color=color("hybrid", dark), width=2),
                             marker=dict(size=9, color=color("hybrid", dark), line=dict(width=2, color=SURFACE[m])),
                             hovertemplate="nDCG@10: <b>%{y:.3f}</b><extra></extra>"))
    fig.add_vline(x=current, line=dict(color=INK2[m], width=1))
    fig.add_annotation(x=current, y=0.02, yref="paper", yanchor="bottom", xanchor="left", xshift=5, showarrow=False,
                       text=f"default = {current:.0%} keyword", font=dict(size=11, color=INK2[m]))
    _base(fig, dark, height=340, hovermode="x unified")
    fig.update_xaxes(title="Keyword weight w   (hybrid = w·keyword + (1 − w)·semantic)", tickformat=".0%", dtick=0.1, showgrid=False)
    fig.update_yaxes(title="Score", tickformat=".2f")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
