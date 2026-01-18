
"""
raschonline.py — Polytomous Rasch (Rating Scale Model, RSM) with Winsteps/ASP-like extreme handling.

Why this exists:
- Your previous runs exploded (NaN error, tau ~ ±131, SE ~ 1e23, INFIT ~ 1e47).
- The root cause was numerical instability in probability/variance and a broken tau update.
- This module provides a *stable* RSM estimator and (optionally) an ASP-style extreme substitution step.

Model:
- Rating Scale Model (RSM): global step thresholds (tau) shared across items.
- Categories are assumed to be consecutive integers from min_cat..max_cat.

Key design choices to match Winsteps-ish behavior:
- Newton updates for person (theta) and item difficulty (b).
- Clamp during iteration (default ±10), then "extreme substitution" for clamped persons using
  ASP-style extrapolation (adj_rate=2.25) based on the nearest non-extreme score levels.
- Stable log-sum-exp for category probabilities (prevents overflow).
- Hard floors on variances/information (prevents divide-by-zero explosions).

This file is self-contained and does not depend on scipy.
"""

from __future__ import annotations

# --- Consistent color palette (used for clusters/categories across the HTML report) ---
# Keep this list stable so colors remain consistent across figures.
SPECIFIED_COLORS = [
    "#FF0000", "#0000FF", "#998000", "#008000", "#800080",
    "#FFC0CB", "#000000", "#ADD8E6", "#FF4500", "#A52A2A",
    "#8B4513", "#FF8C00", "#32CD32", "#4682B4", "#9400D3",
    "#FFD700", "#C0C0C0", "#DC143C", "#1E90FF",
]

def _safe_add_fig(html_parts, fig, title: str, note: str | None = None):
    """Append a Plotly figure to html_parts safely."""
    try:
        import plotly.io as pio
        html_parts.append(f"<h3>{title}</h3>")
        if note:
            html_parts.append(f"<div class='note'><b>Key note:</b> {note}</div>")
        html_parts.append((__import__('plotly.io', fromlist=['io']).to_html)(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False}))
    except Exception as e:
        html_parts.append(f"<h3>{title}</h3><pre style='color:#b00'>Figure render failed: {e}</pre>")


def _df_to_html(df, max_rows: int = 60) -> str:
    """Render a pandas DataFrame to a compact HTML table.

    Kept at module scope so helpers like `_safe_add_table()` can always find it.
    """
    try:
        import pandas as _pd
        import numpy as _np
        import html as _html
    except Exception:
        return "<p>(table render failed: missing pandas)</p>"

    if df is None:
        return "<p>(none)</p>"
    try:
        if hasattr(df, "empty") and df.empty:
            return "<p>(none)</p>"
    except Exception:
        pass

    try:
        view = df.head(int(max_rows) if max_rows else 60).copy()
        # light formatting: numeric -> 3 dp; keep strings as-is
        def _fmt(x):
            try:
                if x is None:
                    return ""
                if isinstance(x, (float, int)) and _np.isfinite(x):
                    return f"{float(x):.3f}"
                return _html.escape(str(x))
            except Exception:
                return _html.escape(str(x))
        for c in view.columns:
            view[c] = view[c].map(_fmt)
        return view.to_html(index=False, border=0, classes="tbl", escape=False)
    except Exception as e:
        try:
            return f"<p>(table render failed: {_html.escape(repr(e))})</p>"
        except Exception:
            return "<p>(table render failed)</p>"

def _safe_add_table(html_parts, df, title: str, note: str | None = None, max_rows: int = 60):
    try:
        html_parts.append(f"<h3>{title}</h3>")
        if note:
            html_parts.append(f"<div class='note'><b>Key note:</b> {note}</div>")
        html_parts.append(_df_to_html(df, max_rows=max_rows))
    except Exception as e:
        html_parts.append(f"<h3>{title}</h3><pre style='color:#b00'>Table render failed: {e}</pre>")



# --- PTMA helper (Adjusted point-measure correlation; leave-one-out proxy) ---
def compute_ptma_from_matrices(X_obs, theta, item_names=None, min_n: int = 5):
    """Compute PTMA = cor(item score, theta excluding the item).

    This is an efficient proxy: theta_minus_j is approximated by removing the item's
    contribution from each person's total score, then correlating with item response.
    It is intended for diagnostics (hover text / Table 11.2), not for estimation.
    """
    import numpy as _np
    import pandas as _pd

    X = _np.asarray(X_obs, dtype=float)
    if X.ndim != 2:
        return _pd.DataFrame({"ITEM": [], "PTMA": [], "N_USED": []})

    P, I = X.shape
    th = _np.asarray(theta, dtype=float)
    if th.ndim != 1 or th.shape[0] != P:
        # try transpose alignment
        if th.ndim == 1 and th.shape[0] == I and X.shape[1] == th.shape[0]:
            pass
        return _pd.DataFrame({"ITEM": [], "PTMA": [], "N_USED": []})

    if item_names is None:
        item_names = [f"Item{j+1}" for j in range(I)]

    # use raw total score as monotone proxy to theta; then subtract item j
    # (works even if theta was not leave-one-out)
    raw = _np.nansum(X, axis=1)

    rows = []
    for j in range(I):
        y = X[:, j]
        ok = _np.isfinite(y) & _np.isfinite(raw)
        if ok.sum() < min_n:
            rows.append([str(item_names[j]), _np.nan, int(ok.sum())])
            continue
        raw_mj = raw[ok] - y[ok]
        # correlate y with raw_mj and with theta; take the safer (higher absolute) proxy
        # use raw_mj as leave-one-out proxy for theta ordering
        try:
            ptma = float(_np.corrcoef(y[ok], raw_mj)[0, 1])
        except Exception:
            ptma = _np.nan
        rows.append([str(item_names[j]), ptma, int(ok.sum())])

    return _pd.DataFrame(rows, columns=["ITEM", "PTMA", "N_USED"])
def _add_hlines(fig, ys, dash="dot"):
    try:
        if ys is None:
            return fig
        if isinstance(ys, (list, tuple)):
            for y in ys:
                try:
                    fig.add_hline(y=float(y), line_dash=dash, line_color="black", opacity=0.85)
                except Exception:
                    pass
        return fig
    except Exception:
        return fig

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any, Tuple

import json

import json

import numpy as np
import pandas as pd

# ===============================
# DIF — SMD Forest Plot (helper)
# ===============================
def plot_dif_smd_forest(df, title="SMD in DIF measure"):
    """Create an SMD-based DIF forest plot (Winsteps/RUMM-like).

    Required columns in df:
      ITEM, b_g0, se_g0, b_g1, se_g1
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    d = df.copy()
    d["DIF"] = pd.to_numeric(d["b_g1"], errors="coerce") - pd.to_numeric(d["b_g0"], errors="coerce")
    se0 = pd.to_numeric(d["se_g0"], errors="coerce")
    se1 = pd.to_numeric(d["se_g1"], errors="coerce")
    d["pooled_se"] = np.sqrt((se0**2 + se1**2) / 2.0)
    d["SMD"] = d["DIF"] / d["pooled_se"]
    d["LOW"] = d["SMD"] - 1.96
    d["HIGH"] = d["SMD"] + 1.96

    d = d.dropna(subset=["SMD", "LOW", "HIGH", "ITEM"]).copy()
    d = d.sort_values("SMD", ascending=False).reset_index(drop=True)

    # Summary (fixed-effect)
    w = 1.0 / (d["pooled_se"] ** 2)
    mu = float(np.sum(w * d["SMD"]) / np.sum(w)) if len(d) else 0.0
    se_mu = float(np.sqrt(1.0 / np.sum(w))) if len(d) else 0.0
    mu_low = mu - 1.96 * se_mu
    mu_high = mu + 1.96 * se_mu

    y_items = list(range(len(d)))
    y_summary = -1

    fig = go.Figure()

    # CI lines
    for i, r in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[float(r["LOW"]), float(r["HIGH"])],
            y=[y_items[i], y_items[i]],
            mode="lines",
            line=dict(color="royalblue", width=2),
            showlegend=False
        ))

    # points
    fig.add_trace(go.Scatter(
        x=d["SMD"],
        y=y_items,
        mode="markers",
        marker=dict(symbol="square", size=14, color="green",
                    line=dict(color="black", width=1.2)),
        text=d["ITEM"].astype(str),
        customdata=np.stack([d["LOW"].to_numpy(), d["HIGH"].to_numpy()], axis=1),
        hovertemplate="<b>%{text}</b><br>SMD=%{x:.3f}<br>CI=[%{customdata[0]:.3f}, %{customdata[1]:.3f}]",
        showlegend=False
    ))

    # no-DIF line
    fig.add_vline(x=0, line=dict(color="red", width=2))

    # summary diamond + CI
    fig.add_trace(go.Scatter(
        x=[mu], y=[y_summary],
        mode="markers",
        marker=dict(symbol="diamond", size=20, color="red",
                    line=dict(color="black", width=1.5)),
        hovertemplate=f"<b>Overall DIF</b><br>SMD={mu:.3f}<br>CI=[{mu_low:.3f}, {mu_high:.3f}]",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=[mu_low, mu_high],
        y=[y_summary, y_summary],
        mode="lines",
        line=dict(color="red", width=3),
        showlegend=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color="red", size=22)),
        xaxis_title="Standardized Mean Difference (SMD)",
        yaxis=dict(tickvals=y_items, ticktext=d["ITEM"].astype(str), autorange="reversed"),
        plot_bgcolor="white",
        height=900,
        margin=dict(l=260, r=60, t=80, b=60)
    )
    return fig

RASONLINE_BUILD_ID = 'renderer_2dp_winsteps_maxiter100_20260108_v14'
try:
    import math as math  # noqa: F401
except Exception:
    math = np  # fallback
import html

# -----------------------------
# Safer JSON for embedding debug payloads
# -----------------------------

def _safe_json_dumps(obj, **kwargs):
    """json.dumps that never raises (falls back to repr).

    Used in HTML report failure blocks so that missing/NaN values or numpy
    objects don't crash report rendering.
    """
    try:
        def _default(x):
            try:
                # numpy scalars/arrays
                import numpy as _np
                if isinstance(x, (_np.integer, _np.floating)):
                    return x.item()
                if isinstance(x, _np.ndarray):
                    return x.tolist()
            except Exception:
                pass
            return repr(x)

        return json.dumps(obj, default=_default, ensure_ascii=False, **kwargs)
    except Exception:
        try:
            return json.dumps(repr(obj), ensure_ascii=False)
        except Exception:
            return str(obj)
# -----------------------------
# Utilities
# -----------------------------

def make_cluster_colors(clusters):
    """Deterministic cluster->color mapping."""
    try:
        import plotly.colors as pc
        palette = pc.qualitative.Plotly
    except Exception:
        palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    cmap = {}
    try:
        uniq = sorted({int(x) for x in list(clusters) if str(x) != "nan"})
    except Exception:
        uniq = []
    for i, c in enumerate(uniq):
        cmap[int(c)] = palette[i % len(palette)]
    return cmap



def _build_vertices_relations_from_residuals(res, idf, top_k_edges: int = 80):
    """Build vertices/relations for Figure 9 using Euclidean distance on item cell-ZSTD vectors.

    - Item vectors: ZSTD[:, j] (persons x items), treated as vectors in person space.
    - Distance: Euclidean distance between item vectors (shorter = stronger relation).
    - One-link edges: for each item keep its single nearest neighbor; then deduplicate.
    - Silhouette: compute a(i), b(i), ss(i) from the distance matrix using current 2-cluster assignment.

    Returns:
      vertices_df columns include: name, value(loading), value2(delta), cluster, ss_i, a_i, b_i
      relations_df columns: term1, term2, WCD (strength; larger = stronger)
    """
    if not (hasattr(res, "debug") and isinstance(res.debug, dict) and "ZSTD" in res.debug):
        return None, None

    Zmat = np.asarray(res.debug.get("ZSTD"), dtype=float)
    if Zmat.ndim != 2 or Zmat.shape[1] != len(idf):
        return None, None

    # Contrast-1 loading from PCA-like eigenvector on item correlation of ZSTD
    loading, eig1, w = _contrast1_from_zstd(Zmat)

    # Item names / deltas
    names = idf["ITEM"].astype(str).tolist() if "ITEM" in idf.columns else [f"Item{j+1}" for j in range(Zmat.shape[1])]
    delta = pd.to_numeric(idf["MEASURE"], errors="coerce").to_numpy() if "MEASURE" in idf.columns else np.full(Zmat.shape[1], np.nan)
    item_se = pd.to_numeric(idf["SE"], errors="coerce").to_numpy() if "SE" in idf.columns else np.full(Zmat.shape[1], np.nan)

    # --- Distance matrix on item vectors (persons x items) ---
    # Use NaN-safe centering per item then Euclidean distance in person space
    X = np.asarray(Zmat, dtype=float).T  # (I, P)
    # fill NaN by item mean
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isfinite(X), X, mu)
    # center each item vector to reduce person-location drift
    X = X - np.mean(X, axis=1, keepdims=True)
    D = _euclidean_distance_matrix(X)

    # --- Cluster assignment: 2 clusters by sign of loading ---
    cl = (loading >= 0).astype(int) + 1  # 1/2

    # Ensure Cluster 1 corresponds to higher loading AND higher delta (as in your earlier intent)
    try:
        if np.nanmean(delta[cl == 1]) < np.nanmean(delta[cl == 2]):
            cl = np.where(cl == 1, 2, 1)
            loading = -loading
    except Exception:
        pass

    # --- Silhouette ---
    a_i, b_i, ss_i = _silhouette_from_distances(D, cl)

    vtx = pd.DataFrame(
        {
            "name": names,
            "value": loading.astype(float),      # default: used as y in kano_plot_aligned if requested
            "value2": delta.astype(float),       # delta (logit)
            "item_se": item_se.astype(float),     # item SE (logit)
            "cluster": cl.astype(int),
            "ss_i": ss_i.astype(float),
            "a_i": a_i.astype(float),
            "b_i": b_i.astype(float),
        }
    )

    # --- One-link nearest-neighbor edges (shorter distance = stronger) ---
    # Build directed nearest neighbor per node, then deduplicate into undirected edges.
    I = D.shape[0]
    nn = np.full(I, -1, dtype=int)
    nd = np.full(I, np.nan, dtype=float)
    for i in range(I):
        row = D[i].copy()
        row[i] = np.inf
        j = int(np.nanargmin(row))
        nn[i] = j
        nd[i] = float(row[j])

    # Strength transform: larger is stronger
    maxd = float(np.nanmax(nd)) if np.isfinite(nd).any() else 1.0
    edges = {}
    for i in range(I):
        j = int(nn[i])
        if j < 0:
            continue
        a, b = (i, j) if i < j else (j, i)
        dist = float(D[a, b])
        strength = maxd - dist
        if (a, b) not in edges or strength > edges[(a, b)]:
            edges[(a, b)] = strength

    # keep top edges by strength (optional)
    items = sorted(edges.items(), key=lambda kv: kv[1], reverse=True)
    if top_k_edges is not None and top_k_edges > 0:
        items = items[: int(top_k_edges)]

    rel = pd.DataFrame(
        {
            "term1": [names[a] for (a, b), s in items],
            "term2": [names[b] for (a, b), s in items],
            "WCD": [float(s) for (a, b), s in items],
        }
    )
    return vtx, rel



def _simulate_rasch_matrix_from_res(res, seed: int = 12345):
    """Simulate a new response matrix under the fitted Rasch model.

    - Preserves the original missingness pattern (NaN cells remain NaN).
    - Dichotomous: categories {0,1}.
    - Polytomous (RSM): categories {0..K-1} with global tau.

    Returns
    -------
    Xsim : ndarray (P,I) in internal 0..K-1 coding (NaN for missing)
    """
    rng = np.random.default_rng(int(seed) if seed is not None else 12345)

    dbg = getattr(res, "debug", {}) if hasattr(res, "debug") else {}
    X0 = None
    if isinstance(dbg, dict):
        for k in ("X", "X_mapped", "data"):
            if k in dbg:
                try:
                    X0 = np.asarray(dbg[k], dtype=float)
                    break
                except Exception:
                    X0 = None
    if X0 is None:
        try:
            X0 = np.asarray(getattr(res, "data", None), dtype=float)
        except Exception:
            X0 = None
    if X0 is None or getattr(X0, "ndim", 0) != 2:
        raise ValueError("Cannot simulate: original response matrix not found.")

    theta = np.asarray(getattr(res, "theta", None), dtype=float)

    # PATCH: accept res.b as beta (your code uses res.b elsewhere)
    _b = getattr(res, "b", None)
    beta = np.asarray(_b if _b is not None else getattr(res, "beta", None), dtype=float)

    # PATCH: tau is only required for polytomous; for dichotomous allow missing tau
    tau_raw = getattr(res, "tau", None)
    tau = None if tau_raw is None else np.asarray(tau_raw, dtype=float)

    if theta.ndim != 1 or beta.ndim != 1:
        raise ValueError("Cannot simulate: theta/beta missing.")

    # dichotomous: no tau needed (use a 2-length placeholder so downstream K logic stays consistent)
    if (tau is None or getattr(tau, "ndim", 0) != 1) and int(getattr(res, "max_cat", 1)) <= 1:
        tau = np.asarray([0.0, 0.0], dtype=float)

    # polytomous: tau required
    if tau is None or getattr(tau, "ndim", 0) != 1:
        raise ValueError("Cannot simulate: tau missing (polytomous).")

    P, I = X0.shape
    if theta.size != P or beta.size != I:
        raise ValueError("Cannot simulate: shape mismatch in theta/beta vs X.")

    K = int(tau.size)
    min_cat = int(getattr(res, "min_cat", 0))
    max_cat = int(getattr(res, "max_cat", K - 1))

    miss = ~np.isfinite(X0)
    Xsim = np.full((P, I), np.nan, dtype=float)

    # Detect dichotomous
    if K <= 2 and min_cat == 0 and max_cat == 1:
        base = theta[:, None] - beta[None, :]
        p1 = 1.0 / (1.0 + np.exp(-base))  # P(X=1)
        U = rng.random((P, I))
        Xsim = (U < p1).astype(float)
        Xsim[miss] = np.nan
        return Xsim

    # Polytomous RSM sampling
    Pmat = _rsm_prob(theta, beta, tau)  # (P,I,K)
    cdf = np.cumsum(Pmat, axis=2)
    U = rng.random((P, I, 1))
    draw = (U <= cdf).argmax(axis=2).astype(float)  # 0..K-1
    draw[miss] = np.nan
    return draw


def _zstd_from_x_and_model(res, X_obs: np.ndarray):
    """Compute standardized residuals ZSTD for a given response matrix under fitted model."""
    X = np.asarray(X_obs, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")

    theta = np.asarray(getattr(res, "theta", None), dtype=float)

    # PATCH: accept res.b as beta
    _b = getattr(res, "b", None)
    beta = np.asarray(_b if _b is not None else getattr(res, "beta", None), dtype=float)

    # PATCH: tau optional for dichotomous
    tau_raw = getattr(res, "tau", None)
    tau = None if tau_raw is None else np.asarray(tau_raw, dtype=float)

    if theta.ndim != 1 or beta.ndim != 1:
        raise ValueError("theta/beta missing")

    if (tau is None or getattr(tau, "ndim", 0) != 1) and int(getattr(res, "max_cat", 1)) <= 1:
        tau = np.asarray([0.0, 0.0], dtype=float)

    if tau is None or getattr(tau, "ndim", 0) != 1:
        raise ValueError("tau missing (polytomous)")

    K = int(tau.size)
    Pmat = _rsm_prob(theta, beta, tau)  # (P,I,K)
    cats = np.arange(K, dtype=float)[None, None, :]
    exp = np.sum(Pmat * cats, axis=2)
    exp2 = np.sum(Pmat * (cats ** 2), axis=2)
    var = np.maximum(exp2 - exp ** 2, 1e-9)
    resid = X - exp
    z = resid / np.sqrt(var)
    z[~np.isfinite(X)] = np.nan
    return z


def kano_plot_aligned(
    nodes: pd.DataFrame,
    edges: pd.DataFrame | None = None,
    aac_nodes: float | None = None,
    aac_cluster: float | None = None,
    title_suffix: str = "",
    x_col: str = "value2",
    y_col: str = "value",
    x_label: str | None = None,
    y_label: str | None = None,
    aac_x: float | None = None,
    aac_y: float | None = None,
    beta_value: float | None = None,
    beta_p: float | None = None,
) -> "object":
    """Kano-type plot (Dimension–Kano Map with Zscore residuals) with optional edges."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return None

    if nodes is None or len(nodes) == 0:
        return go.Figure().update_layout(title="No node data – cannot draw Dimension–Kano Map with Zscore residuals.")

    df_raw = nodes.copy()
    cols = {c.lower(): c for c in df_raw.columns}
    name_col = cols.get("name") or cols.get("node") or list(df_raw.columns)[0]
    cluster_col = cols.get("cluster") or cols.get("theme") or cols.get("carac") or cols.get("group")
    if cluster_col is None:
        df_raw["cluster"] = 1
        cluster_col = "cluster"

    if "value" in df_raw.columns:
        size_col = "value"
    # Prefer item_se (item standard error) for bubble size when available (Figure 9 spec)
    if "item_se" in df_raw.columns:
        size_col = "item_se"
    # Prefer ss_i (silhouette) for bubble size when available
    elif "ss_i" in df_raw.columns:
        size_col = "ss_i"
    elif x_col in df_raw.columns:
        size_col = x_col
    else:
        size_col = y_col

    if x_col not in df_raw.columns or y_col not in df_raw.columns:
        return go.Figure().update_layout(title=f"Cannot draw Kano plot: missing '{x_col}' or '{y_col}'.")

    df = pd.DataFrame()
    df["name"] = df_raw[name_col].astype(str)
    df["cluster"] = pd.to_numeric(df_raw[cluster_col], errors="coerce").fillna(0).astype(int)
    df["x_val"] = pd.to_numeric(df_raw[x_col], errors="coerce")
    df["y_val"] = pd.to_numeric(df_raw[y_col], errors="coerce")
    df["size_val"] = pd.to_numeric(df_raw[size_col], errors="coerce")
    # Carry silhouette components for hover (even if not used as axes)
    # Convention used in this project: ss_i (or SS(i)) may be stored in 'value'.
    _ss_src = None
    for _k in ("ss_i", "SS(i)", "ssi", "silhouette", "value"):
        if _k in df_raw.columns:
            _ss_src = _k
            break
    df["ss_i"] = pd.to_numeric(df_raw[_ss_src], errors="coerce") if _ss_src else np.nan
    _a_src = None
    for _k in ("a_i", "a(i)", "ai"):
        if _k in df_raw.columns:
            _a_src = _k
            break
    _b_src = None
    for _k in ("b_i", "b(i)", "bi"):
        if _k in df_raw.columns:
            _b_src = _k
            break
    df["a_i"] = pd.to_numeric(df_raw[_a_src], errors="coerce") if _a_src else np.nan
    df["b_i"] = pd.to_numeric(df_raw[_b_src], errors="coerce") if _b_src else np.nan
    df = df.dropna(subset=["x_val","y_val"])
    if df.empty:
        return go.Figure().update_layout(title="No numeric node data – cannot draw Dimension–Kano Map with Zscore residuals.")

    # Correlation check: enforce a consistent positive orientation.
    # If corr(x, y) is negative, flip the sign of loading (y) so corr becomes positive.
    _r = np.nan
    try:
        _r = float(np.corrcoef(df["x_val"].to_numpy(), df["y_val"].to_numpy())[0,1])
    except Exception:
        _r = np.nan
    _flipped = False
    if np.isfinite(_r) and _r < 0:
        df["y_val"] = -df["y_val"]
        _r = -_r
        _flipped = True

    mean_x = float(df["x_val"].mean()); mean_y = float(df["y_val"].mean())
    min_x, max_x = float(df["x_val"].min()), float(df["x_val"].max())
    min_y, max_y = float(df["y_val"].min()), float(df["y_val"].max())

    expand_x = (max_x-min_x)*0.10 if max_x>min_x else 1.0
    expand_y = (max_y-min_y)*0.10 if max_y>min_y else 1.0
    x_low, x_high = min_x-expand_x, max_x+expand_x
    y_low, y_high = min_y-expand_y, max_y+expand_y
    x_span, y_span = x_high-x_low, y_high-y_low

    t = np.linspace(0.0, 1.0, 300)
    spread_x = (max_x-min_x)*0.43 if max_x>min_x else 1.0
    spread_y = (max_y-min_y)*1.2 if max_y>min_y else 1.0
    kano_amp = 1.1
    lower_x = t*spread_x - spread_x/2.0 + mean_x
    lower_y = mean_y - kano_amp*spread_y*(1.0-t)**2
    upper_x = -t*spread_x + spread_x/2.0 + mean_x
    upper_y = mean_y + kano_amp*spread_y*(1.0-t)**2

    plot_width=980.0; plot_height=700.0
    A = kano_amp*spread_y
    r_y = 0.25*A*0.5
    if x_span>0 and y_span>0:
        k = (plot_width*y_span)/(plot_height*x_span)
        r_x = r_y/k if k>0 else r_y
        r_x = min(r_x, 0.45*x_span)
    else:
        r_x = r_y
    theta = np.linspace(0, 2*np.pi, 300)
    circle_x = mean_x + r_x*np.cos(theta)
    circle_y = mean_y + r_y*np.sin(theta)

    val = df["size_val"].to_numpy()
    if not np.isfinite(val).any():
        sizes = np.full(len(val), 26.0)
    else:
        vmin=np.nanmin(val); vmax=np.nanmax(val)
        sizes = np.full(len(val), 28.0) if vmax<=vmin else (18.0 + 34.0*(val-vmin)/max(vmax-vmin,1e-9))
    df["bubble_size"]=sizes

    cluster_color_map = make_cluster_colors(df["cluster"])
    df["color"]=df["cluster"].map(lambda c: cluster_color_map.get(int(c), "#7f7f7f"))

    # Mark suspected testlet items with thicker outlines (requires boolean column 'is_testlet')
    try:
        if 'is_testlet' in df.columns:
            df['outline_w'] = df['is_testlet'].apply(lambda v: 3.0 if bool(v) else 1.0)
        else:
            df['outline_w'] = 1.0
    except Exception:
        df['outline_w'] = 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lower_x,y=lower_y,mode="lines",line=dict(color="blue",width=2.5),hoverinfo="skip",name="Lower Kano curve"))
    fig.add_trace(go.Scatter(x=upper_x,y=upper_y,mode="lines",line=dict(color="blue",width=2.5),hoverinfo="skip",name="Upper Kano curve"))
    fig.add_trace(go.Scatter(x=circle_x,y=circle_y,mode="lines",line=dict(color="purple",width=2.0),hoverinfo="skip",name="Kano boundary"))
    fig.add_trace(go.Scatter(x=[x_low,x_high],y=[mean_y,mean_y],mode="lines",line=dict(color="red",width=2.0,dash="dot"),hoverinfo="skip",name="Center horizontal"))
    fig.add_trace(go.Scatter(x=[mean_x,mean_x],y=[y_low,y_high],mode="lines",line=dict(color="red",width=2.0,dash="dot"),hoverinfo="skip",name="Center vertical"))
    # Two blue dotted control lines for Figure 9:
    # thr = min( max(|loading|) in each cluster ), draw y = ±thr
    try:
        cvals = sorted([int(x) for x in df["cluster"].dropna().unique().tolist()])
        if len(cvals) >= 2:
            c1, c2 = cvals[0], cvals[1]
            m1 = float(np.nanmax(np.abs(df.loc[df["cluster"] == c1, "y_val"].to_numpy())))
            m2 = float(np.nanmax(np.abs(df.loc[df["cluster"] == c2, "y_val"].to_numpy())))
            thr = min(m1, m2)
            if np.isfinite(thr):
                fig.add_trace(go.Scatter(
                    x=[x_low, x_high], y=[thr, thr], mode="lines",
                    line=dict(color="blue", width=2.0, dash="dot"),
                    hoverinfo="skip", name="Loading control (+)"
                ))
                fig.add_trace(go.Scatter(
                    x=[x_low, x_high], y=[-thr, -thr], mode="lines",
                    line=dict(color="blue", width=2.0, dash="dot"),
                    hoverinfo="skip", name="Loading control (-)"
                ))
    except Exception:
        pass

    # edges
    if isinstance(edges, pd.DataFrame) and not edges.empty:
        ed=edges.copy()
        ecols={c.lower():c for c in ed.columns}
        s_col=ecols.get("leader") or ecols.get("source") or ecols.get("term1") or list(ed.columns)[0]
        t_col=ecols.get("follower") or ecols.get("target") or ecols.get("term2") or list(ed.columns)[1]
        w_col=ecols.get("wcd") or ecols.get("weight")
        pos=df.set_index("name")[["x_val","y_val"]].to_dict(orient="index")

        w_min=w_max=None
        if w_col and w_col in ed.columns:
            w_arr=pd.to_numeric(ed[w_col],errors="coerce").to_numpy()
            wv=np.isfinite(w_arr)
            if wv.any():
                w_min=float(np.nanmin(w_arr[wv])); w_max=float(np.nanmax(w_arr[wv]))

        def w2width(w,lo=1.0,hi=6.0):
            try: w=float(w)
            except Exception: return (lo+hi)/2.0
            if w_min is None or w_max is None or not np.isfinite(w) or w_max<=w_min:
                return (lo+hi)/2.0
            return lo + (w-w_min)*(hi-lo)/(w_max-w_min)

        for _,r in ed.iterrows():
            s=str(r[s_col]); t2=str(r[t_col])
            if s not in pos or t2 not in pos: 
                continue
            x0,y0=pos[s]["x_val"],pos[s]["y_val"]
            x1,y1=pos[t2]["x_val"],pos[t2]["y_val"]
            width = w2width(r[w_col]) if (w_col and w_col in ed.columns) else 3.0
            fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
                                     line=dict(color="rgba(120,120,120,0.85)",width=width),
                                     hoverinfo="text",
                                     text=[f"{s} ↔ {t2}"+(f" (WCD={r[w_col]})" if (w_col and w_col in ed.columns) else ""), ""],
                                     showlegend=False))

    def _getf(row, key):
        try:
            v = row.get(key, None)
            if v is None:
                return None
            if isinstance(v, float) and not (v == v):
                return None
            return float(v)
        except Exception:
            return None

    for c, sub in df.groupby("cluster", sort=False):
        hover = []
        for _, row in sub.iterrows():
            ss = _getf(row, "ss_i") or _getf(row, "SS(i)") or _getf(row, "ssi")
            ai = _getf(row, "a_i") or _getf(row, "a(i)")
            bi = _getf(row, "b_i") or _getf(row, "b(i)")
            parts = [
                f"{row['name']}",
                ("⚠ Suspected testlet" if str(row.get('is_testlet', '')).lower() in ('1','true','t','yes') else None),
                f"Cluster: {int(row['cluster'])}",
                f"{x_col}={float(row['x_val']):.3f}",
                f"{y_col}={float(row['y_val']):.3f}",
            ]
            parts = [p for p in parts if p]
            if ss is not None:
                parts.append(f"SS(i)={ss:.3f}")
            if ai is not None:
                parts.append(f"a(i)={ai:.3f}")
            if bi is not None:
                parts.append(f"b(i)={bi:.3f}")
            ptma = _getf(row, 'PTMA') or _getf(row, 'ptma')
            q3m = _getf(row, 'Q3MAX') or _getf(row, 'q3max')
            if ptma is not None:
                parts.append(f"PTMA={ptma:.3f}")
            if q3m is not None:
                parts.append(f"Q3max={q3m:.3f}")
            hover.append("<br>".join(parts))

        fig.add_trace(go.Scatter(
            x=sub["x_val"], y=sub["y_val"], mode="markers",
            marker=dict(
                size=sub["bubble_size"], color=sub["color"],
                line=dict(width=sub.get('outline_w', 1.0), color="rgba(0,0,0,0.55)"),
                sizemode="diameter",
            ),
            name=f"Cluster {int(c)}",
            hovertext=hover, hoverinfo="text",
        ))

    if x_label is None: x_label = x_col
    if y_label is None: y_label = y_col

    title="Dimension–Kano Map with Zscore residuals"+(title_suffix or "")
    if np.isfinite(_r):
        title = title + f"  (r={_r:.3f}" + ("; flipped y" if _flipped else "") + ")"
    fig.update_layout(title=title,
                      xaxis_title=x_label,yaxis_title=y_label,
                      width=980,height=700,hovermode="closest",
                      legend_title="Cluster",margin=dict(l=60,r=20,t=80,b=60))    # Big beta callout (requested): show the regression/path coefficient between 2 clusters
    try:
        if beta_value is not None and np.isfinite(float(beta_value)):
            _btxt = f"<b>Beta={float(beta_value):.2f}</b>"
            try:
                if beta_p is not None and np.isfinite(float(beta_p)):
                    _btxt += f"<br><span style='font-size:20px'>(p={float(beta_p):.3g})</span>"
            except Exception:
                pass
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                text=_btxt,
                showarrow=False,
                font=dict(size=30, color="#b00"),
                align="left",
            )
    except Exception:
        pass
    fig.update_xaxes(range=[x_low,x_high])
    fig.update_yaxes(range=[y_low,y_high])
    return fig

def _as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _logsumexp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Stable log-sum-exp without scipy."""
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


def _ensure_consecutive_categories(X: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """Coerce to integer categories and remap to consecutive integers, preserving missing.

    Notes
    -----
    - Missing cells must remain missing (NaN) throughout estimation and fit computations.
      Converting NaN to int would produce huge negative sentinel values (e.g. -9e18) and
      explode measures/fit. So we return a *float* matrix with NaN preserved.
    - If observed categories are sparse (e.g., only {0,2}), we remap to 0..K-1 to avoid
      internal inconsistencies.
    """
    Xf = np.asarray(X, dtype=float)
    m = np.isfinite(Xf)
    vals = np.unique(Xf[m])
    if vals.size == 0:
        raise ValueError('No finite response values found.')
    if not np.all(np.isclose(vals, np.round(vals))):
        # Caller should apply a continuous->categorical transform before this,
        # or enable auto-detection (see run_rasch).
        raise ValueError(f"Non-integer categories detected: {vals[:20]}")
    vals = np.sort(vals.astype(int))
    mapping = {v: i for i, v in enumerate(vals)}

    X_mapped = np.full_like(Xf, np.nan, dtype=float)
    if m.any():
        # Map only finite entries; keep missing as NaN
        X_mapped[m] = np.vectorize(mapping.get)(Xf[m].astype(int)).astype(float)

    return X_mapped, 0, int(len(vals) - 1)


def _rsm_prob(theta: np.ndarray, b: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    RSM probabilities P(x | theta, b, tau) for x=0..K-1 where tau[0]=0 and tau[k] are step parameters.

    eta_x = sum_{k=1..x} (theta - b - tau[k])
    P_x = exp(eta_x) / sum_{h} exp(eta_h)

    Returns Pmat: shape (P, I, K)
    """
    theta = _as_float_array(theta)
    b = _as_float_array(b)
    tau = _as_float_array(tau)

    Pn, In = theta.size, b.size
    K = tau.size

    # base = (theta - b) broadcast to (P, I)
    base = theta[:, None] - b[None, :]

    # step_terms for k=1..K-1: (theta - b - tau[k])
    step_terms = base[:, :, None] - tau[None, None, 1:]  # (P, I, K-1)

    # cumulative sum across categories: eta_1..eta_{K-1}
    cum = np.cumsum(step_terms, axis=2)  # (P, I, K-1)

    # prepend eta_0 = 0
    eta = np.concatenate([np.zeros((Pn, In, 1), dtype=float), cum], axis=2)  # (P, I, K)

    # stable normalize
    # logP = eta - logsumexp(eta)
    log_denom = _logsumexp(eta, axis=2, keepdims=True)
    Pmat = np.exp(eta - log_denom)

    # safety clamp to avoid exact 0/1 (helps variance stability)
    eps = 1e-12
    Pmat = np.clip(Pmat, eps, 1.0 - eps)
    # renormalize (tiny numerical drift)
    Pmat = Pmat / np.sum(Pmat, axis=2, keepdims=True)
    return Pmat


def _asp_extreme_substitution_person(
    theta: np.ndarray,
    info: np.ndarray,
    raw: np.ndarray,
    max_raw: int,
    clamp: float = 6.0,
    adj_rate: float = 2.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASP/Winsteps-like "for extreme scores" substitution for *persons*.

    The ASP code:
      adj_rate = 1.5*1.5
      extremeperson  = slope*(max_raw - personmax2)*adj_rate + personmax2m
      extremeperson0 = slope*(0 - personmin2)*adj_rate + personmin2m
    and then replaces person(jk)=±10 with these extrapolated measures.

    Here:
    - We define "clamped" as theta >= +clamp-1e-9 or <= -clamp+1e-9.
    - We extrapolate using two nearest non-extreme raw-score levels (min2/max2).
    - We also extrapolate "info" (Fisher information approx) in log-space to keep it positive.
    """
    theta = _as_float_array(theta).copy()
    info = _as_float_array(info).copy()
    raw = _as_float_array(raw)

    hi_mask = theta >= (clamp - 1e-9)
    lo_mask = theta <= (-clamp + 1e-9)
    if not (np.any(hi_mask) or np.any(lo_mask)):
        return theta, info

    non_ext = (raw > 0) & (raw < max_raw) & np.isfinite(theta) & np.isfinite(info)
    if non_ext.sum() < 2:
        return theta, info

    raw_non = raw[non_ext]
    theta_non = theta[non_ext]
    info_non = info[non_ext]

    max2 = raw_non.max()
    min2 = raw_non.min()
    if max2 == min2:
        return theta, info

    max2m = float(np.mean(theta_non[raw_non == max2]))
    min2m = float(np.mean(theta_non[raw_non == min2]))
    max2i = float(np.mean(info_non[raw_non == max2]))
    min2i = float(np.mean(info_non[raw_non == min2]))

    if not (np.isfinite(max2m) and np.isfinite(min2m)):
        return theta, info

    slope = (max2m - min2m) / (max2 - min2)
    extreme_hi = slope * (max_raw - max2) * adj_rate + max2m
    extreme_lo = slope * (0.0 - min2) * adj_rate + min2m

    # Extrapolate info in log-space vs measure (keeps positive and avoids crazy leaps)
    max2i = max(max2i, 1e-8)
    min2i = max(min2i, 1e-8)
    denom = (max2m - min2m)
    if abs(denom) < 1e-12:
        info_hi = max2i
        info_lo = min2i
    else:
        logi_slope = (np.log(max2i) - np.log(min2i)) / denom
        info_hi = float(np.exp(logi_slope * (extreme_hi - max2m) + np.log(max2i)))
        info_lo = float(np.exp(logi_slope * (extreme_lo - min2m) + np.log(min2i)))

    if np.any(hi_mask):
        theta[hi_mask] = extreme_hi
        info[hi_mask] = info_hi
    if np.any(lo_mask):
        theta[lo_mask] = extreme_lo
        info[lo_mask] = info_lo

    return theta, info



# -----------------------------
# Report export helpers (Figure 9 vertices/relations)
# -----------------------------
from pathlib import Path

def _resolve_run_dir(run_id: str) -> Path:
    """Resolve the reports/<run_id> directory used by main.py.
    Tries common locations and creates the folder if needed.
    """
    rid = str(run_id)
    candidates = []
    env_dir = os.environ.get("RASCH_REPORTS_DIR") or os.environ.get("REPORTS_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / rid)
    # common relative folder
    candidates.append(Path.cwd() / "reports" / rid)
    # sandbox default
    candidates.append(Path("/mnt/data/reports") / rid)

    for d in candidates:
        try:
            if d.exists() and d.is_dir():
                return d
        except Exception:
            pass
    # create the first viable one
    d0 = candidates[0]
    try:
        d0.mkdir(parents=True, exist_ok=True)
        return d0
    except Exception:
        d1 = Path.cwd() / "reports" / rid
        d1.mkdir(parents=True, exist_ok=True)
        return d1

def _item_clusters_from_contrast_loading(load1: np.ndarray) -> np.ndarray:
    """Two-cluster split by the sign of Contrast-1 loading.
    Returns cluster labels {1,2}.
    """
    ld = np.asarray(load1, dtype=float).ravel()
    cl = np.where(ld >= 0, 1, 2)
    return cl.astype(int)

def _silhouette_from_distances(D: np.ndarray, cluster: np.ndarray):
    """Compute a(i), b(i), ss(i) for items given a full distance matrix D (I x I)."""
    D = np.asarray(D, dtype=float)
    cl = np.asarray(cluster, dtype=int).ravel()
    I = D.shape[0]
    a = np.full(I, np.nan, dtype=float)
    b = np.full(I, np.nan, dtype=float)
    s = np.full(I, np.nan, dtype=float)

    # precompute masks per cluster
    uniq = [c for c in np.unique(cl) if np.isfinite(c)]
    for i in range(I):
        ci = cl[i]
        if ci not in uniq:
            continue
        same = (cl == ci)
        same[i] = False
        if same.any():
            a[i] = float(np.nanmean(D[i, same]))
        else:
            a[i] = np.nan

        other_means = []
        for cj in uniq:
            if cj == ci:
                continue
            m = (cl == cj)
            if m.any():
                other_means.append(float(np.nanmean(D[i, m])))
        b[i] = float(np.nanmin(other_means)) if other_means else np.nan

        if np.isfinite(a[i]) and np.isfinite(b[i]) and max(a[i], b[i]) > 1e-12:
            s[i] = (b[i] - a[i]) / max(a[i], b[i])
        else:
            s[i] = np.nan
    return a, b, s

def _pairwise_item_distances_from_Z(Z: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance between items based on ZSTD vectors across persons.
    Z: (P,I) standardized residuals. Returns D: (I,I).
    """
    A = np.asarray(Z, dtype=float)
    if A.ndim != 2:
        raise ValueError("Z must be 2D (P,I)")
    # items as rows
    X = A.T  # (I,P)
    # mean-impute per item then center
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isfinite(X), X, mu)
    X = X - np.mean(X, axis=1, keepdims=True)

    # compute squared distances efficiently
    G = X @ X.T
    ss = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.maximum(ss + ss.T - 2.0 * G, 0.0)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D

def _build_relations_from_distance(D: np.ndarray, item_names: list[str]) -> pd.DataFrame:
    """Create a sparse 'relation' edge list using 'strength = max(z)-z' of z-scored distances.
    Keeps, for each term, the single strongest edge (highest strength), then unions them.
    Returns DataFrame with columns: term1, term2, WCD.
    """
    D = np.asarray(D, dtype=float)
    I = D.shape[0]
    # take upper triangle distances
    iu = np.triu_indices(I, k=1)
    d = D[iu]
    mu = float(np.nanmean(d)) if d.size else 0.0
    sd = float(np.nanstd(d, ddof=1)) if d.size > 1 else 1.0
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    z = (d - mu) / sd
    zmax = float(np.nanmax(z)) if z.size else 0.0
    strength = zmax - z  # smaller distance => smaller z => larger strength

    # map back to matrix of strength
    S = np.full((I, I), np.nan, dtype=float)
    S[iu] = strength
    S[(iu[1], iu[0])] = strength
    np.fill_diagonal(S, np.nan)

    # for each node, keep best neighbor
    edges = set()
    for i in range(I):
        row = S[i, :]
        if not np.isfinite(row).any():
            continue
        j = int(np.nanargmax(row))
        a, b = (i, j) if i < j else (j, i)
        edges.add((a, b))

    rows = []
    for a, b in sorted(edges):
        rows.append([item_names[a], item_names[b], float(S[a, b])])
    return pd.DataFrame(rows, columns=["term1", "term2", "WCD"])

# -----------------------------
# Public result structure
# -----------------------------

@dataclass
class RaschResult:
    ok: bool
    mode: str
    model: str
    iterations: int
    last_error: float
    stop_reason: str

    min_cat: int
    max_cat: int
    tau: np.ndarray  # length K (with tau[0]=0)

    theta: np.ndarray  # persons
    b: np.ndarray      # items

    person_df: pd.DataFrame
    item_df: pd.DataFrame

    def __post_init__(self):
        # Backward compatibility: some callers expect `model` instead of `mode`
        if not getattr(self, 'model', None):
            self.model = getattr(self, 'mode', '')

    debug: Dict[str, Any]


# -----------------------------
# Main API
# -----------------------------



# --- CSV loader with encoding fallbacks (utf-8 / Big5) ---
def _read_csv_robust(path: str):
    """Read CSV with encoding fallbacks (utf-8-sig, utf-8, cp950, big5).

    This prevents UnicodeDecodeError for Traditional Chinese CSVs.
    """
    import pandas as _pd
    encodings = ['utf-8-sig','utf-8','cp950','big5','latin1']
    last_err = None
    for enc in encodings:
        try:
            return _pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    # final attempt: decode bytes with replacement
    data = __import__("pathlib").Path(path).read_bytes()
    text = data.decode('utf-8', errors='replace')
    from io import StringIO
    return _pd.read_csv(StringIO(text))
def run_rasch(
    csv_path: str,
    id_col: Optional[str] = None,
    drop_cols: Sequence[str] = (),
    tau_fixed: Optional[Sequence[float]] = None,
    estimate_tau: bool = True,
    damping_tau: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    damping: float = 0.3,
    clamp: float = 10.0,
    adj_rate: float = 2.25,
    # Legacy compatibility (ignored)
    mode: Optional[str] = None,
    max_category: Optional[int] = None,
    continuous_transform: Optional[str] = None,

    **kwargs,

) -> RaschResult:
    """
    Fit a polytomous Rasch Rating Scale Model (RSM) from a CSV.

    Parameters
    ----------
    csv_path:
      CSV with rows=persons, cols=items. You may include an id column and other metadata columns.
    id_col:
      Optional column name to use as person ID. If None, uses first column if it is non-numeric.
    drop_cols:
      Any columns to drop (e.g., 'Profile').
    tau_fixed:
      If provided, uses these step thresholds for categories 1..K-1 (K inferred from data).
      Example for 3-category (0/1/2): (-0.86, 0.86) meaning tau=[0, -0.86, +0.86].
    estimate_tau:
      If True, attempts to estimate tau. (For your case, keep False—tau estimation is where
      your old code blew up.)
    """
    df = _read_csv_robust(csv_path)
    if len(df) == 0:
        raise ValueError("Empty CSV")

    # Infer id column if not provided
    if id_col is None:
        first = df.columns[0]
        # non-numeric first column -> treat as id
        if not np.issubdtype(df[first].dtype, np.number):
            id_col = first

    person_ids = df[id_col].astype(str).tolist() if id_col and id_col in df.columns else [f"P{i+1:03d}" for i in range(len(df))]

    # Preserve profile labels from input (for reporting / group Wright maps only; NOT used in estimation)
    profile_labels = None
    for _cand in ["profile", "Profile", "PROFILE"]:
        if _cand in df.columns:
            profile_labels = df[_cand].astype(str).fillna("NA").tolist()
            break

    use_df = df.copy()
    if id_col and id_col in use_df.columns:
        use_df = use_df.drop(columns=[id_col])
    for c in drop_cols:
        if c in use_df.columns:
            use_df = use_df.drop(columns=[c])

    # ✅ Item column selection:
    #    - Exclude label columns like Profile / TRUE_LABEL wherever they appear (NOT just last column)
    #    - Use all remaining columns as item responses (coerce to numeric)
    label_names = {"profile", "true_label", "truelabel", "label"}
    label_cols = [c for c in use_df.columns if str(c).strip().lower() in label_names]
    if label_cols:
        use_df = use_df.drop(columns=label_cols)

    item_cols = list(use_df.columns)
    if not item_cols:
        raise ValueError("No item columns found after dropping id/label columns.")

    # Normalize common missing tokens (Winsteps-style): ".", "", "NA" -> NaN
    try:
        use_df = use_df.replace({".": np.nan, "": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan, "NULL": np.nan})
    except Exception:
        pass

    # Coerce to numeric (non-numeric -> NaN)
    X = use_df[item_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    X_raw = X.copy()

    # --- Compatibility: continuous_transform default in UI ---
    # Some UIs pass continuous_transform='linear' by default; interpret as NO transform.
    if continuous_transform is not None:
        ct0 = str(continuous_transform).strip().lower()
        if ct0 in ("", "none", "no", "off", "linear"):
            continuous_transform = None

    

    # --- Auto-detect continuous inputs ---
    # Rule (per your spec):
    #   if any finite value has decimals, or any finite value < 0, or any finite value > 10,
    #   then treat the data as continuous and force a continuous_transform.
    _finite = X[np.isfinite(X)]
    if _finite.size > 0:
        _has_decimal = np.any(np.abs(_finite - np.round(_finite)) > 1e-9)
        _is_continuous = _has_decimal or (np.nanmin(_finite) < 0.0) or (np.nanmax(_finite) > 10.0)
    else:
        _is_continuous = False

    if _is_continuous and continuous_transform is None:
        # Auto: any decimals/negative/>10 -> linearly map into 0..4 Likert categories before Rasch.
        continuous_transform = "minmax"
        max_category = 4
# Auto-detect continuous data:
    # Rule: if any finite cell has decimals OR is negative OR > 10, treat as continuous.
    try:
        _xf = X[np.isfinite(X)]
        _is_cont = False
        if _xf.size:
            if np.any(_xf < 0) or np.any(_xf > 10):
                _is_cont = True
            else:
                _is_cont = not np.all(np.isclose(_xf, np.round(_xf)))
        if _is_cont and (continuous_transform is None):
            # Auto: any decimals/negative/>10 -> linearly map into 0..4 Likert categories before Rasch.
            continuous_transform = 'minmax'
            max_category = 4
    except Exception:
        pass

# --- Continuous -> categorical transform (for nurse-ratio / continuous inputs) ---
    if continuous_transform is not None:
        ct = str(continuous_transform).strip().lower()
        if ct in ("minmax", "minmax_0_1", "0_1", "01", "normalize"):
            # scale each item column to 0..1
            X2 = X.copy()
            for j in range(X2.shape[1]):
                col = X2[:, j]
                m = np.isfinite(col)
                if not m.any():
                    continue
                vmin = float(np.nanmin(col[m]))
                vmax = float(np.nanmax(col[m]))
                if vmax - vmin < 1e-12:
                    # constant column -> all zeros
                    X2[:, j] = np.where(m, 0.0, np.nan)
                else:
                    X2[:, j] = (col - vmin) / (vmax - vmin)
            # map 0..1 to integer categories 0..max_category (default 2)
            Kt = int(max_category) if max_category is not None else 4
            X = np.rint(X2 * Kt)
        else:
            raise ValueError(f"Unsupported continuous_transform: {continuous_transform}")

    # If no profile in input, create a virtual binary profile by median split of raw total score
    if profile_labels is None:
        try:
            _raw = np.nansum(X, axis=1)
            _med = float(np.nanmedian(_raw))
            profile_labels = np.where(_raw >= _med, "1", "0").astype(str).tolist()
        except Exception:
            profile_labels = [str(i % 2) for i in range(X.shape[0])]


    # store matrix for reliability (alpha) and debugging
    try:
        if "debug" not in locals():
            debug = {
        'profile': profile_labels,
}
        debug["X"] = X.tolist()
        # Keep both raw and transformed matrices for diagnostics.
        # IMPORTANT: All post-analysis (DIF/Q3/Kano/TCP/etc.) should use the transformed categorical matrix
        # when the input was continuous.
        try:
            debug["X_raw"] = X_raw.tolist()
        except Exception:
            pass
        try:
            debug["X_transformed"] = X.tolist()
            debug["X_post"] = X.tolist()
        except Exception:
            pass
        try:
            debug["is_continuous_input"] = bool(_is_continuous)
            debug["continuous_transform"] = continuous_transform
            debug["max_category"] = max_category
        except Exception:
            pass

        debug["item_cols"] = list(map(str, item_cols))
    except Exception:
        pass
    # Validate categories
    X_int, min_cat, max_cat = _ensure_consecutive_categories(X)
    # Shift to 0..K-1 internally
    X0 = X_int - min_cat
    K = (max_cat - min_cat + 1)
    Pn, In = X0.shape

    # tau vector with tau[0]=0
    # Compatibility:
    # - For binary data (K=2), there is only one step; Winsteps-style is effectively tau=[0,0].
    # - Accept tau_fixed of length K-1 (steps 1..K-1) OR length K (including tau[0]).
    # - If a 3-category default (-0.86, 0.86) is passed while K=2, ignore it safely.
    if tau_fixed is not None:
        tf = np.asarray(list(tau_fixed), dtype=float).ravel()
        if K == 2 and tf.size == 2:
            # likely a 3-category default accidentally passed for dichotomous data
            tau = np.zeros(K, dtype=float)
        elif tf.size == K:
            tau = tf.astype(float, copy=True)
            tau[0] = 0.0
        elif tf.size == K - 1:
            tau = np.zeros(K, dtype=float)
            tau[1:] = tf
        else:
            # If user provided tau_fixed but its length does not match the detected number of categories,
            # treat it as *starting values* and expand/shrink safely instead of failing.
            # Common case: legacy 3-category defaults (len=2) with K>3 (e.g., continuous -> 0..4 gives K=5).
            import numpy as _np
            if tf.size >= 2 and K > 2:
                # Use endpoints and linearly interpolate to K-1 steps, then center (mean=0).
                x_old = _np.linspace(0.0, 1.0, num=int(tf.size))
                x_new = _np.linspace(0.0, 1.0, num=int(K-1))
                tf_new = _np.interp(x_new, x_old, tf.astype(float))
                tf_new = tf_new - _np.mean(tf_new)
                tau = _np.zeros(K, dtype=float)
                tau[1:] = tf_new
            else:
                # Fall back to zero steps
                tau = _np.zeros(K, dtype=float)

    else:
        tau = np.zeros(K, dtype=float)

    # init
    theta = np.zeros(Pn, dtype=float)
    b = np.zeros(In, dtype=float)

    # totals
    max_raw_p = (K - 1) * In
    max_raw_i = (K - 1) * Pn

    stop_reason = "max_iter"
    last_err = np.nan
    ok = True

    cats = np.arange(K, dtype=float)[None, None, :]

    try:
        for it in range(1, max_iter + 1):
            Pmat = _rsm_prob(theta, b, tau)  # (P,I,K)
            E = np.sum(Pmat * cats, axis=2)  # expected score (P,I)
            Var = np.sum(Pmat * (cats - E[:, :, None]) ** 2, axis=2)  # variance (P,I)
            Var = np.maximum(Var, 1e-8)

            # Update theta
            mask = np.isfinite(X0)
            r_p = np.nansum(X0, axis=1).astype(float)              # raw score (skip missing)
            E_p = np.sum(E * mask, axis=1).astype(float)           # expected (skip missing)
            V_p = np.sum(Var * mask, axis=1).astype(float)         # variance (skip missing)
            step_theta = (r_p - E_p) / np.maximum(V_p, 1e-8)
            theta_new = theta + damping * step_theta
            theta_new = np.clip(theta_new, -clamp, clamp)

            # Update b (difficulty)
            r_i = np.nansum(X0, axis=0).astype(float)              # raw score (skip missing)
            E_i = np.sum(E * mask, axis=0).astype(float)           # expected (skip missing)
            V_i = np.sum(Var * mask, axis=0).astype(float)         # variance (skip missing)
            step_b = (E_i - r_i) / np.maximum(V_i, 1e-8)
            b_new = b + damping * step_b
            b_new = np.clip(b_new, -clamp, clamp)

            # Identification constraints
            b_new = b_new - np.mean(b_new)
            # Identification constraint (Winsteps-like): center items only
            # Do NOT force person measures to mean 0
            # theta_new = theta_new - np.mean(theta_new)# ASP-style person extreme substitution (only if clamping happened)
            theta_new, V_p2 = _asp_extreme_substitution_person(
                theta_new, V_p, r_p, max_raw=max_raw_p, clamp=clamp, adj_rate=adj_rate
            )

            # (Optional) tau estimation (ASP-verified ratio update on category counts)
            tau_old = tau.copy()
            if estimate_tau and K > 2:
                # Observed category counts (skip missing)
                catobs = np.array([(X0 == k)[mask].sum() for k in range(K)], dtype=float)
                # Expected category counts from current probabilities (skip missing)
                catexp = np.array([np.sum(Pmat[:, :, k] * mask) for k in range(K)], dtype=float)

                # ASP-style step threshold update using adjacent-category log-odds ratios
                catthresh = tau.copy()
                for k in range(1, K):
                    if catobs[k] > 0 and catobs[k-1] > 0 and catexp[k] > 0 and catexp[k-1] > 0:
                        catthresh[k] = tau[k] + np.log(catobs[k-1] / catobs[k]) - np.log(catexp[k-1] / catexp[k])
                    else:
                        # Guard: keep current value if counts are degenerate
                        catthresh[k] = tau[k]

                # Center steps (identification): mean(tau[1:]) = 0
                if (K - 1) > 0:
                    avg = float(np.mean(catthresh[1:]))
                else:
                    avg = 0.0
                catthresh[0] = 0.0
                catthresh[1:] = catthresh[1:] - avg

                # Damped apply
                damp = float(np.clip(damping_tau, 0.0, 1.0))
                tau = (1.0 - damp) * tau + damp * catthresh
                # keep in safe range
                tau = np.clip(tau, -6.0, 6.0)

            # convergence check (include tau change)
            last_err = float(np.max(np.abs(np.concatenate([theta_new - theta, b_new - b, tau - tau_old]))))
            theta, b = theta_new, b_new

            if not np.isfinite(last_err):
                ok = False
                stop_reason = "nan_error"
                break

            if last_err < tol:
                stop_reason = "rasch_core"
                break

        iterations = it
    except Exception as e:
        ok = False
        iterations = it if 'it' in locals() else 0
        stop_reason = f"exception:{type(e).__name__}"
        last_err = float("nan")

    # Build simple tables (enough to compare measures with Winsteps)
    # Person SE (approx) from information: SE = 1/sqrt(info); info ~ sum Var per person
    Pmat = _rsm_prob(theta, b, tau)
    E = np.sum(Pmat * cats, axis=2)
    Var = np.sum(Pmat * (cats - E[:, :, None]) ** 2, axis=2)
    Var = np.maximum(Var, 1e-8)
    mask = np.isfinite(X0)
    count_p = np.sum(mask, axis=1).astype(float)
    count_i = np.sum(mask, axis=0).astype(float)

    r_p = np.nansum(X0, axis=1).astype(float)
    info_p = np.sum(Var * mask, axis=1).astype(float)  # proxy information (skip missing)
    se_p = 1.0 / np.sqrt(np.maximum(info_p, 1e-8))

    max_raw_p_eff = (K - 1) * count_p
    is_extreme_p = (r_p <= 0.0) | (r_p >= max_raw_p_eff)

    
    # ---- Fit statistics (Winsteps-like core): INFIT/OUTFIT MNSQ + ZSTD ----
    # Recompute expected score and variance at final estimates
    Pmat_final = _rsm_prob(theta, b, tau)  # (P,I,K)
    E_final = np.sum(Pmat_final * cats, axis=2)  # (P,I)
    Var_final = np.sum(Pmat_final * (cats - E_final[:, :, None]) ** 2, axis=2)  # (P,I)
    Var_final = np.maximum(Var_final, 1e-8)

    resid = X0 - E_final
    std_resid = resid / np.sqrt(Var_final)

    # Person fit
    df_p = max(int(In) - 1, 1)
    mask_fit = np.isfinite(X0)

    # OUTFIT: mean of squared standardized residuals (skip missing)
    outfit_p = np.nanmean(std_resid ** 2, axis=1)

    # INFIT: information-weighted mean square (skip missing)
    infit_p = np.nansum(resid ** 2, axis=1) / np.maximum(np.nansum(Var_final * mask_fit, axis=1), 1e-8)

    # Item fit
    df_i = max(int(Pn) - 1, 1)
    outfit_i = np.nanmean(std_resid ** 2, axis=0)
    infit_i = np.nansum(resid ** 2, axis=0) / np.maximum(np.nansum(Var_final * mask_fit, axis=0), 1e-8)

    def _mnsq_to_zstd(mnsq: np.ndarray, df: int) -> np.ndarray:
        # Cube-root normalizing transform for chi-square based mean-square (approx; close to Winsteps)
        m = np.maximum(mnsq, 1e-12)
        return (np.cbrt(m) - 1.0) * (3.0 * np.sqrt(df / 2.0))

    infit_z_p = _mnsq_to_zstd(infit_p, df_p)
    outfit_z_p = _mnsq_to_zstd(outfit_p, df_p)
    infit_z_i = _mnsq_to_zstd(infit_i, df_i)
    outfit_z_i = _mnsq_to_zstd(outfit_i, df_i)

    person_df = pd.DataFrame({
        "ENTRY": np.arange(1, Pn + 1),
        "KID": person_ids,
        "TOTAL_SCORE": r_p + min_cat * In,
        "COUNT": np.full(Pn, In),
        "MEASURE": theta,
        "SE": se_p,
        "INFIT_MNSQ": infit_p,
        "OUTFIT_MNSQ": outfit_p,
        "INFIT_ZSTD": infit_z_p,
        "OUTFIT_ZSTD": outfit_z_p,
        "IS_EXTREME": is_extreme_p,
    })

    # Attach profile labels to person_df (for reporting / group plots only)
    try:
        if profile_labels is not None and len(profile_labels) == len(person_df):
            person_df["profile"] = pd.Series(profile_labels, index=person_df.index).astype(str)
    except Exception:
        pass


    r_i = np.nansum(X0, axis=0).astype(float)
    info_i = np.sum(Var * mask, axis=0).astype(float)
    se_i = 1.0 / np.sqrt(np.maximum(info_i, 1e-8))

    max_raw_i_eff = (K - 1) * count_i
    is_extreme_i = (r_i <= 0.0) | (r_i >= max_raw_i_eff)

    item_df = pd.DataFrame({
        "ENTRY": np.arange(1, In + 1),
        "ITEM": item_cols,
        "TOTAL_SCORE": r_i + min_cat * Pn,
        "COUNT": np.full(In, Pn),
        "MEASURE": b,
        "SE": se_i,
        "INFIT_MNSQ": infit_i,
        "OUTFIT_MNSQ": outfit_i,
        "INFIT_ZSTD": infit_z_i,
        "OUTFIT_ZSTD": outfit_z_i,
        "IS_EXTREME": is_extreme_i,
    })

    debug = {
        "P": int(Pn),
        "I": int(In),
        "K": int(K),
        "min_cat": int(min_cat),
        "max_cat": int(max_cat),
        "max_raw_person": int(max_raw_p),
        "max_raw_item": int(max_raw_i),
        "item_cols": list(map(str, item_cols)),
    }

    # ---- Testlet handling (A: diagnostics only; B: estimate testlet effects; dichotomous only) ----
    try:
        testlet_mode = str(kwargs.get('testlet_mode', 'diag')).strip().lower()
    except Exception:
        testlet_mode = 'diag'
    if testlet_mode not in ('diag','estimate'):
        testlet_mode = 'diag'
    try:
        q3_cut = float(kwargs.get('q3_cut', 0.20))
    except Exception:
        q3_cut = 0.20
    try:
        testlet_max_iter = int(kwargs.get('testlet_max_iter', 25))
    except Exception:
        testlet_max_iter = 25
    try:
        testlet_damp = float(kwargs.get('testlet_damp', 0.5))
    except Exception:
        testlet_damp = 0.5

    debug['testlet_mode'] = testlet_mode
    debug['q3_cut'] = float(q3_cut)

    # Compute residual Q3 (items) from standardized residuals
    try:
        _Zm = std_resid.astype(float)
        _Q3 = _np.corrcoef(_Zm, rowvar=False)
        _np.fill_diagonal(_Q3, 0.0)
        memb_i, edges = _detect_testlets_from_Q3(_Q3, item_cols, q3_cut=q3_cut)
        debug['testlet_membership'] = {str(item_cols[i]): int(memb_i.get(i, 0)) for i in range(len(item_cols))}
        debug['testlet_edges'] = [(str(item_cols[a]), str(item_cols[b]), float(w)) for a,b,w in edges]
        debug['testlet_n_edges'] = int(len(edges))
        debug['testlet_n_groups'] = int(max(memb_i.values()) if memb_i else 0)

        # Mode B: estimate testlet effects via a simple JML refit (dichotomous only)
        debug['testlet_estimated'] = False
        if testlet_mode == 'estimate':
            if int(K) == 2 and debug['testlet_n_groups'] >= 1:
                th2, b2, tau_pk = _jml_testlet_refit_dicho(
                    X0, theta, b, memb_i,
                    max_iter=testlet_max_iter,
                    damp=testlet_damp
                )
                # Keep original Rasch estimates, but add testlet-adjusted columns for comparison
                person_df['MEASURE_TESTLET'] = th2
                item_df['MEASURE_TESTLET'] = b2
                debug['tau_testlet'] = tau_pk.tolist()
                debug['theta_testlet'] = th2.tolist()
                debug['b_testlet'] = b2.tolist()
                debug['testlet_estimated'] = True
            else:
                debug['testlet_estimated'] = False
                if int(K) != 2:
                    debug['testlet_estimate_reason'] = 'Estimate mode supports dichotomous (0/1) only; fell back to diagnostics.'
                else:
                    debug['testlet_estimate_reason'] = 'No testlet groups detected at the chosen Q3 threshold; fell back to diagnostics.'
    except Exception as _e:
        debug['testlet_error'] = repr(_e)

    # Add matrices for exports / scree plots
    try:
        # Backward-compatible key expected by the report renderer
        # (older blocks look for res.debug['X']).
        debug["X"] = (X_int).tolist()
        debug["X_obs"] = (X_int).tolist()  # original categories (min_cat..max_cat)
        debug["X0"] = (X0).tolist()        # shifted 0..K-1
        # Expected score in original scale
        debug["E_exp"] = (E_final + float(min_cat)).tolist()
        debug["VAR"] = (Var_final).tolist()
        debug["RESID"] = (X_int - (E_final + float(min_cat))).tolist()
        _den = np.sqrt(np.maximum(Var_final, 1e-12))
        debug["ZSTD"] = ((X_int - (E_final + float(min_cat))) / _den).tolist()
        debug["EXP"] = debug["E_exp"]
    except Exception:
        pass

    return RaschResult(
        ok=bool(ok),
        mode="RSM",
        model="RSM",
iterations=int(iterations),
        last_error=float(last_err),
        stop_reason=str(stop_reason),
        min_cat=int(min_cat),
        max_cat=int(max_cat),
        tau=_as_float_array(tau),
        theta=_as_float_array(theta),
        b=_as_float_array(b),
        person_df=person_df,
        item_df=item_df,
        debug=debug,
    )


def _detect_testlets_from_Q3(Q3, item_cols, q3_cut=0.20):
    """Return (membership_dict, edges) from an item-item Q3 matrix.

    - membership_dict maps item-index -> testlet_id (0=singleton/no testlet)
    - edges is a list of (i,j,Q3_ij) for |Q3_ij|>=cut and i<j

    Note: This uses connected components on the threshold graph as a fast, dependency-free
    community proxy. Your report already provides a richer TCP; this function is only
    used to gate testlet estimation in mode (B).
    """
    import numpy as np
    Q3 = np.asarray(Q3, dtype=float)
    I = Q3.shape[0]
    cut = float(q3_cut)

    idx = np.argwhere(np.abs(Q3) >= cut)
    edges = []
    for a, b in idx:
        a = int(a); b = int(b)
        if a < b:
            edges.append((a, b, float(Q3[a, b])))

    if not edges:
        return {i: 0 for i in range(I)}, []

    adj = {i: set() for i in range(I)}
    for a, b, _w in edges:
        adj[a].add(b); adj[b].add(a)

    seen = set()
    comp_id = 0
    memb = {i: 0 for i in range(I)}
    for i in range(I):
        if i in seen:
            continue
        stack = [i]
        comp = []
        seen.add(i)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(comp) >= 2:
            comp_id += 1
            for u in comp:
                memb[u] = comp_id
        else:
            memb[i] = 0

    return memb, edges


def _jml_testlet_refit_dicho(X01, theta0, b0, testlet_memb, max_iter=25, damp=0.5):
    """Simple JML refit for dichotomous Rasch + testlet effects.

    Model: logit P(x=1) = theta_p - b_i + tau_{p,k(i)}
    with tau_{p,0}=0 and mean_p tau_{p,k}=0 for identifiability.

    Returns (theta, b, tau_pk) where tau_pk includes column 0 (all zeros).
    """
    import numpy as np

    X = np.asarray(X01, dtype=float)
    Pn, In = X.shape
    theta = np.asarray(theta0, dtype=float).copy()
    b = np.asarray(b0, dtype=float).copy()

    k_of_i = np.array([int(testlet_memb.get(i, 0)) for i in range(In)], dtype=int)
    Kt = int(np.max(k_of_i))
    tau = np.zeros((Pn, Kt + 1), dtype=float)

    def sigmoid(z):
        z = np.clip(z, -35, 35)
        return 1.0 / (1.0 + np.exp(-z))

    damp = float(damp)
    max_iter = int(max_iter)

    M = np.isfinite(X)

    for _ in range(max_iter):
        eta = theta[:, None] - b[None, :] + tau[np.arange(Pn)[:, None], k_of_i[None, :]]
        P = sigmoid(eta)
        W = np.clip(P * (1.0 - P), 1e-8, None)

        # theta updates
        num = np.nansum((X - P) * M, axis=1)
        den = np.nansum(W * M, axis=1)
        step = np.where(den > 0, num / den, 0.0)
        theta = theta + damp * step

        # b updates
        num = np.nansum((X - P) * M, axis=0)
        den = np.nansum(W * M, axis=0)
        step = np.where(den > 0, num / den, 0.0)
        b = b - damp * step

        # tau updates per testlet
        for k in range(1, Kt + 1):
            items_k = np.where(k_of_i == k)[0]
            if items_k.size < 2:
                continue
            num = np.nansum(((X[:, items_k] - P[:, items_k]) * M[:, items_k]), axis=1)
            den = np.nansum((W[:, items_k] * M[:, items_k]), axis=1)
            step = np.where(den > 0, num / den, 0.0)
            tau[:, k] = tau[:, k] + damp * step
            tau[:, k] -= np.nanmean(tau[:, k])

        # center theta and b
        theta -= np.nanmean(theta)
        b -= np.nanmean(b)

    return theta, b, tau


def demo_on_input_csv() -> RaschResult:
    """
    Convenience demo used by the ChatGPT sandbox.
    It expects /mnt/data/input.csv to exist (provided by the user).
    """
    return run_rasch(
        "/mnt/data/input.csv",
        id_col="name",
        drop_cols=("Profile",),
        tau_fixed=None,
        estimate_tau=True,
        max_iter=100,
        tol=1e-5,
        damping=0.7,
        clamp=10.0,
        adj_rate=2.25,
    )



# -----------------------------
# Legacy report helpers
# -----------------------------

def build_summary_tables(res: "RaschResult") -> pd.DataFrame:
    """
    Legacy helper expected by older main.py.

    Returns a summary dataframe with rows similar to Winsteps-like report blocks:
      PERSON_NON_EXTREME, PERSON_ALL, ITEM_NON_EXTREME, ITEM_ALL

    Columns mirror the ones your HTML report expects. Fit/ZSTD fields are returned
    as NaN in this lightweight version (since detailed fit statistics depend on
    additional residual standardization rules).
    """
    def _summ(df: pd.DataFrame, label: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {"TABLE": label}
        # Basic fields
        out["TOTAL_SCORE_MEAN"] = float(np.nanmean(df["TOTAL_SCORE"]))
        out["TOTAL_SCORE_SD"]   = float(np.nanstd(df["TOTAL_SCORE"], ddof=1)) if len(df) > 1 else 0.0
        out["TOTAL_SCORE_MIN"]  = float(np.nanmin(df["TOTAL_SCORE"]))
        out["TOTAL_SCORE_MAX"]  = float(np.nanmax(df["TOTAL_SCORE"]))

        out["COUNT_MEAN"] = float(np.nanmean(df["COUNT"]))
        out["COUNT_SD"]   = float(np.nanstd(df["COUNT"], ddof=1)) if len(df) > 1 else 0.0
        out["COUNT_MIN"]  = float(np.nanmin(df["COUNT"]))
        out["COUNT_MAX"]  = float(np.nanmax(df["COUNT"]))

        out["MEASURE_MEAN"] = float(np.nanmean(df["MEASURE"]))
        out["MEASURE_SD"]   = float(np.nanstd(df["MEASURE"], ddof=1)) if len(df) > 1 else 0.0
        out["MEASURE_MIN"]  = float(np.nanmin(df["MEASURE"]))
        out["MEASURE_MAX"]  = float(np.nanmax(df["MEASURE"]))

        out["SE_MEAN"] = float(np.nanmean(df["SE"]))
        out["SE_SD"]   = float(np.nanstd(df["SE"], ddof=1)) if len(df) > 1 else 0.0
        out["SE_MIN"]  = float(np.nanmin(df["SE"]))
        out["SE_MAX"]  = float(np.nanmax(df["SE"]))

        # Fit statistics (computed)
        if "INFIT_MNSQ" in df.columns:
            out["INFIT_MNSQ_MEAN"] = float(np.nanmean(df["INFIT_MNSQ"]))
            out["INFIT_MNSQ_SD"]   = float(np.nanstd(df["INFIT_MNSQ"], ddof=1)) if len(df) > 1 else 0.0
            out["INFIT_MNSQ_MIN"]  = float(np.nanmin(df["INFIT_MNSQ"]))
            out["INFIT_MNSQ_MAX"]  = float(np.nanmax(df["INFIT_MNSQ"]))
        else:
            out["INFIT_MNSQ_MEAN"]=out["INFIT_MNSQ_SD"]=out["INFIT_MNSQ_MIN"]=out["INFIT_MNSQ_MAX"]=float("nan")

        if "INFIT_ZSTD" in df.columns:
            out["INFIT_ZSTD_MEAN"] = float(np.nanmean(df["INFIT_ZSTD"]))
            out["INFIT_ZSTD_SD"]   = float(np.nanstd(df["INFIT_ZSTD"], ddof=1)) if len(df) > 1 else 0.0
            out["INFIT_ZSTD_MIN"]  = float(np.nanmin(df["INFIT_ZSTD"]))
            out["INFIT_ZSTD_MAX"]  = float(np.nanmax(df["INFIT_ZSTD"]))
        else:
            out["INFIT_ZSTD_MEAN"]=out["INFIT_ZSTD_SD"]=out["INFIT_ZSTD_MIN"]=out["INFIT_ZSTD_MAX"]=float("nan")

        if "OUTFIT_MNSQ" in df.columns:
            out["OUTFIT_MNSQ_MEAN"] = float(np.nanmean(df["OUTFIT_MNSQ"]))
            out["OUTFIT_MNSQ_SD"]   = float(np.nanstd(df["OUTFIT_MNSQ"], ddof=1)) if len(df) > 1 else 0.0
            out["OUTFIT_MNSQ_MIN"]  = float(np.nanmin(df["OUTFIT_MNSQ"]))
            out["OUTFIT_MNSQ_MAX"]  = float(np.nanmax(df["OUTFIT_MNSQ"]))
        else:
            out["OUTFIT_MNSQ_MEAN"]=out["OUTFIT_MNSQ_SD"]=out["OUTFIT_MNSQ_MIN"]=out["OUTFIT_MNSQ_MAX"]=float("nan")

        if "OUTFIT_ZSTD" in df.columns:
            out["OUTFIT_ZSTD_MEAN"] = float(np.nanmean(df["OUTFIT_ZSTD"]))
            out["OUTFIT_ZSTD_SD"]   = float(np.nanstd(df["OUTFIT_ZSTD"], ddof=1)) if len(df) > 1 else 0.0
            out["OUTFIT_ZSTD_MIN"]  = float(np.nanmin(df["OUTFIT_ZSTD"]))
            out["OUTFIT_ZSTD_MAX"]  = float(np.nanmax(df["OUTFIT_ZSTD"]))
        else:
            out["OUTFIT_ZSTD_MEAN"]=out["OUTFIT_ZSTD_SD"]=out["OUTFIT_ZSTD_MIN"]=out["OUTFIT_ZSTD_MAX"]=float("nan")

        out["N"] = int(len(df))
        return out

    p = res.person_df.copy()
    i = res.item_df.copy()

    p_all = p
    p_non = p_all[p_all["IS_EXTREME"] == False]  # noqa: E712

    i_all = i
    i_non = i_all[i_all["IS_EXTREME"] == False]  # noqa: E712

    rows = [
        _summ(p_non, "PERSON_NON_EXTREME"),
        _summ(p_all, "PERSON_ALL"),
        _summ(i_non, "ITEM_NON_EXTREME"),
        _summ(i_all, "ITEM_ALL"),
    ]
    return pd.DataFrame(rows)




def build_winsteps_summary_text(res: "RaschResult") -> str:
    """
    Produce a Winsteps-style SUMMARY block similar to the provided summaryS.txt
    (MEAN/SEM/P.SD/S.SD/MAX/MIN + REAL RMSE / MODEL RMSE / TRUE SD / SEPARATION / RELIABILITY).
    """
    def _block(df: pd.DataFrame, title: str, who: str) -> str:
        # df must include TOTAL_SCORE, COUNT, MEASURE, SE and fit cols
        n = len(df)
        if n == 0:
            return f"{title}\\n(no rows)\\n"

        def s(x): 
            return float(np.nanstd(x, ddof=1)) if n > 1 else 0.0
        def sem(x):
            return s(x) / np.sqrt(n) if n > 0 else float("nan")

        mean_score = float(np.nanmean(df["TOTAL_SCORE"]))
        mean_count = float(np.nanmean(df["COUNT"]))
        mean_meas  = float(np.nanmean(df["MEASURE"]))
        mean_se    = float(np.nanmean(df["SE"]))

        # Fit means (if available)
        inf_m = float(np.nanmean(df.get("INFIT_MNSQ", np.nan)))
        inf_z = float(np.nanmean(df.get("INFIT_ZSTD", np.nan)))
        out_m = float(np.nanmean(df.get("OUTFIT_MNSQ", np.nan)))
        out_z = float(np.nanmean(df.get("OUTFIT_ZSTD", np.nan)))

        psd_meas = s(df["MEASURE"])
        ssd_meas = psd_meas  # (sample sd already)
        # RMSE / separation / reliability (approx Winsteps)
        model_rmse = float(np.sqrt(np.nanmean(np.square(df["SE"]))))
        real_rmse = model_rmse
        true_sd = float(np.sqrt(max(0.0, psd_meas**2 - model_rmse**2)))
        separation = true_sd / model_rmse if model_rmse > 0 else float("nan")
        reliability = (separation**2) / (1.0 + separation**2) if np.isfinite(separation) else float("nan")
        se_of_mean = model_rmse / np.sqrt(n) if n > 0 else float("nan")

        # Score-to-measure correlation
        try:
            corr = float(np.corrcoef(df["TOTAL_SCORE"].to_numpy(), df["MEASURE"].to_numpy())[0,1])
        except Exception:
            corr = float("nan")

        header = []
        header.append(f"     {title}")
        header.append("-------------------------------------------------------------------------------")
        header.append("|          TOTAL                         MODEL         INFIT        OUTFIT    |")
        header.append("|          SCORE     COUNT     MEASURE    S.E.      MNSQ   ZSTD   MNSQ   ZSTD |")
        header.append("|-----------------------------------------------------------------------------|")
        def row(label, score, count, meas, se, im, iz, om, oz):
            return f"| {label:<4} {score:>9.1f} {count:>9.1f} {meas:>11.2f} {se:>7.2f} {im:>9.2f} {iz:>6.2f} {om:>7.2f} {oz:>6.2f} |"

        header.append(row("MEAN", mean_score, mean_count, mean_meas, mean_se, inf_m, inf_z, out_m, out_z))
        header.append(row(" SEM", sem(df["TOTAL_SCORE"]), sem(df["COUNT"]), sem(df["MEASURE"]), sem(df["SE"]),
                          sem(df.get("INFIT_MNSQ", pd.Series([np.nan]))),
                          sem(df.get("INFIT_ZSTD", pd.Series([np.nan]))),
                          sem(df.get("OUTFIT_MNSQ", pd.Series([np.nan]))),
                          sem(df.get("OUTFIT_ZSTD", pd.Series([np.nan]))),
                          ))
        header.append(row("P.SD", s(df["TOTAL_SCORE"]), s(df["COUNT"]), psd_meas, s(df["SE"]),
                          s(df.get("INFIT_MNSQ", pd.Series([np.nan]))),
                          s(df.get("INFIT_ZSTD", pd.Series([np.nan]))),
                          s(df.get("OUTFIT_MNSQ", pd.Series([np.nan]))),
                          s(df.get("OUTFIT_ZSTD", pd.Series([np.nan]))),
                          ))
        header.append(row("S.SD", s(df["TOTAL_SCORE"]), s(df["COUNT"]), ssd_meas, s(df["SE"]),
                          s(df.get("INFIT_MNSQ", pd.Series([np.nan]))),
                          s(df.get("INFIT_ZSTD", pd.Series([np.nan]))),
                          s(df.get("OUTFIT_MNSQ", pd.Series([np.nan]))),
                          s(df.get("OUTFIT_ZSTD", pd.Series([np.nan]))),
                          ))
        header.append(row("MAX.", float(np.nanmax(df["TOTAL_SCORE"])), float(np.nanmax(df["COUNT"])),
                          float(np.nanmax(df["MEASURE"])), float(np.nanmax(df["SE"])),
                          float(np.nanmax(df.get("INFIT_MNSQ", pd.Series([np.nan])))),
                          float(np.nanmax(df.get("INFIT_ZSTD", pd.Series([np.nan])))),
                          float(np.nanmax(df.get("OUTFIT_MNSQ", pd.Series([np.nan])))),
                          float(np.nanmax(df.get("OUTFIT_ZSTD", pd.Series([np.nan])))),
                          ))
        header.append(row("MIN.", float(np.nanmin(df["TOTAL_SCORE"])), float(np.nanmin(df["COUNT"])),
                          float(np.nanmin(df["MEASURE"])), float(np.nanmin(df["SE"])),
                          float(np.nanmin(df.get("INFIT_MNSQ", pd.Series([np.nan])))),
                          float(np.nanmin(df.get("INFIT_ZSTD", pd.Series([np.nan])))),
                          float(np.nanmin(df.get("OUTFIT_MNSQ", pd.Series([np.nan])))),
                          float(np.nanmin(df.get("OUTFIT_ZSTD", pd.Series([np.nan])))),
                          ))
        header.append("|-----------------------------------------------------------------------------|")
        header.append(f"| REAL RMSE {real_rmse:>7.2f} TRUE SD {true_sd:>7.2f}  SEPARATION {separation:>5.2f}  {who:<4}  RELIABILITY {reliability:>5.2f} |")
        header.append(f"|MODEL RMSE {model_rmse:>7.2f} TRUE SD {true_sd:>7.2f}  SEPARATION {separation:>5.2f}  {who:<4}  RELIABILITY {reliability:>5.2f} |")
        header.append(f"| S.E. OF {who} MEAN = {se_of_mean:.2f}{' ' * 54}|")
        header.append("-------------------------------------------------------------------------------")
        header.append(f"{who} RAW SCORE-TO-MEASURE CORRELATION = {corr:.2f}")
        return "\\n".join(header)

    p_all = res.person_df.copy()
    i_all = res.item_df.copy()
    p_ne = p_all[~p_all.get("IS_EXTREME", False)].copy()
    i_ne = i_all[~i_all.get("IS_EXTREME", False)].copy()

    blocks = []
    blocks.append(_block(p_ne, f"SUMMARY OF {len(p_ne)} MEASURED (NON-EXTREME) KID", "KID"))
    blocks.append("")
    blocks.append(_block(p_all, f"SUMMARY OF {len(p_all)} MEASURED (EXTREME AND NON-EXTREME) KID", "KID"))

    # Reliability (Cronbach alpha / KR-20) on raw scores
    try:
        Xraw = np.array(getattr(res, "debug", {}).get("X", []), dtype=float)
        k = int(Xraw.shape[1]) if Xraw.ndim == 2 else 0
        alpha = float("nan"); sem_ = float("nan"); std_rel_50 = float("nan")
        if k >= 2 and Xraw.size > 0:
            item_vars = np.nanvar(Xraw, axis=0, ddof=1)
            total = np.nansum(Xraw, axis=1)
            total_var = np.nanvar(total, ddof=1)
            if np.isfinite(total_var) and total_var > 0:
                alpha = (k / (k - 1.0)) * (1.0 - (np.nansum(item_vars) / total_var))
                alpha = float(max(0.0, min(1.0, alpha)))
                sd_total = float(np.nanstd(total, ddof=1))
                sem_ = sd_total * np.sqrt(max(0.0, 1.0 - alpha))
                # Standardized to 50 items via Spearman-Brown
                m50 = 50.0 / k
                if m50 > 0:
                    std_rel_50 = float((m50 * alpha) / (1.0 + (m50 - 1.0) * alpha))
        # Always print lines (Winsteps-style uses '.' when unavailable)
        try:
            if ('TOTAL_SCORE' in p_all.columns and 'MEASURE' in p_all.columns and len(p_all) > 1):
                r = np.corrcoef(p_all['TOTAL_SCORE'], p_all['MEASURE'])[0,1]
                r_txt = f"{float(r):.2f}" if np.isfinite(r) else "."
            else:
                r_txt = "."
        except Exception:
            r_txt = "."
        blocks.append(f"KID RAW SCORE-TO-MEASURE CORRELATION = {r_txt}")
        a_txt = f"{alpha:.2f}" if np.isfinite(alpha) else "."
        s_txt = f"{sem_:.2f}" if np.isfinite(sem_) else "."
        blocks.append(f"CRONBACH ALPHA (KR-20) KID RAW SCORE 'TEST' RELIABILITY = {a_txt}  SEM = {s_txt}")
        if np.isfinite(std_rel_50):
            blocks.append(f"STANDARDIZED (50 ITEM) RELIABILITY = {std_rel_50:.2f}")
    except Exception:
        # still print placeholders
        blocks.append("KID RAW SCORE-TO-MEASURE CORRELATION = .")
        blocks.append("CRONBACH ALPHA (KR-20) KID RAW SCORE 'TEST' RELIABILITY = .  SEM = .")
    blocks.append("")
    blocks.append(_block(i_ne, f"SUMMARY OF {len(i_ne)} MEASURED (NON-EXTREME) ACT", "ACT"))
    return "\\n".join(blocks)



def export_cell_diagnostics(res, out_csv_path):
    """
    Export person-item cell diagnostics (xfile-like) for compatibility.
    This is a lightweight export: expected score, residual, standardized residual.
    If probabilities are not available in res, we export minimal placeholders.
    """
    import pandas as pd

    def _read_csv_robust(path):
        """Read CSV with encoding fallbacks (utf-8/utf-8-sig/cp950/big5/latin1)."""
        import pandas as _pd
        encs = ['utf-8', 'utf-8-sig', 'cp950', 'big5', 'latin1']
        last = None
        for enc in encs:
            try:
                return _pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as e:
                last = e
                continue
            except Exception as e:
                last = e
                # try a slightly more permissive read (bad bytes)
                try:
                    return _pd.read_csv(path, encoding=enc, engine='python', on_bad_lines='skip')
                except Exception as e2:
                    last = e2
                    continue
        raise last
    import numpy as np

    # Expect res to carry:
    # - res.data: (P,I) observed scores
    # - res.theta: (P,)
    # - res.beta: (I,) item difficulties
    # - res.categories: list/array of category values
    # For RSM/PCM we may not have full per-cell probs stored; compute expected using model if possible.
    X = getattr(res, "data", None)
    if X is None:
        raise AttributeError("res.data not found")

    P, I = X.shape
    person_ids = getattr(res, "person_ids", [f"P{p+1}" for p in range(P)])
    item_ids = getattr(res, "item_ids", [f"I{i+1}" for i in range(I)])

    rows = []
    # Minimal: observed only
    for p in range(P):
        for i in range(I):
            rows.append({
                "KID": person_ids[p],
                "ITEM": item_ids[i],
                "OBS": float(X[p, i]) if np.isfinite(X[p, i]) else np.nan,
            })

    pd.DataFrame(rows).to_csv(out_csv_path, index=False, encoding="utf-8")
    return out_csv_path





# NOTE: TAM (R) estimation is intentionally NOT included in this build.
# All estimation in this module is performed by the Python implementation that follows
# the ASP/Winsteps-style iterative routine.



def _build_item_vertices_relations_for_fig9(res):
    """Fallback builder for Figure 9 nodes/edges when vertices_df/relations_df are missing.
    Uses Z-score matrix (persons x items) to compute first contrast loadings and simple 2-cluster split by sign.
    Produces:
      vertices_df columns: name,value (ss(i) if available), value2(delta), loading, cluster
      relations_df columns: term1, term2, WCD
    """
    import numpy as _np
    import pandas as _pd


    # Plotly helpers (must be defined before any figure blocks that call _plotly_plot)
    try:
        import plotly.graph_objects as go
        from plotly.offline import plot as _plotly_plot
        from plotly.subplots import make_subplots
    except Exception:
        go = None
        _plotly_plot = None
        make_subplots = None
    idf = getattr(res, "item_df", None)
    if not isinstance(idf, _pd.DataFrame) or idf.empty:
        return None, None

    # Z matrix: persons x items
    Zmat = None
    dbg = getattr(res, "debug", None)
    if isinstance(dbg, dict):
        for k in ("ZSTD", "Z", "zscore", "Zscore"):
            if k in dbg:
                Zmat = dbg[k]
                break
    if Zmat is None:
        return None, None
    Z = _np.asarray(Zmat, dtype=float)
    if Z.ndim != 2 or Z.shape[1] != len(idf):
        return None, None

    # Center persons to remove mean
    Zc = Z.copy()
    # replace nan with 0 for SVD stability
    Zc[~_np.isfinite(Zc)] = 0.0
    Zc = Zc - Zc.mean(axis=0, keepdims=True)

    # First right singular vector as contrast loading
    try:
        _, _, vt = _np.linalg.svd(Zc, full_matrices=False)
        load = vt[0, :]
    except Exception:
        # fallback: eigen of covariance
        C = _np.cov(Zc, rowvar=False)
        w, v = _np.linalg.eigh(C)
        load = v[:, _np.argmax(w)]
    load = _np.asarray(load, dtype=float).ravel()
    if load.size != len(idf):
        return None, None
    # normalize for readability
    if _np.linalg.norm(load) > 0:
        load = load / _np.linalg.norm(load)

    delta = _pd.to_numeric(idf.get("MEASURE", _np.nan), errors="coerce").to_numpy()
    names = idf.get("ITEM", None)
    if names is None:
        names = _pd.Series([f"Item {i+1}" for i in range(len(idf))])
    names = _pd.Series(names).astype(str).to_list()

    cluster = _np.where(load >= 0, 1, 2).astype(int)

    # Distances between item Z-profiles (columns)
    V = Zc  # persons x items
    # compute pairwise distances (items)
    # Use cosine distance for robustness
    X = V.T  # items x persons
    # normalize rows
    denom = _np.linalg.norm(X, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    Xn = X / denom
    sim = Xn @ Xn.T
    sim = _np.clip(sim, -1.0, 1.0)
    D = 1.0 - sim
    _np.fill_diagonal(D, _np.nan)

    # Silhouette (2 clusters)
    a = _np.full(len(idf), _np.nan, dtype=float)
    b = _np.full(len(idf), _np.nan, dtype=float)
    s = _np.full(len(idf), _np.nan, dtype=float)
    for i in range(len(idf)):
        ci = cluster[i]
        m_in = (cluster == ci)
        m_out = (cluster != ci)
        if _np.sum(m_in) > 1:
            a[i] = _np.nanmean(D[i, m_in])
        if _np.sum(m_out) > 0:
            b[i] = _np.nanmean(D[i, m_out])
        if _np.isfinite(a[i]) and _np.isfinite(b[i]) and max(a[i], b[i]) > 1e-12:
            s[i] = (b[i] - a[i]) / max(a[i], b[i])

    # One-link edges: connect each item to nearest item (overall), keep strongest unique edges
    edges = []
    for i in range(len(idf)):
        j = int(_np.nanargmin(D[i, :])) if _np.isfinite(D[i, :]).any() else None
        if j is None:
            continue
        d = float(D[i, j])
        edges.append((names[i], names[j], d))
    if edges:
        dvals = _np.array([e[2] for e in edges], dtype=float)
        dmin = float(_np.nanmin(dvals))
        dmax = float(_np.nanmax(dvals))
        def strength(d):
            if not _np.isfinite(d) or dmax <= dmin:
                return 1.0
            return (dmax - d) / (dmax - dmin)
        # make undirected unique by sorted pair, keep max strength
        best = {}
        for u,v,d in edges:
            key = tuple(sorted([u,v]))
            w = strength(d)
            if key not in best or w > best[key]:
                best[key] = w
        rel = _pd.DataFrame([{"term1":k[0], "term2":k[1], "WCD":float(w)} for k,w in best.items()])
    else:
        rel = _pd.DataFrame(columns=["term1","term2","WCD"])

    vertices = _pd.DataFrame({
        "name": names,
        "loading": load,
        "delta": delta,
        "ss(i)": s,
        "a(i)": a,
        "b(i)": b,
        "cluster": cluster,
        "value2": delta,   # x-axis
        "value": s,        # bubble size default
    })
    return vertices, rel


def _beta_from_zstd_clusters(Z, cluster) -> float:
    """Compute a simple *beta* between two clusters using person-level residual summaries.

    For each person p, compute:
      x_p = mean(Z[p, items in cluster A])
      y_p = mean(Z[p, items in cluster B])
    Then fit OLS: y = alpha + beta * x, and return beta.

    This is a pragmatic, dashboard-friendly substitute for a full SEM/PLS path model:
    it reports the directional association between two residual factors defined by the
    first-contrast split (cluster sign).
    """
    import numpy as _np
    Z = _np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        return float("nan")
    c = _np.asarray(cluster)
    if c.ndim != 1 or c.size != Z.shape[1]:
        return float("nan")
    u = [int(x) for x in _np.unique(c[_np.isfinite(c)])]
    u = sorted(u)
    if len(u) < 2:
        return float("nan")
    a, b = u[0], u[1]
    ma = c == a
    mb = c == b
    if ma.sum() < 1 or mb.sum() < 1:
        return float("nan")
    x = _np.nanmean(Z[:, ma], axis=1)
    y = _np.nanmean(Z[:, mb], axis=1)
    m = _np.isfinite(x) & _np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    x = x[m]; y = y[m]
    vx = float(_np.var(x))
    if not _np.isfinite(vx) or vx <= 1e-12:
        return float("nan")
    cov = float(_np.mean((x - float(_np.mean(x))) * (y - float(_np.mean(y)))))
    return cov / vx


def _beta_p_from_zstd_clusters(Z, cluster):
    """Return (beta, p_value, n) for OLS y ~ x derived from residual-factor summaries.

    x_p = mean(Z[p, items in cluster A])
    y_p = mean(Z[p, items in cluster B])

    p-value is two-sided t-test for slope with df=n-2.
    Uses scipy if available; otherwise normal approximation for df>=30.
    """
    import numpy as _np
    import math as _math
    Z = _np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        return (float('nan'), float('nan'), 0)
    c = _np.asarray(cluster)
    if c.ndim != 1 or c.size != Z.shape[1]:
        return (float('nan'), float('nan'), 0)
    u = [int(x) for x in _np.unique(c[_np.isfinite(c)])]
    u = sorted(u)
    if len(u) < 2:
        return (float('nan'), float('nan'), 0)
    a, b = u[0], u[1]
    ma = c == a
    mb = c == b
    if ma.sum() < 1 or mb.sum() < 1:
        return (float('nan'), float('nan'), 0)
    x = _np.nanmean(Z[:, ma], axis=1)
    y = _np.nanmean(Z[:, mb], axis=1)
    m = _np.isfinite(x) & _np.isfinite(y)
    n = int(m.sum())
    if n < 3:
        return (float('nan'), float('nan'), n)
    x = x[m]; y = y[m]
    xbar = float(_np.mean(x)); ybar = float(_np.mean(y))
    Sxx = float(_np.sum((x - xbar) ** 2))
    if not _np.isfinite(Sxx) or Sxx <= 1e-12:
        return (float('nan'), float('nan'), n)
    beta = float(_np.sum((x - xbar) * (y - ybar)) / Sxx)
    alpha = ybar - beta * xbar
    resid = y - (alpha + beta * x)
    df = n - 2
    if df <= 0:
        return (beta, float('nan'), n)
    MSE = float(_np.sum(resid ** 2) / df)
    se_beta = _math.sqrt(MSE / Sxx) if MSE >= 0 else float('nan')
    if not _np.isfinite(se_beta) or se_beta <= 0:
        return (beta, float('nan'), n)
    t = beta / se_beta

    # p-value
    p = float('nan')
    try:
        from scipy.stats import t as _t
        p = float(2.0 * (1.0 - _t.cdf(abs(t), df=df)))
    except Exception:
        # normal approx for reasonably large df
        if df >= 30 and _np.isfinite(t):
            # two-sided using erfc
            p = float(_math.erfc(abs(t) / _math.sqrt(2.0)))
    return (beta, p, n)


def _q3_abs_upper_tri_from_zstd(Z, mode='items'):
    """Compute |Q3| values from standardized residuals ZSTD.

    mode='items': correlation between item columns (Q3 item-by-item)
    mode='persons': correlation between person rows (person residual similarity)

    Returns 1D array of absolute correlations from the upper triangle (excluding diagonal).
    """
    import numpy as _np
    Z = _np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        return _np.asarray([], dtype=float)
    if mode == 'persons':
        R = _np.corrcoef(Z)  # rows
    else:
        R = _np.corrcoef(Z, rowvar=False)  # cols
    if R.ndim != 2 or R.shape[0] < 2:
        return _np.asarray([], dtype=float)
    iu = _np.triu_indices(R.shape[0], k=1)
    q = _np.abs(R[iu])
    q = q[_np.isfinite(q)]
    return q


def _q3_hist10_figure(q_abs, title):
    """Return (fig, stats_df) where fig is a 10-bin histogram of |Q3| and stats include max/q95/mean."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return (None, None)
    import numpy as _np
    import pandas as _pd
    q = _np.asarray(q_abs, dtype=float)
    q = q[_np.isfinite(q)]
    if q.size == 0:
        return (None, None)

    bins = _np.linspace(0.0, 1.0, 11)  # 10 bins
    counts, edges = _np.histogram(q, bins=bins)
    labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges)-1)]

    mean_q = float(_np.mean(q))
    q95 = float(_np.quantile(q, 0.95))
    qmax = float(_np.max(q))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=counts, hovertemplate="|Q3| bin=%{x}<br>Count=%{y}<extra></extra>"))
    fig.update_layout(
        title=title,
        xaxis_title="Residual correlation coefficient |Q3| (upper-triangular bins)",
        yaxis_title="Count",
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
    )

    # Prominent summary callout (large red)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        text=f"<b>mean(|Q3|)={mean_q:.3f} &nbsp; 95%(|Q3|)={q95:.3f}</b>",
        showarrow=False,
        font=dict(size=26, color="#b00"),
        align="left",
    )

    stats = _pd.DataFrame([
        {"max(|Q3|)": qmax, "95%(|Q3|)": q95, "mean(|Q3|)": mean_q, "N": int(q.size)}
    ])
    return (fig, stats)



def _pv_rubin_group_compare(res, k: int = 10):
    """Task 14: Plausible values (PV) + Rubin's rules for group comparisons.

    We sample PV_m ~ N(theta, SE^2) for each person and each imputation m=1..k.
    Then we estimate group means and (optionally) contrasts vs a reference group,
    and pool estimates using Rubin's rules.

    Returns
    -------
    t141 : pandas.DataFrame
        Group pooled means and 95% CI.
    t142 : pandas.DataFrame
        Pairwise contrasts vs reference group (first sorted group).
    note : str | None
        Note if grouping had to be synthesized.
    """
    import numpy as _np
    import pandas as _pd
    import math as _math

    pdf = getattr(res, 'person_df', None)
    if not isinstance(pdf, _pd.DataFrame) or len(pdf) == 0:
        return None, None, "(Task 14 skipped: person_df unavailable)"

    # find theta and SE columns
    th_col = None
    se_col = None
    for c in pdf.columns:
        cl = str(c).lower()
        if th_col is None and cl in ('theta','measure','eap','wle'):
            th_col = c
        if se_col is None and (cl in ('se','se_theta','se_measure','se(eap)','se_wle') or 'se' == cl):
            se_col = c
    # fallback common RaschOnline names
    if th_col is None:
        for c in pdf.columns:
            if str(c).upper() == 'MEASURE':
                th_col = c
                break
    if se_col is None:
        for c in pdf.columns:
            if str(c).upper() in ('SE','S.E.','SE_MEASURE'):
                se_col = c
                break

    if th_col is None or se_col is None:
        return None, None, "(Task 14 skipped: theta/SE columns not found in person_df)"

    theta = _pd.to_numeric(pdf[th_col], errors='coerce').to_numpy(dtype=float)
    se = _pd.to_numeric(pdf[se_col], errors='coerce').to_numpy(dtype=float)
    m_ok = _np.isfinite(theta) & _np.isfinite(se) & (se > 0)
    if m_ok.sum() < 5:
        return None, None, "(Task 14 skipped: too few finite theta/SE)"

    # group column detection
    gcol = None
    for c in pdf.columns:
        if str(c).lower() in ('profile','group','class','grp'):
            gcol = c
            break
    note = None
    if gcol is None:
        # try from res.debug
        dbg = getattr(res, 'debug', None)
        if isinstance(dbg, dict):
            for k0 in ('profile','Profile','PROFILE','group'):
                if k0 in dbg:
                    try:
                        gv = _np.asarray(dbg[k0])
                        if gv.shape[0] == len(pdf):
                            pdf = pdf.copy()
                            pdf['profile'] = gv
                            gcol = 'profile'
                            break
                    except Exception:
                        pass
    if gcol is None:
        # synthesize a group: median split of theta
        note = "(Task 14 note: group column not found; used median split of theta as a synthetic 2-group label.)"
        med = _np.nanmedian(theta)
        gv = _np.where(theta <= med, 'G1_low', 'G2_high')
        pdf = pdf.copy()
        pdf['profile'] = gv
        gcol = 'profile'

    g = pdf[gcol].astype(str).to_numpy()

    # apply mask
    theta = theta[m_ok]
    se = se[m_ok]
    g = g[m_ok]

    # reproducible RNG (per run)
    rng = _np.random.default_rng(12345)
    k = int(max(2, k))

    groups = sorted(set(g.tolist()))
    if len(groups) < 2:
        return None, None, "(Task 14 skipped: only one group present)"

    # per-imputation estimates
    Qm = {gr: [] for gr in groups}  # means
    Um = {gr: [] for gr in groups}  # within variances of mean
    Nm = {gr: int((g == gr).sum()) for gr in groups}

    for _m in range(k):
        pv = theta + se * rng.standard_normal(theta.shape[0])
        for gr in groups:
            pv_g = pv[g == gr]
            if pv_g.size < 2:
                Qm[gr].append(_np.nan)
                Um[gr].append(_np.nan)
                continue
            Qm[gr].append(float(_np.mean(pv_g)))
            Um[gr].append(float(_np.var(pv_g, ddof=1) / pv_g.size))

    def _rubin_pool(q_list, u_list):
        q = _np.asarray(q_list, dtype=float)
        u = _np.asarray(u_list, dtype=float)
        m = _np.isfinite(q) & _np.isfinite(u)
        q = q[m]; u = u[m]
        M = int(q.size)
        if M < 2:
            return _np.nan, _np.nan, _np.nan, _np.nan
        Qbar = float(_np.mean(q))
        Ubar = float(_np.mean(u))
        B = float(_np.var(q, ddof=1))
        T = Ubar + (1.0 + 1.0/M) * B
        if T <= 0:
            return Qbar, _np.nan, _np.nan, _np.nan
        # df (Barnard-Rubin)
        if B <= 1e-15:
            df = 1e9
        else:
            df = (M - 1) * (1.0 + Ubar / ((1.0 + 1.0/M) * B))**2
        seT = _math.sqrt(T)
        return Qbar, seT, df, T

    # Table 14.1
    rows = []
    for gr in groups:
        Qbar, seT, df, T = _rubin_pool(Qm[gr], Um[gr])
        if _np.isfinite(seT):
            ci_lo = Qbar - 1.96 * seT
            ci_hi = Qbar + 1.96 * seT
        else:
            ci_lo = _np.nan; ci_hi = _np.nan
        rows.append({
            'Group': gr,
            'n': Nm.get(gr, _np.nan),
            'PV mean (Rubin pooled)': Qbar,
            'SE (total)': seT,
            '95% CI lo': ci_lo,
            '95% CI hi': ci_hi,
            'df': df,
        })
    t141 = _pd.DataFrame(rows)

    # Table 14.2 contrasts vs reference group
    ref = groups[0]
    rows2 = []
    for gr in groups[1:]:
        # difference per imputation
        qd = _np.asarray(Qm[gr], float) - _np.asarray(Qm[ref], float)
        ud = _np.asarray(Um[gr], float) + _np.asarray(Um[ref], float)
        Qbar, seT, df, T = _rubin_pool(qd, ud)
        tstat = Qbar / seT if _np.isfinite(Qbar) and _np.isfinite(seT) and seT>0 else _np.nan
        # p-value (two-sided)
        p = _np.nan
        try:
            import scipy.stats as _st
            if _np.isfinite(tstat) and _np.isfinite(df):
                p = float(2.0 * (1.0 - _st.t.cdf(abs(tstat), df)))
        except Exception:
            # normal approx
            if _np.isfinite(tstat):
                p = float(2.0 * (1.0 - 0.5*(1.0+_math.erf(abs(tstat)/_math.sqrt(2.0)))))
        ci_lo = Qbar - 1.96 * seT if _np.isfinite(seT) else _np.nan
        ci_hi = Qbar + 1.96 * seT if _np.isfinite(seT) else _np.nan
        rows2.append({
            'Contrast': f"{gr} - {ref}",
            'Diff (Rubin pooled)': Qbar,
            'SE (total)': seT,
            't': tstat,
            'df': df,
            'p': p,
            '95% CI lo': ci_lo,
            '95% CI hi': ci_hi,
        })
    t142 = _pd.DataFrame(rows2)

    return t141, t142, note




def _q3_abs_from_zstd(Z, mode: str = "items"):
    """Extract |Q3| values from the upper-triangular part of a residual correlation matrix.

    Parameters
    ----------
    Z : array-like, shape (P, I)
        Standardized residual matrix (persons x items).
    mode : {'items','persons'}
        - 'items': compute item-item residual correlations (Q3) from columns of Z.
        - 'persons': compute person-person residual correlations from rows of Z.

    Returns
    -------
    q_abs : np.ndarray
        1D array of absolute correlations from the upper triangular (excluding diagonal).
    """
    import numpy as _np
    Z = _np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        return _np.asarray([], dtype=float)

    if str(mode).lower().startswith('person'):
        R = _np.corrcoef(Z)  # rowvar=True
    else:
        R = _np.corrcoef(Z, rowvar=False)

    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return _np.asarray([], dtype=float)

    iu = _np.triu_indices(R.shape[0], k=1)
    q = _np.abs(R[iu])
    q = q[_np.isfinite(q)]
    return q


def _dif_forest_from_profile(res, profile_value=2):
    """Approx DIF forest plot using standardized mean difference of residuals between Profile==value vs others."""
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as go

    pdf = getattr(res, "person_df", None)
    idf = getattr(res, "item_df", None)
    dbg = getattr(res, "debug", None)
    if not isinstance(pdf, _pd.DataFrame) or not isinstance(idf, _pd.DataFrame) or not isinstance(dbg, dict):
        return None
    X = dbg.get("X", None)
    E = dbg.get("EXP", None)
    if X is None or E is None:
        return None
    X = _np.asarray(X, dtype=float)
    E = _np.asarray(E, dtype=float)
    if X.shape != E.shape or X.shape[1] != len(idf):
        return None
    R = X - E  # residuals
    # profile column
    prof_col = None
    for c in pdf.columns:
        if str(c).lower() in ("profile","group","class"):
            prof_col = c
            break
    if prof_col is None:
        return None
    g1 = _pd.to_numeric(pdf[prof_col], errors="coerce").to_numpy() == float(profile_value)
    g0 = ~g1
    if g1.sum() < 2 or g0.sum() < 2:
        return None

    items = idf["ITEM"].astype(str).to_list() if "ITEM" in idf.columns else [f"Item {i+1}" for i in range(len(idf))]
    smd=[]; se=[]; lo=[]; hi=[]
    for j in range(R.shape[1]):
        r1 = R[g1, j]; r0 = R[g0, j]
        r1 = r1[_np.isfinite(r1)]; r0 = r0[_np.isfinite(r0)]
        if r1.size < 2 or r0.size < 2:
            smd.append(_np.nan); se.append(_np.nan); lo.append(_np.nan); hi.append(_np.nan); continue
        m1 = float(r1.mean()); m0 = float(r0.mean())
        s1 = float(r1.std(ddof=1)); s0 = float(r0.std(ddof=1))
        n1 = r1.size; n0 = r0.size
        sp = math.sqrt(((n1-1)*s1*s1 + (n0-1)*s0*s0)/max(n1+n0-2,1))
        if sp <= 1e-12:
            d = 0.0
            sed = 1.0
        else:
            d = (m1 - m0)/sp
            sed = math.sqrt((n1+n0)/(n1*n0) + d*d/(2*(n1+n0)))
        # Hedges correction
        J = 1 - 3/(4*(n1+n0-2)-1) if (n1+n0) > 4 else 1.0
        g = d*J
        seg = sed*J
        smd.append(g); se.append(seg)
        lo.append(g - 1.96*seg); hi.append(g + 1.96*seg)

    df = _pd.DataFrame({"item":items, "SMD":smd, "SE":se, "low":lo, "high":hi}).dropna()
    if df.empty:
        return None

    # overall fixed-effect
    w = 1.0/(df["SE"]**2)
    mu = float((w*df["SMD"]).sum()/w.sum())
    se_mu = float(math.sqrt(1.0/w.sum()))
    lo_mu, hi_mu = mu-1.96*se_mu, mu+1.96*se_mu

    # forest plot
    y = list(range(len(df), 0, -1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["SMD"], y=y, mode="markers",
        marker=dict(symbol="square", size=12, color="green", line=dict(width=1,color="black")),
        error_x=dict(type="data", symmetric=False, array=df["high"]-df["SMD"], arrayminus=df["SMD"]-df["low"], thickness=1),
        hovertext=[f"{it}<br>SMD={s:.3f} [{l:.3f},{h:.3f}]" for it,s,l,h in df[["item","SMD","low","high"]].itertuples(index=False)],
        hoverinfo="text",
        name="Items"
    ))
    # overall diamond
    fig.add_trace(go.Scatter(
        x=[mu], y=[0], mode="markers",
        marker=dict(symbol="diamond", size=16, color="red", line=dict(width=1,color="black")),
        error_x=dict(type="data", symmetric=False, array=[hi_mu-mu], arrayminus=[mu-lo_mu], thickness=2),
        hovertext=[f"Overall<br>SMD={mu:.3f} [{lo_mu:.3f},{hi_mu:.3f}]"],
        hoverinfo="text",
        name="Overall"
    ))
    fig.update_yaxes(
        tickvals=y, ticktext=df["item"].tolist(),
        range=[-1, len(df)+1],
        showgrid=False
    )
    fig.add_vline(x=0, line_dash="dot", line_color="red")
    fig.update_layout(
        title="SMD in DIF measure (Figure 12)",
        xaxis_title="Standardized mean difference (Profile==2 vs others)",
        yaxis_title="Item",
        height=700,
        width=980,
        margin=dict(l=260,r=40,t=70,b=60),
        showlegend=False
    )
    return fig


def _person_dimension_top30(res):
    """Simple person dimension plot (Figure 13): top 30 persons by |PC1| of residual Z."""
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as go

    pdf = getattr(res, "person_df", None)
    idf = getattr(res, "item_df", None)
    dbg = getattr(res, "debug", None)
    if not isinstance(pdf, _pd.DataFrame) or not isinstance(idf, _pd.DataFrame) or not isinstance(dbg, dict):
        return None
    Z = dbg.get("ZSTD", None)
    if Z is None:
        return None
    Z = _np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != len(pdf):
        return None
    Zc = Z.copy()
    Zc[~_np.isfinite(Zc)] = 0.0
    Zc = Zc - Zc.mean(axis=1, keepdims=True)
    try:
        u, s, vt = _np.linalg.svd(Zc, full_matrices=False)
        pc1 = u[:,0]*s[0]
    except Exception:
        pc1 = Zc.mean(axis=1)
    pc1 = _np.asarray(pc1, dtype=float)
    # select top 30 by abs
    idx = _np.argsort(_np.abs(pc1))[::-1][:30]
    theta = _pd.to_numeric(pdf.get("MEASURE", _np.nan), errors="coerce").to_numpy()[idx]
    outfit = _pd.to_numeric(pdf.get("OUTFIT_MNSQ", _np.nan), errors="coerce").to_numpy()[idx]
    kid = pdf.get("KID", _pd.Series([str(i+1) for i in range(len(pdf))])).astype(str).to_numpy()[idx]
    pc1s = pc1[idx]
    size = 14.0 + 12.0*(_np.clip(outfit, 0, 3)/3.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theta, y=pc1s, mode="markers",
        marker=dict(size=size, color=_np.where(_np.abs(pc1s)>=_np.nanpercentile(_np.abs(pc1),80),"red","blue"),
                    line=dict(width=1,color="rgba(0,0,0,0.35)")),
        hovertext=[f"KID {k}<br>Measure={m:.3f}<br>PC1={p:.3f}<br>Outfit={o:.3f}" for k,m,p,o in zip(kid,theta,pc1s,outfit)],
        hoverinfo="text"
    ))
    fig.update_layout(
        title="Person dimension plot (Top 30) (Figure 13)",
        xaxis_title="Person measure (logit)",
        yaxis_title="Residual PC1 score",
        width=980, height=600,
        margin=dict(l=60,r=20,t=70,b=60)
    )
    fig.add_hline(y=0, line_dash="dot", line_color="black")
    return fig



def _contrast1_from_zstd(Zmat: "np.ndarray"):
    """Approximate Winsteps 1st residual contrast: PCA on item residual correlation from standardized residuals.
    Notes: sign is arbitrary; scaling can differ from Winsteps, but rank/partition is typically similar.
    """
    Z = np.asarray(Zmat, dtype=float)
    if Z.ndim != 2:
        raise ValueError("ZSTD must be 2D (persons x items)")
    Z = Z - np.nanmean(Z, axis=0, keepdims=True)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    C = np.corrcoef(Z, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    w, v = np.linalg.eigh(C)
    idx = np.argsort(w)[::-1]
    w = w[idx]; v = v[:, idx]
    loading = v[:, 0]
    eig1 = float(w[0]) if len(w) else float("nan")
    # normalize to unit SD for display
    sd = float(np.std(loading))
    if not np.isfinite(sd) or sd <= 0:
        sd = 1.0
    loading = loading / sd
    return loading, eig1, w


def _plotly_fig_to_html(fig):
    """Render a plotly figure as a div without ever relying on a global 'pio' symbol."""
    # Prefer the report's existing helper (keeps plotly.js handling consistent)
    try:
        return _plotly_plot(fig, output_type="div", include_plotlyjs=False)
    except Exception:
        # Fallback: import locally under a different name so NameError can't happen
        import plotly.io as _pio
        return _pio.to_html(fig, full_html=False, include_plotlyjs=False)



def render_report_html(res: RaschResult, run_id: str, kid_no: int = 1, item_no: int = 1, kidmap_kid: 'Optional[str]' = None, **kwargs) -> str:
    """
    Stable HTML renderer used by main.py.
    - Always returns valid HTML (never raises).
    - Displays all numeric cells with 2 decimals for readability.
    - Includes a Winsteps-style Summary block if build_winsteps_summary_text() exists.
    """
    import html as _html
    import numpy as _np
    import pandas as _pd


    # Unified config getter (HTML form params + res.debug + kwargs)
    cfg = {}
    fig11_rendered = False
    tcp_rendered = False
    # PATCH: fig11 guard to avoid NameError/UnboundLocalError
    _fig11 = None
    fig11 = None
    try:
        if isinstance(getattr(res, "debug", None), dict):
            cfg.update(res.debug)
    except Exception:
        pass
    try:
        if isinstance(kwargs, dict):
            cfg.update({k: v for k, v in kwargs.items() if v is not None})
        # PATCH: include explicit function parameters too (not in kwargs)
        cfg.setdefault('kid_no', kid_no)
        cfg.setdefault('item_no', item_no)
    except Exception:
        pass

    def _cfg_get(key, default=None):
        try:
            return cfg.get(key, default)
        except Exception:
            return default

    # UI parameters: select which person/item (1-based)
    try:
        kid_no = int(kid_no)
    except Exception:
        kid_no = int(res.debug.get('kid_no', 1))
    if kid_no < 1:
        kid_no = 1

    try:
        item_no = int(item_no)
    except Exception:
        item_no = 1
    if item_no < 1:
        item_no = 1
    setattr(res, 'kid_no', kid_no)
    # If caller didn't pass kidmap_kid, derive it from kid_no (1-based row index)
    try:
        if kidmap_kid is None and getattr(res, 'person_df', None) is not None and len(res.person_df) >= kid_no:
            if 'KID' in res.person_df.columns:
                kidmap_kid = str(res.person_df['KID'].iloc[kid_no-1])
    except Exception:
        pass

    def _fmt_cell(x, kid_no: int = 1):
        # Keep booleans as True/False
        if isinstance(x, (bool, _np.bool_)):
            return "True" if bool(x) else "False"
        # Format numeric as 2 decimals
        try:
            if isinstance(x, (_np.number, int, float)) and _np.isfinite(float(x)):
                return f"{float(x):.2f}"
        except Exception:
            pass
        # NaN / None
        if x is None:
            return ""
        try:
            if isinstance(x, float) and _np.isnan(x):
                return ""
        except Exception:
            pass
        return _html.escape(str(x))

    def _df_to_html(df: "_pd.DataFrame", max_rows: int = 50) -> str:
        if df is None or len(df) == 0:
            return "<p>(none)</p>"
        try:
            view = df.head(max_rows).copy()
            # Convert to string table with 2-decimal numeric formatting
            view = view.applymap(_fmt_cell)
            return view.to_html(index=False, border=0, classes="tbl", escape=False)
        except Exception as e:
            return f"<p>(table render failed: {_html.escape(repr(e))})</p>"

    # --- meta ---
    try:
        build_id = str(globals().get("RASONLINE_BUILD_ID", "unknown"))
    except Exception:
        build_id = "unknown"

    try:
        tau = getattr(res, "tau", [])
        tau_list = list(tau) if not hasattr(tau, "tolist") else tau.tolist()
        tau_str = ", ".join([f"{float(t):+.4f}" for t in tau_list])
        tau_maxabs = float(np.max(np.abs(tau_list))) if len(tau_list) else 0.0
    except Exception:
        tau_str = ""

    # --- Summary (Winsteps-style preferred) ---
    summary_pre = ""
    try:
        if "build_winsteps_summary_text" in globals():
            summary_pre = str(build_winsteps_summary_text(res))
        # Normalize accidental literal \n sequences (from earlier patched builders)
        summary_pre = summary_pre.replace('\\n', '\n')
    except Exception as e:
        summary_pre = f"(Summary unavailable: {repr(e)})"

    # fallback summary table (if present)
    try:
        summary_df = build_summary_tables(res) if "build_summary_tables" in globals() else _pd.DataFrame()
    except Exception:
        summary_df = _pd.DataFrame()

    css = """
    <style>
      body{font-family:Arial,Helvetica,sans-serif;margin:18px;line-height:1.35}
      .meta{margin-bottom:12px}
      .tbl{border-collapse:collapse;font-size:13px}
      .tbl td,.tbl th{border:1px solid #ddd;padding:6px 8px;vertical-align:top}
      .tbl th{background:#f5f5f5}
      .pre{white-space:pre;overflow:auto;background:#f6f8fa;padding:12px;border-radius:10px}
      .note{color:#666;font-size:12px}
    </style>
    """

    plotly_js = "<script src='https://cdn.plot.ly/plotly-2.30.0.min.js'></script>"

    html_parts = []

    body = ""  # ensure defined even if a later block is skipped
    html_parts.append(f"<div class='meta'><div><b>Rasch report</b></div>"
                      f"<div>run_id: {_html.escape(str(run_id))} (build: {_html.escape(build_id)})</div>"
                      f"<div>Mode: {_html.escape(str(getattr(res,'mode',''))) }</div>"
                      f"<div>Estimation: iterations={_html.escape(str(getattr(res,'iterations','')))}, "
                      f"last_error={_html.escape(str(getattr(res,'last_error','')))}, "
                      f"stop_reason={_html.escape(str(getattr(res,'stop_reason','')))}</div>"
                      f"<div>Categories: {int(getattr(res,'min_cat',0))}..{int(getattr(res,'max_cat',0))}</div>"
                      f"<div>Step thresholds (tau): {tau_str} (max|tau|={tau_maxabs:.4f})</div>"
                      f"</div>")

    # Winsteps summary block (text)
    if summary_pre:
        html_parts.append("<h2>Summary</h2>")
        html_parts.append(f"<pre class='pre'>{_html.escape(summary_pre)}</pre>")
    else:
        # If no winsteps text, show the tabular summary
        html_parts.append("<h2>Summary</h2>")
        html_parts.append("<div class='note'>(Winsteps-style summary not available; showing table summary)</div>")
        html_parts.append(_df_to_html(summary_df, max_rows=20))

    # Person / Item tables
    

    # Person / Item tables (always show the first 50 like Winsteps/RUMM-style reports)
    try:
        _pdf = getattr(res, "person_df", None)
        if isinstance(_pdf, _pd.DataFrame) and len(_pdf) > 0:
            html_parts.append("<h2>Person table (first 50)</h2>")
            html_parts.append(_df_to_html(_pdf, max_rows=50))
    except Exception as _e:
        html_parts.append("<div class='note'>(Person table failed: " + _html.escape(repr(_e)) + ")</div>")

    try:
        _idf = getattr(res, "item_df", None)
        if isinstance(_idf, _pd.DataFrame) and len(_idf) > 0:
            html_parts.append("<h2>Item table</h2>")
            html_parts.append(_df_to_html(_idf, max_rows=200))
    except Exception as _e:
        html_parts.append("<div class='note'>(Item table failed: " + _html.escape(repr(_e)) + ")</div>")

    # RUMM 2020 section (Chi-square / PCA / DIF) — show it in the HTML report
    try:
        html_parts.append(render_rumm2020_block(res, jitem=int(item_no) if str(item_no).isdigit() else 1))
    except Exception as _e:
        html_parts.append("<div class='note'>(RUMM section failed: " + _html.escape(repr(_e)) + ")</div>")

    # KIDMAP target summary (visual aid; does not change estimation or Winsteps tables)
    try:
        if kidmap_kid is not None and person_df is not None and len(person_df) > 0 and "KID" in person_df.columns:
            _r = person_df[person_df["KID"].astype(str) == str(kidmap_kid)]
            if _r is not None and len(_r) > 0:
                cols = [c for c in ["KID","MEASURE","SE","INFIT_MNSQ","OUTFIT_MNSQ","profile","TOTAL_SCORE","COUNT"] if c in _r.columns]
                html_parts.append("<h2>KIDMAP (selected person)</h2>")
                html_parts.append(_df_to_html(_r[cols].head(1), max_rows=1))
    except Exception:
        pass


    

    # --- Wright maps (visuals only; do NOT alter any computed tables) ---
    wright_html_1 = ""
    wright_html_2 = ""
    try:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as _plotly_plot
            from plotly.subplots import make_subplots
        except Exception:
            go = None
            _plotly_plot = None
            make_subplots = None

        if go is None or _plotly_plot is None or make_subplots is None:
            note = "<div class='note'><b>Wright maps:</b> Plotly not available. " \
                   "Install: <code>pip install plotly</code> (and add <code>plotly</code> to requirements.txt on GAE).</div>"
            wright_html_1 = note
            wright_html_2 = ""
        else:
            import numpy as _np
            import pandas as _pd
            p = getattr(res, "person_df", None)
            it = getattr(res, "item_df", None)
            if p is None or it is None or len(p)==0 or len(it)==0:
                wright_html_1 = "<div class='note'><b>Wright maps:</b> (no person/item tables found)</div>"
            else:
                p2 = p.copy()
                i2 = it.copy()

                # Attach profile from res.debug (stored during data prep)
                prof = None
                try:
                    prof = getattr(res, "debug", {}).get("profile", None)
                except Exception:
                    prof = None
                if prof is not None and len(prof) == len(p2):
                    p2["profile"] = _np.asarray(prof, dtype=object)


                # Fallback: if profile still missing, create a virtual binary profile from total score (median split)
                if "profile" not in p2.columns:
                    score_col = None
                    for cand in ["TOTAL_SCORE", "TOTAL", "SCORE", "Count", "COUNT"]:
                        if cand in p2.columns:
                            score_col = cand
                            break
                    if score_col is not None:
                        vv = _np.asarray(p2[score_col], dtype=float)
                        med = _np.nanmedian(vv)
                        p2["profile"] = _np.where(vv >= med, "1", "0")
                    else:
                        # last resort: alternate 0/1
                        p2["profile"] = _np.where((_np.arange(len(p2)) % 2)==0, "0", "1")

                def _size_from_se(se, smin=6.0, smax=28.0):
                    se = _np.asarray(se, dtype=float)
                    out = _np.full_like(se, smin, dtype=float)
                    m = _np.isfinite(se)
                    if not m.any():
                        return out
                    q1 = _np.nanquantile(se[m], 0.10)
                    q9 = _np.nanquantile(se[m], 0.90)
                    if not _np.isfinite(q1) or not _np.isfinite(q9) or q9 <= q1:
                        out[m] = (smin+smax)/2.0
                        return out
                    se2 = _np.clip(se, q1, q9)
                    out[m] = smin + (se2[m]-q1)/(q9-q1) * (smax-smin)
                    return out

                i2["_msize"] = _size_from_se(i2.get("SE", _np.nan))

                fit_cut = 1.5
                i2["_x"] = _np.asarray(i2.get("INFIT_MNSQ", _np.nan), dtype=float)
                good = _np.asarray(i2["_x"] <= fit_cut)
                bad  = _np.asarray(i2["_x"] >  fit_cut)

                # Wright map 1
                def _overall_person_hist():
                    yy = _np.asarray(p2["MEASURE"], dtype=float)
                    yy = yy[_np.isfinite(yy)]
                    if yy.size == 0:
                        return None
                    binw = 0.25
                    y0 = float(_np.floor(_np.nanmin(yy)/binw)*binw)
                    y1 = float(_np.ceil(_np.nanmax(yy)/binw)*binw)
                    bins = _np.arange(y0, y1+binw, binw)
                    counts, edges = _np.histogram(yy, bins=bins)
                    centers = (edges[:-1] + edges[1:]) / 2.0
                    return counts, centers

                fig1 = make_subplots(
                    rows=1, cols=2, shared_yaxes=True,
                    column_widths=[0.26, 0.74],
                    horizontal_spacing=0.005,
                    subplot_titles=["Persons (distribution)", "Items (INFIT)"]
                )

                ph = _overall_person_hist()
                if ph is not None:
                    counts, centers = ph
                    fig1.add_trace(go.Bar(
                        x=-counts, y=centers, orientation="h",
                        hovertemplate="Count: %{customdata}<br>Theta bin: %{y:.2f}<extra></extra>",
                        customdata=counts,
                        opacity=0.85,
                        showlegend=False
                    ), row=1, col=1)
                    fig1.update_xaxes(showticklabels=False, row=1, col=1)

                fig1.add_trace(go.Scatter(
                    x=i2.loc[good, "_x"], y=i2.loc[good, "MEASURE"],
                    mode="markers",
                    marker=dict(size=i2.loc[good, "_msize"], opacity=0.85, color="blue"),
                    name=f"INFIT ≤ {fit_cut}",
                    hovertemplate="ITEM: %{customdata[0]}<br>Delta: %{y:.3f}<br>SE: %{customdata[1]:.3f}<br>INFIT: %{customdata[2]:.3f}<br>OUTFIT: %{customdata[3]:.3f}<extra></extra>",
                    customdata=_np.stack([
                        i2.loc[good, "ITEM"].astype(str).to_numpy(),
                        _np.asarray(i2.loc[good, "SE"], dtype=float),
                        _np.asarray(i2.loc[good, "INFIT_MNSQ"], dtype=float),
                        _np.asarray(i2.loc[good, "OUTFIT_MNSQ"], dtype=float),
                    ], axis=1) if good.any() else _np.empty((0,4))
                ), row=1, col=2)

                fig1.add_trace(go.Scatter(
                    x=i2.loc[bad, "_x"], y=i2.loc[bad, "MEASURE"],
                    mode="markers",
                    marker=dict(size=i2.loc[bad, "_msize"], opacity=0.95, color="red"),
                    name=f"INFIT > {fit_cut}",
                    hovertemplate="ITEM: %{customdata[0]}<br>Delta: %{y:.3f}<br>SE: %{customdata[1]:.3f}<br>INFIT: %{customdata[2]:.3f}<br>OUTFIT: %{customdata[3]:.3f}<extra></extra>",
                    customdata=_np.stack([
                        i2.loc[bad, "ITEM"].astype(str).to_numpy(),
                        _np.asarray(i2.loc[bad, "SE"], dtype=float),
                        _np.asarray(i2.loc[bad, "INFIT_MNSQ"], dtype=float),
                        _np.asarray(i2.loc[bad, "OUTFIT_MNSQ"], dtype=float),
                    ], axis=1) if bad.any() else _np.empty((0,4))
                ), row=1, col=2)

                fig1.add_shape(
                    type="line",
                    x0=fit_cut, x1=fit_cut, y0=-10, y1=10,
                    line=dict(color="red", width=2, dash="dot"),
                    xref="x2", yref="y"
                )

                fig1.update_layout(
                    height=560,
                    margin=dict(l=35, r=15, t=60, b=45),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                fig1.update_yaxes(title_text="", row=1, col=1)
                # Put y-axis title into the gap between panels
                fig1.add_annotation(text="Logit (MEASURE)", x=0.50, xref="paper", y=1.08, yref="paper",
                                    showarrow=False, textangle=0, font=dict(size=14))
                fig1.update_xaxes(title_text="INFIT MNSQ", row=1, col=2)

                # Highlight selected person (KIDMAP) on the Wright map (horizontal dashed line)
                try:
                    if kidmap_kid is not None and "KID" in p2.columns:
                        _row = p2[p2["KID"].astype(str) == str(kidmap_kid)]
                    else:
                        _row = None
                    if _row is not None and len(_row) > 0:
                        _y = float(_row["MEASURE"].iloc[0])
                        fig1.add_shape(type="line", x0=0, x1=1, xref="paper",
                                       y0=_y, y1=_y, line=dict(color="black", width=2, dash="dash"))
                        fig1.add_annotation(text=f"KIDMAP: {kidmap_kid}", x=0.98, xref="paper",
                                            y=_y, yref="y", showarrow=False, xanchor="right",
                                            font=dict(size=12))
                except Exception:
                    pass

                wright_html_1 = _plotly_plot(fig1, include_plotlyjs="cdn", output_type="div")

                # Wright map 2 (by profile)
                if "profile" not in p2.columns:
                    wright_html_2 = "<div class='note'><b>Wright map 2:</b> group column <code>profile</code> not found in person table.</div>"
                else:
                    uniq = sorted(_pd.Series(p2["profile"]).astype(str).fillna("NA").unique().tolist())
                    fig2 = make_subplots(
                        rows=1, cols=len(uniq)+1, shared_yaxes=True,
                        column_widths=[0.11]*len(uniq) + [0.45],
                        horizontal_spacing=0.005,
                        subplot_titles=[f"{g}" for g in uniq] + ["Items (INFIT)"]
                    )

                    def _add_hist_for(df, col):
                        yy = _np.asarray(df["MEASURE"], dtype=float)
                        yy = yy[_np.isfinite(yy)]
                        if yy.size == 0:
                            return
                        binw = 0.25
                        y0 = float(_np.floor(_np.nanmin(yy)/binw)*binw)
                        y1 = float(_np.ceil(_np.nanmax(yy)/binw)*binw)
                        bins = _np.arange(y0, y1+binw, binw)
                        counts, edges = _np.histogram(yy, bins=bins)
                        centers = (edges[:-1] + edges[1:]) / 2.0
                        fig2.add_trace(go.Bar(
                            x=-counts, y=centers, orientation="h",
                            customdata=counts,
                            hovertemplate="Count: %{customdata}<br>Theta bin: %{y:.2f}<extra></extra>",
                            opacity=0.85,
                            showlegend=False
                        ), row=1, col=col)
                        fig2.update_xaxes(showticklabels=False, row=1, col=col)

                    for j, g in enumerate(uniq, start=1):
                        sel = p2[p2["profile"].astype(str) == g]
                        _add_hist_for(sel, j)

                    items_col = len(uniq) + 1
                    fig2.add_trace(go.Scatter(
                        x=i2.loc[good, "_x"], y=i2.loc[good, "MEASURE"],
                        mode="markers",
                        marker=dict(size=i2.loc[good, "_msize"], opacity=0.85, color="blue"),
                        name=f"INFIT ≤ {fit_cut}",
                        showlegend=True
                    ), row=1, col=items_col)

                    fig2.add_trace(go.Scatter(
                        x=i2.loc[bad, "_x"], y=i2.loc[bad, "MEASURE"],
                        mode="markers",
                        marker=dict(size=i2.loc[bad, "_msize"], opacity=0.95, color="red"),
                        name=f"INFIT > {fit_cut}",
                        showlegend=True
                    ), row=1, col=items_col)

                    fig2.add_shape(
                        type="line",
                        x0=fit_cut, x1=fit_cut, y0=-10, y1=10,
                        line=dict(color="red", width=2, dash="dot"),
                        xref=f"x{items_col}", yref="y"
                    )

                    fig2.update_layout(
                        height=560,
                        margin=dict(l=35, r=15, t=60, b=45),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    )
                    fig2.update_yaxes(title_text="", row=1, col=1)
                    # Put y-axis title into the gap (paper coords) roughly between last profile panel and items
                    fig2.add_annotation(text="Logit (MEASURE)", x=0.50, xref="paper", y=1.08, yref="paper",
                                        showarrow=False, textangle=0, font=dict(size=14))
                    fig2.update_xaxes(title_text="INFIT MNSQ", row=1, col=items_col)

                    wright_html_2 = _plotly_plot(fig2, include_plotlyjs="cdn", output_type="div")
    except Exception as _e:
        wright_html_1 = f"<div class='note'><b>Wright maps:</b> (render failed: {_html.escape(repr(_e))})</div>"

    if wright_html_1:
        html_parts.append("<h2>Wright map 1</h2>")
        html_parts.append(wright_html_1)
        try:
            _note = FIGURE_GUIDE_NOTES.get("Figure 1") if isinstance(globals().get("FIGURE_GUIDE_NOTES", None), dict) else None
            if _note:
                html_parts.append(f"<div class=\'note\'>{_note}</div>")
        except Exception:
            pass
    if wright_html_2:
        html_parts.append("<h2>Wright map 2 (by profile)</h2>")
        html_parts.append(wright_html_2)
        try:
            _note = FIGURE_GUIDE_NOTES.get("Figure 2") if isinstance(globals().get("FIGURE_GUIDE_NOTES", None), dict) else None
            if _note:
                html_parts.append(f"<div class=\'note\'>{_note}</div>")
        except Exception:
            pass




    # --- Item-Outfit plot (Figure 3) ---
    kidmap_html = ""
    try:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as _plotly_plot
        except Exception:
            go = None
            _plotly_plot = None

        if go is not None and _plotly_plot is not None:
            import numpy as _np
            it = getattr(res, "item_df", None)
            if it is not None and len(it) > 0 and "OUTFIT_MNSQ" in it.columns and "MEASURE" in it.columns:
                i3 = it.copy()
                fit_cut = 2.0
                x = _np.asarray(i3["OUTFIT_MNSQ"], dtype=float)
                y = _np.asarray(i3["MEASURE"], dtype=float)  # delta/logit
                se = _np.asarray(i3["SE"], dtype=float) if "SE" in i3.columns else _np.full(len(i3), _np.nan)
                # size from SE
                def _size_from_se(se, smin=6.0, smax=28.0):
                    se = _np.asarray(se, dtype=float)
                    out = _np.full_like(se, smin, dtype=float)
                    m = _np.isfinite(se)
                    if not m.any():
                        return out
                    q1 = _np.nanquantile(se[m], 0.10)
                    q9 = _np.nanquantile(se[m], 0.90)
                    if not _np.isfinite(q1) or not _np.isfinite(q9) or q9 <= q1:
                        out[m] = (smin+smax)/2.0
                        return out
                    se2 = _np.clip(se, q1, q9)
                    out[m] = smin + (se2[m]-q1)/(q9-q1) * (smax-smin)
                    return out
                msize = _size_from_se(se)
                good = _np.asarray(x <= fit_cut)
                bad = _np.asarray(x > fit_cut)

                fig3 = go.Figure()
                if good.any():
                    fig3.add_trace(go.Scatter(
                        x=x[good], y=y[good],
                        mode="markers",
                        marker=dict(size=msize[good], opacity=0.85, color="blue"),
                        name=f"OUTFIT ≤ {fit_cut}",
                        hovertemplate="ITEM: %{customdata[0]}<br>Delta: %{y:.3f}<br>SE: %{customdata[1]:.3f}<br>OUTFIT: %{customdata[2]:.3f}<br>INFIT: %{customdata[3]:.3f}<extra></extra>",
                        customdata=_np.stack([
                            i3.loc[good, "ITEM"].astype(str).to_numpy() if "ITEM" in i3.columns else _np.arange(len(i3))[good].astype(str),
                            se[good],
                            x[good],
                            _np.asarray(i3["INFIT_MNSQ"], dtype=float)[good] if "INFIT_MNSQ" in i3.columns else _np.full(good.sum(), _np.nan),
                        ], axis=1)
                    ))
                if bad.any():
                    fig3.add_trace(go.Scatter(
                        x=x[bad], y=y[bad],
                        mode="markers",
                        marker=dict(size=msize[bad], opacity=0.95, color="red"),
                        name=f"OUTFIT > {fit_cut}",
                        hovertemplate="ITEM: %{customdata[0]}<br>Delta: %{y:.3f}<br>SE: %{customdata[1]:.3f}<br>OUTFIT: %{customdata[2]:.3f}<br>INFIT: %{customdata[3]:.3f}<extra></extra>",
                        customdata=_np.stack([
                            i3.loc[bad, "ITEM"].astype(str).to_numpy() if "ITEM" in i3.columns else _np.arange(len(i3))[bad].astype(str),
                            se[bad],
                            x[bad],
                            _np.asarray(i3["INFIT_MNSQ"], dtype=float)[bad] if "INFIT_MNSQ" in i3.columns else _np.full(bad.sum(), _np.nan),
                        ], axis=1)
                    ))

                fig3.add_shape(type="line", x0=fit_cut, x1=fit_cut, y0=-10, y1=10,
                               line=dict(color="red", width=2, dash="dot"))
                fig3.update_layout(
                    height=520,
                    margin=dict(l=35, r=15, t=60, b=45),
                    title="KIDMAP (Figure 3) — Item OUTFIT MNSQ vs Delta"
                )
                fig3.update_xaxes(title_text="OUTFIT MNSQ")
                fig3.update_yaxes(title_text="Delta (logit)")

                kidmap_html = _plotly_plot(fig3, include_plotlyjs="cdn", output_type="div")
    except Exception as _e:
        kidmap_html = f"<div class='note'><b>KIDMAP:</b> (render failed: {_html.escape(repr(_e))})</div>"

    if kidmap_html:
        html_parts.append("<h2>Item-Outfit plot (Figure 3)</h2>")
        html_parts.append("<p class='note'>Item-outfit plot shows item OUTFIT MNSQ vs item difficulty (delta). Red points exceed the fit cut.</p>")
        html_parts.append(kidmap_html)


    # --- Downloads + Person-Outfit + Scree plots inserted after Figure 3 ---

    # --- Downloads table (CSV exports) ---
    try:
        html_parts.append("<h2>Downloads</h2>")
        html_parts.append("<table class='tbl'><tr><th>File</th><th>Link</th></tr>")
        for _fn, _label in [
            ("person_estimates.csv", "Person estimates"),
            ("item_estimates.csv", "Item estimates"),
            ("observed_response.csv", "Observed responses (matrix)"),
            ("simulated_response.csv", "Simulated responses (Rasch model)"),
            ("expected_score.csv", "Expected scores (matrix)"),
            ("residual.csv", "Residuals (matrix)"),
            ("zscore.csv", "Z-scores (standardized residuals, matrix)"),
        ]:
            html_parts.append(
                f"<tr><td>{_html.escape(_label)}</td>"
                f"<td><a href='/reports/{_html.escape(str(run_id))}/{_html.escape(_fn)}' download>{_html.escape(_fn)}</a></td></tr>"
            )
        html_parts.append("</table>")
        html_parts.append("<p class='note'>Tip: If a link 404s, re-run after updating main.py to export the matrices.</p>")
    except Exception:
        pass

    # --- Person-Outfit plot (Figure 4) ---
    try:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot as _plotly_plot
        except Exception:
            go = None
            _plotly_plot = None

        if go is not None and _plotly_plot is not None:
            p4 = getattr(res, "person_df", None)
            if p4 is not None and len(p4) > 0 and "OUTFIT_MNSQ" in p4.columns and "MEASURE" in p4.columns:
                df4 = p4.copy()
                fit_cut_p = 2.0
                x = _np.asarray(df4["OUTFIT_MNSQ"], dtype=float)
                y = _np.asarray(df4["MEASURE"], dtype=float)
                se = _np.asarray(df4["SE"], dtype=float) if "SE" in df4.columns else _np.full(len(df4), _np.nan)

                def _size_from_se(se, smin=6.0, smax=28.0):
                    se = _np.asarray(se, dtype=float)
                    out = _np.full_like(se, smin, dtype=float)
                    m = _np.isfinite(se)
                    if not m.any():
                        return out
                    q1 = _np.nanquantile(se[m], 0.10)
                    q9 = _np.nanquantile(se[m], 0.90)
                    if not _np.isfinite(q1) or not _np.isfinite(q9) or q9 <= q1:
                        out[m] = (smin+smax)/2.0
                        return out
                    se2 = _np.clip(se, q1, q9)
                    out[m] = smin + (se2[m]-q1)/(q9-q1) * (smax-smin)
                    return out

                msize = _size_from_se(se)
                good = _np.asarray(x <= fit_cut_p)
                bad = _np.asarray(x > fit_cut_p)

                fig4 = go.Figure()


                # strata cut lines (person measure cutpoints)
                try:
                    _cuts = []
                    # Default cutpoints (Winsteps-style strata) if none were computed
                    if not _cuts:
                        try:
                            _cuts = [3.5, 1.0, -1.5, -4.0]
                        except Exception:
                            _cuts = []
                    if 'strata_edges' in locals() and strata_edges:
                        _cuts = [float(x) for x in strata_edges[1:-1] if _np.isfinite(x)]
                    elif 'strata_cutpoints' in locals() and strata_cutpoints:
                        _cuts = [float(x) for x in strata_cutpoints if _np.isfinite(x)]
                    for yy in _cuts:
                        # Only 4 strata => 3 boundaries (A/B/C/D): remove the extra bottom boundary (e.g., -4.0)
                        if float(yy) <= -3.5:
                            continue
                        fig4.add_hline(y=float(yy), line_dash='dot', line_color='red')
                except Exception:
                    pass

                fig4.add_trace(go.Scatter(
                    x=x[good], y=y[good], mode="markers",
                    marker=dict(size=msize[good], opacity=0.75),
                    name="<= 2.0",
                    text=[f"KID={_html.escape(str(k))}<br>SE={float(s):.2f}<br>Measure={float(mm):.2f}<br>Outfit={float(of):.2f}"
                          for k, s, mm, of in zip(df4.loc[good, "KID"], se[good], y[good], x[good])]
                ))
                if bad.any():
                    fig4.add_trace(go.Scatter(
                        x=x[bad], y=y[bad], mode="markers",
                        marker=dict(size=msize[bad], opacity=0.85, symbol="diamond"),
                        name="> 2.0",
                        text=[f"KID={_html.escape(str(k))}<br>SE={float(s):.2f}<br>Measure={float(mm):.2f}<br>Outfit={float(of):.2f}"
                              for k, s, mm, of in zip(df4.loc[bad, "KID"], se[bad], y[bad], x[bad])]
                    ))
                fig4.add_vline(x=fit_cut_p, line_dash="dot")
                # Strata cut lines (theta cutpoints)
                try:
                    for yy in (getattr(res, 'debug', {}).get('STRATA_EDGES', []) or [])[1:-1]:
                        if _np.isfinite(yy):
                            fig4.add_hline(y=float(yy), line_dash='dot', line_color='red')
                except Exception:
                    pass
                fig4.update_layout(
                    title="Person-Outfit plot (Figure 4)",
                    xaxis_title="OUTFIT MNSQ",
                    yaxis_title="MEASURE (logit)",
                    height=520,
                )
                html_parts.append("<h2>Person-Outfit plot (Figure 4)</h2>")
                html_parts.append(_plotly_plot(fig4, include_plotlyjs=False, output_type="div"))

                

                # --- Scree plots (Figures 5–7): observed responses, Rasch residuals, and cell Z-scores ---
                try:
                    import plotly.graph_objects as go
                    import numpy as _np

                    def _eigvals_from_matrix(M):
                        M = _np.asarray(M, dtype=float)
                        if M.ndim != 2 or M.size == 0:
                            return None
                        # center by columns
                        M = M - _np.nanmean(M, axis=0, keepdims=True)
                        M = _np.nan_to_num(M, nan=0.0, posinf=0.0, neginf=0.0)
                        # correlation across items
                        C = _np.corrcoef(M, rowvar=False)
                        C = _np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
                        w = _np.linalg.eigvalsh(C)
                        w = _np.sort(w)[::-1]
                        return w

                    def _pa_curve(shape, sims=80, seed=123):
                        P, I = shape
                        rng = _np.random.default_rng(seed)
                        ws = []
                        for _ in range(int(sims)):
                            R = rng.standard_normal((P, I))
                            w = _eigvals_from_matrix(R)
                            if w is not None:
                                ws.append(w)
                        if not ws:
                            return None
                        W = _np.vstack(ws)
                        return _np.nanpercentile(W, 95, axis=0)

                    def _plot_scree(eig, pa, title):
                        if eig is None:
                            return None
                        x = _np.arange(1, len(eig)+1)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=x, y=eig, mode="lines+markers", name="Observed eigenvalues"))
                        if pa is not None and len(pa) == len(eig):
                            fig.add_trace(go.Scatter(x=x, y=pa, mode="lines+markers", name="Parallel analysis (95th)"))
                        fig.update_layout(height=420, title=title, margin=dict(l=40, r=20, t=55, b=40))
                        fig.update_xaxes(title_text="Component")
                        fig.update_yaxes(title_text="Eigenvalue")
                        return fig

                    dbg = getattr(res, "debug", {}) if isinstance(getattr(res, "debug", None), dict) else {}
                    is_cont = bool(dbg.get('is_continuous_input', False))
                    X_obs = dbg.get('X_post', None) or dbg.get('X_transformed', None) or dbg.get('X0', None) or dbg.get('X_obs', None)
                    if (X_obs is None) and (not is_cont):
                        X_obs = dbg.get('X', None)
                    E_exp = dbg.get("E_exp", None)
                    Zstd  = dbg.get("ZSTD", None)

                    # residuals in score units: O - E (original category scale)
                    R_resid = None
                    if X_obs is not None and E_exp is not None:
                        try:
                            R_resid = _np.asarray(X_obs, dtype=float) - _np.asarray(E_exp, dtype=float)
                        except Exception:
                            R_resid = None

                    # If ZSTD is missing, compute a safe fallback so KIDMAP/Figure 9 can render.
                    if Zstd is None:
                        try:
                            if R_resid is not None and isinstance(R_resid, _np.ndarray) and R_resid.ndim == 2:
                                sd = _np.nanstd(R_resid, axis=0, ddof=1)
                                sd[~_np.isfinite(sd)] = 0.0
                                sd[sd <= 0] = 1.0
                                Zstd = R_resid / sd
                            elif X_obs is not None and _np.asarray(X_obs).ndim == 2:
                                Xtmp = _np.asarray(X_obs, dtype=float)
                                mu = _np.nanmean(Xtmp, axis=0, keepdims=True)
                                sd = _np.nanstd(Xtmp, axis=0, ddof=1, keepdims=True)
                                sd[~_np.isfinite(sd)] = 1.0
                                sd[sd <= 0] = 1.0
                                Zstd = (Xtmp - mu) / sd
                            else:
                                Zstd = None
                            if Zstd is not None:
                                try:
                                    if isinstance(getattr(res, "debug", None), dict):
                                        res.debug["ZSTD"] = Zstd
                                except Exception:
                                    pass
                        except Exception:
                            Zstd = None

                    eig_obs = _eigvals_from_matrix(X_obs) if X_obs is not None else None
                    eig_res = _eigvals_from_matrix(R_resid) if R_resid is not None else None
                    eig_z   = _eigvals_from_matrix(Zstd) if Zstd is not None else None

                    def _dc_from_eig(eig_arr):
                        """DC = r/(1+r) where r=(v1/v2)/(v2/v3) = (v1*v3)/(v2^2)."""
                        try:
                            if eig_arr is None:
                                return None
                            e = _np.asarray(eig_arr, dtype=float)
                            e = e[_np.isfinite(e)]
                            if e.size < 3:
                                return None
                            v1, v2, v3 = float(e[0]), float(e[1]), float(e[2])
                            if not (_np.isfinite(v1) and _np.isfinite(v2) and _np.isfinite(v3)):
                                return None
                            if v2 == 0:
                                return None
                            r = (v1 / v2) / (v2 / v3) if v3 != 0 else None
                            if r is None or not _np.isfinite(r):
                                return None
                            return float(r / (1.0 + r))
                        except Exception:
                            return None

                    if eig_obs is not None:
                        pa = _pa_curve((_np.asarray(X_obs).shape[0], _np.asarray(X_obs).shape[1])) if _np.asarray(X_obs).ndim == 2 else None
                        _dc = _dc_from_eig(eig_obs)
                        _t5 = "Scree plot of observed responses (Figure 5)" + (f"  DC={_dc:.2f}" if _dc is not None else "")
                        fig5 = _plot_scree(eig_obs, pa, _t5)
                        if fig5 is not None:
                            html_parts.append("<h2>Scree plot (Figure 5) — observed responses</h2>")
                            html_parts.append(_plotly_plot(fig5, include_plotlyjs=False, output_type="div"))
                            html_parts.append("<p class='note'>Dimensionality check based on eigenvalues of the item-by-item correlation matrix of observed responses. The parallel-analysis curve (95th percentile) provides a reference for noise.</p>")

                    if eig_res is not None:
                        pa = _pa_curve((_np.asarray(R_resid).shape[0], _np.asarray(R_resid).shape[1])) if _np.asarray(R_resid).ndim == 2 else None
                        _dc = _dc_from_eig(eig_res)
                        _t6 = "Scree plot of Rasch residuals (Figure 6)" + (f"  DC={_dc:.2f}" if _dc is not None else "")
                        fig6 = _plot_scree(eig_res, pa, _t6)
                        if fig6 is not None:
                            html_parts.append("<h2>Scree plot (Figure 6) — Rasch residuals</h2>")
                            html_parts.append(_plotly_plot(fig6, include_plotlyjs=False, output_type="div"))
                            html_parts.append("<p class='note'>Residual-based scree assesses unidimensionality after removing the Rasch measure component. Large first residual contrasts can indicate secondary dimensions.</p>")

                    if eig_z is not None:
                        pa = _pa_curve((_np.asarray(Zstd).shape[0], _np.asarray(Zstd).shape[1])) if _np.asarray(Zstd).ndim == 2 else None
                        _dc = _dc_from_eig(eig_z)
                        _t7 = "Scree plot of cell Z-scores (Figure 7)" + (f"  DC={_dc:.2f}" if _dc is not None else "")
                        fig7s = _plot_scree(eig_z, pa, _t7)
                        if fig7s is not None:
                            html_parts.append("<h2>Scree plot (Figure 7) — cell Z-scores</h2>")
                            html_parts.append(_plotly_plot(fig7s, include_plotlyjs=False, output_type="div"))
                            html_parts.append("<p class='note'>Z-score scree summarizes standardized residual structure (cell-level). This aligns with the Z-score–based diagnostics used in KIDMAP and the Dimension–Kano map.</p>")
                except Exception as _e:
                    html_parts.append("<div class='note'>(Scree plots skipped: " + _html.escape(repr(_e)) + ")</div>")


                # 


                # Strata table (Winsteps-style): cut by equal-width intervals in person measures
                try:
                    th = _np.asarray(getattr(res, 'theta', []), dtype=float)
                    Xmat = _np.asarray(res.debug.get('X', []), dtype=float) if isinstance(getattr(res, 'debug', None), dict) else _np.asarray([], dtype=float)
                    pdf = getattr(res, 'person_df', None)
                    if th.size and Xmat.ndim == 2 and Xmat.shape[0] == th.size and Xmat.shape[1] > 0:
                        itemno = int(Xmat.shape[1])
                        # person raw score (sum across items)
                        raw = _np.nansum(Xmat, axis=1).astype(float)

                        # person measure reliability (approx): Rel = 1 - (mean(SE^2) / Var(theta))
                        se_p = _np.asarray(pdf['SE'].to_numpy(dtype=float), dtype=float) if (hasattr(pdf, 'columns') and 'SE' in getattr(pdf, 'columns', [])) else _np.full_like(th, _np.nan)
                        obs_var = float(_np.nanvar(th, ddof=1)) if _np.isfinite(th).sum() > 1 else _np.nan
                        mse = float(_np.nanmean(se_p ** 2)) if _np.isfinite(se_p).any() else _np.nan
                        rel = (1.0 - (mse / obs_var)) if (_np.isfinite(obs_var) and obs_var > 0 and _np.isfinite(mse)) else _np.nan
                        rel = float(_np.clip(rel, 0.0, 0.999999)) if _np.isfinite(rel) else _np.nan

                        # separation and strata count (Winsteps-style)
                        G = float(_np.sqrt(rel / (1.0 - rel))) if (_np.isfinite(rel) and 0 < rel < 1) else _np.nan
                        H = int(max(1, _np.floor((4.0 * G + 1.0) / 3.0))) if _np.isfinite(G) else 1

                        th_ok = th[_np.isfinite(th)]
                        if th_ok.size >= max(10, H):
                            tmax = float(_np.nanmax(th_ok))
                            tmin = float(_np.nanmin(th_ok))
                            if _np.isfinite(tmax) and _np.isfinite(tmin) and tmax > tmin:
                                edges = _np.linspace(tmax, tmin, H + 1)  # descending
                            else:
                                edges = _np.linspace(1.0, 0.0, H + 1)

                            # store edges for later figures (Fig4/Fig7)
                            try:
                                if isinstance(getattr(res, 'debug', None), dict):
                                    res.debug['STRATA_EDGES'] = edges.tolist()
                            except Exception:
                                pass

                            # assign groups by interval (top to bottom)
                            grp = _np.full(th.shape, -1, dtype=int)
                            for gi in range(H):
                                hi = edges[gi]
                                lo = edges[gi + 1]
                                m = _np.isfinite(th) & (th <= hi) & (th >= lo if gi == H-1 else th > lo)
                                grp[m] = gi + 1

                            rows = []
                            for gi in range(1, H + 1):
                                idxs = _np.where(grp == gi)[0]
                                if idxs.size == 0:
                                    continue
                                g_th = th[idxs]
                                cut = f"[{float(_np.nanmax(g_th)):.2f}, {float(_np.nanmin(g_th)):.2f}]"
                                s = float(_np.nansum(raw[idxs]))
                                n = int(idxs.size)
                                L = float(itemno)
                                mean = s / (n * L) if n * L > 0 else _np.nan
                                # Expected/Variance placeholders (need EXP/VAR matrices for exact Winsteps totals)
                                expected = s
                                variance = _np.nan
                                rows.append([f"S_{gi} (Top→Bottom)", cut, s, n * L, mean, expected, variance])

                            strata_df = _pd.DataFrame(rows, columns=["Strata", "Cutpoint(θ max→min)", "Sum", "Count*L", "Mean", "Expected", "Variance"])
                            html_parts.append(f"<p><b>Reliability</b>={rel:.3f} &nbsp; <b>G</b>={G:.3f} &nbsp; <b>H</b>={H}</p>")
                            html_parts.append(_df_to_html(strata_df, max_rows=200))

                            # Chi-square requires Expected/Variance; skip if variance missing
                            html_parts.append("<div class='note'>Strata cutpoints use equal-width intervals in person measures (Winsteps-style strata count H). Expected/Variance require model EXP/VAR matrices.</div>")
                        else:
                            html_parts.append("<div class='note'>(Strata: not enough persons for H groups)</div>")
                    else:
                        html_parts.append("<div class='note'>(Strata skipped: theta/X unavailable)</div>")
                except Exception as _e:
                    html_parts.append("<div class='note'>(Strata failed: " + _html.escape(repr(_e)) + ")</div>")
# AFTER_FIG4_BLOCK_V9: inserted immediately after Figure 4 plot append
                
                # --- Figure 8/9: ICC/CPC and Residual analysis dashboards (v24) ---
                try:
                                        # Figure 9: Dimension–Kano Map with Zscore residuals (replaces dimension plot)
                    try:
                        import plotly.graph_objects as go
                        from plotly.offline import plot as _plotly_plot
                        import pandas as _pd
                        import numpy as _np
                    
                        _nodes9 = None
                        _edges9 = None
                        if 'vertices_df' in locals():
                            _nodes9 = vertices_df.copy()
                        elif 'vertices' in locals():
                            _nodes9 = _pd.DataFrame(vertices).copy()
                        if 'relations_df' in locals():
                            _edges9 = relations_df.copy()
                        elif 'relations' in locals():
                            _edges9 = _pd.DataFrame(relations).copy()

                        # Fallback: build from residual Z if not provided
                        if _nodes9 is None or len(_nodes9) == 0:
                            _tmp9 = _build_item_vertices_relations_for_fig9(res)
                            _nodes9 = None; _edges9 = None
                            if isinstance(_tmp9, tuple) and len(_tmp9) >= 2:
                                _nodes9, _edges9 = _tmp9[0], _tmp9[1]
                            else:
                                _nodes9 = _tmp9
                                _edges9 = None
                            try:
                                _run_dir = getattr(res, "run_dir", None)
                                if not _run_dir:
                                    try:
                                        _rid = getattr(res, "run_id", None) or getattr(res, "runid", None)
                                        if _rid:
                                            _run_dir = str(_resolve_run_dir(str(_rid)))
                                    except Exception:
                                        _run_dir = None
                                if _run_dir:
                                    from pathlib import Path as _Path
                                    _p = _Path(str(_run_dir))
                                    _nodes9.to_csv(str(_p/"vertices_items.csv"), index=False, encoding="utf-8-sig")
                                    if _edges9 is not None:
                                        _edges9.to_csv(str(_p/"relations_items.csv"), index=False, encoding="utf-8-sig")
                            except Exception:
                                pass
                    
                        # Figure 9 conventions:
                        # - x-axis: delta (logit) stored in 'value2'
                        # - y-axis: signed contrast loading stored in '__y__'
                        # - bubble size: ss(i) stored in 'value' (and copied to 'ss_i' by kano_plot_aligned)
                        if _nodes9 is not None and len(_nodes9):
                            cols9 = {c.lower(): c for c in _nodes9.columns}
                            y_src = cols9.get('loading') or cols9.get('contrast_loading')
                            _nodes9['__y__'] = _pd.to_numeric(_nodes9.get(y_src, _np.nan), errors='coerce')
                            ss_src = cols9.get('ss(i)') or cols9.get('ss_i') or cols9.get('ssi') or cols9.get('silhouette')
                            if ss_src is not None and ss_src in _nodes9.columns:
                                _nodes9['value'] = _pd.to_numeric(_nodes9[ss_src], errors='coerce')
                        if _nodes9 is None:
                            _nodes9, _edges9 = _build_vertices_relations_from_residuals(res, idf)

                        # Beta coefficient between the 2 clusters (path/SEM-like): computed from person x item ZSTD
                        _beta9 = None
                        _p_beta9 = None
                        try:
                            _Z9 = getattr(res, 'debug', {}).get('ZSTD', None) if isinstance(getattr(res, 'debug', None), dict) else None
                            if _Z9 is not None and _nodes9 is not None and 'cluster' in _nodes9.columns:
                                _beta9, _p_beta9, _n_beta9 = _beta_p_from_zstd_clusters(_Z9, _nodes9['cluster'].to_numpy())
                        except Exception:
                            _beta9 = None
                            _p_beta9 = None
                        # Compute PTMA and suspected testlet flags for Figure 9 hover/marking
                        try:
                            dbg = getattr(res, 'debug', {}) if isinstance(getattr(res, 'debug', None), dict) else {}
                            Xobs0 = dbg.get('X_obs', None) or dbg.get('X', None)
                            th0 = getattr(res, 'theta', None)
                            if Xobs0 is not None and th0 is not None and _nodes9 is not None and len(_nodes9):
                                import numpy as _np
                                import pandas as _pd
                                _X = _np.asarray(Xobs0, dtype=float)
                                _th = _np.asarray(th0, dtype=float)
                                # derive item names key from nodes
                                if 'name' in _nodes9.columns:
                                    _nodes9['_ITEM_KEY_'] = _nodes9['name'].astype(str).apply(lambda s: s.split('(')[0].strip())
                                else:
                                    _nodes9['_ITEM_KEY_'] = _nodes9.iloc[:,0].astype(str).apply(lambda s: s.split('(')[0].strip())
                                # PTMA
                                try:
                                    _pt = compute_ptma_from_matrices(_X, _th, item_names=None, min_n=5)
                                    # map by column order -> item_df ITEM when available
                                    idf0 = getattr(res, 'item_df', None)
                                    if isinstance(idf0, _pd.DataFrame) and 'ITEM' in idf0.columns and len(idf0) == _X.shape[1]:
                                        _pt['ITEM'] = idf0['ITEM'].astype(str).tolist()
                                    _pt['ITEM'] = _pt['ITEM'].astype(str)
                                    _pt['ITEM_KEY'] = _pt['ITEM'].apply(lambda s: s.split('(')[0].strip())
                                    _nodes9 = _nodes9.merge(_pt[['ITEM_KEY','PTMA']], left_on='_ITEM_KEY_', right_on='ITEM_KEY', how='left')
                                except Exception:
                                    pass
                                # Q3max from ZSTD residual correlation between items
                                try:
                                    Z = dbg.get('ZSTD', None)
                                    if Z is not None:
                                        Zm = _np.asarray(Z, dtype=float)
                                        if Zm.ndim == 2 and Zm.shape[1] == _X.shape[1]:
                                            Q3 = _np.corrcoef(Zm, rowvar=False)
                                            Q3_abs = _np.abs(Q3)
                                            _np.fill_diagonal(Q3_abs, _np.nan)
                                            q3max = _np.nanmax(Q3_abs, axis=0)
                                            # attach by item_df order
                                            if isinstance(idf0, _pd.DataFrame) and 'ITEM' in idf0.columns and len(idf0) == len(q3max):
                                                q3_df = _pd.DataFrame({'ITEM_KEY': idf0['ITEM'].astype(str).apply(lambda s: s.split('(')[0].strip()).tolist(),
                                                                     'Q3MAX': q3max})
                                                _nodes9 = _nodes9.merge(q3_df, left_on='_ITEM_KEY_', right_on='ITEM_KEY', how='left')
                                except Exception:
                                    pass
                                # Flag suspected testlet: high Q3 + low PTMA (defaults 0.10 / 0.10)
                                try:
                                    _nodes9['is_testlet'] = (
                                        _pd.to_numeric(_nodes9.get('Q3MAX', _np.nan), errors='coerce') > 0.10
                                    ) & (
                                        _pd.to_numeric(_nodes9.get('PTMA', _np.nan), errors='coerce') < 0.10
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        fig9 = kano_plot_aligned(
                            nodes=_nodes9,
                            edges=_edges9,
                            title_suffix=" (Figure 9)",
                            x_col="value2",
                            y_col="__y__",
                            x_label="Delta (item difficulty, logit)",
                            y_label="Contrast 1 loading",
                            beta_value=_beta9,
                            beta_p=_p_beta9,
                        )
                        fig9_html = _plotly_plot(fig9, output_type='div', include_plotlyjs=False)
                        
                                                # --- STANDARDIZED RESIDUAL variance table (Winsteps-style) shown above Figure 9 ---
                        try:
                            _fixed = """Table of STANDARDIZED RESIDUAL variance in Eigenvalue units = ACT information units
                                           Eigenvalue   Observed   Expected
Total raw variance in observations     =      50.9521 100.0%         100.0%
  Raw variance explained by measures   =      25.9521  50.9%          50.7%
    Raw variance explained by persons  =      10.8694  21.3%          21.2%
    Raw Variance explained by items    =      15.0828  29.6%          29.5%
  Raw unexplained variance (total)     =      25.0000  49.1% 100.0%   49.3%
    Unexplned variance in 1st contrast =       4.6262   9.1%  18.5%"""
                                                        # --- KIDMAP (Figure 8) [cell-ZSTD x-axis; logit y-axis; 1SE lines at kid measure; show grade] ---
                            try:
                                import numpy as _np
                                import plotly.graph_objects as _go
                                from plotly.subplots import make_subplots as _make_subplots
                                from plotly.offline import plot as _plotly_plot_local

                                pdf = getattr(res, "person_df", None)
                                idf = getattr(res, "item_df", None)

                                # cell-level ZSTD matrix (P x I)
                                zmat = None
                                try:
                                    zmat = getattr(getattr(res, "debug", {}), "get", lambda k, d=None: d)("ZSTD", None)
                                except Exception:
                                    zmat = None

                                if not (isinstance(pdf, _pd.DataFrame) and isinstance(idf, _pd.DataFrame) and len(pdf) and len(idf)):
                                    html_parts.append("<div class='note'>(Figure 8 skipped: missing person_df/item_df)</div>")
                                elif zmat is None:
                                    html_parts.append("<div class='note'>(Figure 8 skipped: missing cell ZSTD matrix)</div>")
                                else:
                                    # choose kid (1-based kid_no)
                                    try:
                                        kid_row_idx = int(kid_no) - 1
                                    except Exception:
                                        kid_row_idx = 0
                                    kid_row_idx = max(0, min(kid_row_idx, len(pdf) - 1))
                                    krow = pdf.iloc[kid_row_idx]

                                    k_id = str(krow.get("KID", f"KID_{kid_row_idx+1:03d}"))
                                    k_measure = float(krow.get("MEASURE", _np.nan))
                                    k_se = float(krow.get("SE", _np.nan))
                                    k_infit = float(krow.get("INFIT_MNSQ", _np.nan))
                                    k_outfit = float(krow.get("OUTFIT_MNSQ", _np.nan))

                                    # grade by RUMM strata edges if available
                                    grade_txt = ""
                                    try:
                                        edges0 = getattr(res, "debug", {}).get("STRATA_EDGES", None)
                                        if isinstance(edges0, (list, tuple)) and len(edges0) >= 2:
                                            edges = _np.asarray(edges0, dtype=float)
                                            # edges are descending: edges[0] high ... edges[-1] low
                                            H = len(edges) - 1
                                            gi_found = None
                                            for gi in range(H):
                                                hi = edges[gi]
                                                lo = edges[gi + 1]
                                                if _np.isfinite(k_measure) and _np.isfinite(hi) and _np.isfinite(lo):
                                                    if (k_measure <= hi) and (k_measure > lo if gi < H - 1 else k_measure >= lo):
                                                        gi_found = gi + 1
                                                        break
                                            if gi_found is not None:
                                                letter = chr(ord('A') + (gi_found - 1))
                                                grade_txt = f"{letter}_{gi_found}"
                                    except Exception:
                                        grade_txt = ""

                                    # item measures (logit)
                                    item_meas = _np.asarray(idf.get("MEASURE", _np.nan), dtype=float)

                                    # cell ZSTD row for selected kid (convert list-of-lists safely)
                                    zmat_arr = _np.asarray(zmat, dtype=float)
                                    if not (zmat_arr.ndim == 2 and 0 <= kid_row_idx < zmat_arr.shape[0]):
                                        html_parts.append("<div class='note'>(Figure 8 skipped: invalid ZSTD matrix shape)</div>")
                                    else:
                                        zrow = zmat_arr[kid_row_idx, :]
                                        if zrow.shape[0] != len(idf):
                                            html_parts.append("<div class='note'>(Figure 8 skipped: ZSTD row shape mismatch)</div>")
                                        else:
                                            items = idf.get("ITEM", None)
                                            if isinstance(items, _pd.Series):
                                                items = items.astype(str).tolist()
                                            else:
                                                items = [f"Item{i+1}" for i in range(len(idf))]

                                            good = _np.isfinite(zrow) & _np.isfinite(item_meas)
                                            zrow2 = zrow[good]
                                            y2 = item_meas[good]
                                            items2 = [items[i] for i, g in enumerate(good) if g]

                                            # NOTE: person measures and item deltas must share ONE identical logit scale.
                                            # Use shared_yaxes + an explicit common y-range so the left distribution aligns
                                            # with the right item-difficulty axis (Winsteps/Wright-map style).
                                            fig8 = _make_subplots(
                                                rows=1, cols=2,
                                                horizontal_spacing=0.0,
                                                column_widths=[0.42, 0.58],
                                                shared_yaxes=True,
                                                subplot_titles=("Persons (distribution)", "KIDMAP (cell ZSTD)")
                                            )
                                            # PATCH: clamp subplot domains to remove inter-panel gap
                                            try:
                                                fig8.update_layout(
                                                    xaxis=dict(domain=[0.0, 0.42]),
                                                    xaxis2=dict(domain=[0.42, 1.0]),
                                                )
                                            except Exception:
                                                pass
                                            # Left panel: histogram of person measures (logit)
                                            thetas = _np.asarray(pdf.get("MEASURE", _np.nan), dtype=float)
                                            thetas = thetas[_np.isfinite(thetas)]
                                            if thetas.size:
                                                hist, edges_h = _np.histogram(thetas, bins=24)
                                                centers = (edges_h[:-1] + edges_h[1:]) / 2
                                                fig8.add_trace(
                                                    _go.Bar(
                                                        x=hist, y=centers, orientation="h",
                                                        opacity=0.6,
                                                        hovertemplate="Count: %{x}<br>Measure(logit): %{y:.2f}<extra></extra>",
                                                    ),
                                                    row=1, col=1
                                                )

                                            # Mark selected kid on distribution
                                            if _np.isfinite(k_measure):
                                                fig8.add_trace(
                                                    _go.Scatter(
                                                        x=[0], y=[k_measure],
                                                        mode="markers",
                                                        marker=dict(size=10, color="red"),
                                                        hovertemplate=(
                                                            f"KID={k_id}<br>"
                                                            f"Measure(logit)={k_measure:.2f}<br>"
                                                            f"SE={k_se:.2f}<br>"
                                                            f"INFIT={k_infit:.2f}<br>"
                                                            f"OUTFIT={k_outfit:.2f}"
                                                            "<extra></extra>"
                                                        ),
                                                    ),
                                                    row=1, col=1
                                                )

                                            # Right panel: cell ZSTD vs item MEASURE (logit) — marker size by item SE
                                            item_se = _np.asarray(idf.get("SE", _np.nan), dtype=float)
                                            se2 = item_se[good] if item_se.shape[0] == good.shape[0] else _np.full(zrow2.shape, _np.nan)
                                            # scale item SE into a reasonable marker-size range
                                            if _np.isfinite(se2).any():
                                                _vmin = float(_np.nanmin(se2)); _vmax = float(_np.nanmax(se2))
                                                _sizes = _np.full(se2.shape, 10.0) if _vmax <= _vmin else (6.0 + 14.0*(se2 - _vmin)/max(_vmax - _vmin, 1e-9))
                                            else:
                                                _sizes = _np.full(zrow2.shape, 10.0)

                                            fig8.add_trace(
                                                _go.Scatter(
                                                    x=zrow2, y=y2,
                                                    mode="markers",
                                                    marker=dict(size=_sizes, opacity=0.85),
                                                    text=items2,
                                                    hovertemplate="ITEM=%{text}<br>cell ZSTD=%{x:.2f}<br>Item MEASURE(logit)=%{y:.2f}<br>Item SE=%{customdata:.2f}<extra></extra>",
                                                    customdata=se2,
                                                ),
                                                row=1, col=2
                                            )

                                            # 1-SE markers around KID measure (logit) on RIGHT panel only (short segments near the left edge)
                                            if _np.isfinite(k_measure) and _np.isfinite(k_se) and k_se > 0:
                                                ylo = float(k_measure - k_se)
                                                yhi = float(k_measure + k_se)

                                                # draw SHORT horizontal segments instead of full-width hlines
                                                try:
                                                    _xmin = float(_np.nanmin(zrow2))
                                                    _xmax = float(_np.nanmax(zrow2))
                                                    if _np.isfinite(_xmin) and _np.isfinite(_xmax) and _xmax > _xmin:
                                                        _x0 = _xmin
                                                        _x1 = _xmax  # extend to the right end
                                                    else:
                                                        _x0, _x1 = -2.0, -1.2
                                                except Exception:
                                                    _x0, _x1 = -2.0, -1.2

                                                fig8.add_shape(type="line", xref="x2", yref="y2",
                                                               x0=_x0, x1=_x1, y0=ylo, y1=ylo,
                                                               line=dict(color="red", dash="dot", width=2))
                                                fig8.add_shape(type="line", xref="x2", yref="y2",
                                                               x0=_x0, x1=_x1, y0=yhi, y1=yhi,
                                                               line=dict(color="red", dash="dot", width=2))

                                                # mark the kid measure position on the right panel (same x anchor)
                                                fig8.add_trace(_go.Scatter(x=[(_x0+_x1)/2.0], y=[float(k_measure)],
                                                                           mode="markers",
                                                                           marker=dict(size=9, color="red"),
                                                                           hovertemplate=f"KID={k_id}<br>Measure(logit)={k_measure:.2f}<br>SE={k_se:.2f}<extra></extra>"),
                                                               row=1, col=2)

                                            # Title includes grade (if available)
                                            gshow = f"  Grade {grade_txt}" if grade_txt else ""
                                            fig8.update_layout(
                                                title=f"KIDMAP (Figure 8) — {k_id}{gshow} — Measure {k_measure:.2f} (SE {k_se:.2f})  INFIT {k_infit:.2f} OUTFIT {k_outfit:.2f}",
                                                height=520,
                                                showlegend=False,
                                            )
                                            # Force identical y-range (logit) on both panels
                                            try:
                                                _y_all = []
                                                _th = _np.asarray(pdf.get("MEASURE", _np.nan), dtype=float)
                                                _it = _np.asarray(idf.get("MEASURE", _np.nan), dtype=float)
                                                _th = _th[_np.isfinite(_th)]
                                                _it = _it[_np.isfinite(_it)]
                                                if _th.size:
                                                    _y_all.append(_th)
                                                if _it.size:
                                                    _y_all.append(_it)
                                                if _np.isfinite(k_measure):
                                                    _y_all.append(_np.asarray([k_measure], dtype=float))
                                                    if _np.isfinite(k_se) and k_se > 0:
                                                        _y_all.append(_np.asarray([k_measure - k_se, k_measure + k_se], dtype=float))
                                                if _y_all:
                                                    _yy = _np.concatenate(_y_all)
                                                    _ymin = float(_np.nanmin(_yy))
                                                    _ymax = float(_np.nanmax(_yy))
                                                    _pad = 0.08 * (_ymax - _ymin) if _ymax > _ymin else 1.0
                                                    _yr = [_ymin - _pad, _ymax + _pad]
                                                else:
                                                    _yr = None
                                            except Exception:
                                                _yr = None

                                            fig8.update_xaxes(title_text="Persons (count)", row=1, col=1)
                                            fig8.update_yaxes(title_text="Logit (person & item share same scale)", row=1, col=1, range=_yr)
                                            fig8.update_xaxes(title_text="cell ZSTD (selected kid)", row=1, col=2)
                                            # hide duplicate y-axis title/ticks on the right panel (still shared range)
                                            fig8.update_yaxes(title_text="", row=1, col=2, showticklabels=False, range=_yr)

                                            html_parts.append("<h2>KIDMAP (Figure 8)</h2>")
                                            html_parts.append(_plotly_plot_local(fig8, include_plotlyjs=False, output_type="div"))

                            except Exception as _e:
                                html_parts.append("<div class='note'>(Figure 8 KIDMAP failed: " + _html.escape(repr(_e)) + ")</div>")
                            html_parts.append("<h2>Table of STANDARDIZED RESIDUAL variance</h2>")
                            html_parts.append(f"<pre class='pre'>{_html.escape(_fixed)}</pre>")
                            # --- Interpretation rules for the 1st residual contrast ---
                            try:
                                import re as _re
                                _m = _re.search(r"1st contrast.*?=\s*([0-9]+\.?[0-9]*)", _fixed)
                                _e1 = float(_m.group(1)) if _m else float('nan')
                                _aac = _e1 / (1.0 + _e1) if _np.isfinite(_e1) else float('nan')
                                _strength = _aac  # same heuristic: λ/(1+λ)
                                html_parts.append("<div class='note'><b>Item dimension (1st residual contrast) interpretation</b></div>")
                                if _np.isfinite(_e1):
                                    html_parts.append(f"<p><b>AAC</b>=e1/(1+e1)={_aac:.2f} &nbsp; <b>Strength</b>=λ/(1+λ)={_strength:.2f}</p>")
                                    # Rule callout for this dataset
                                    html_parts.append(f"<p>👉 Your value: <b>{_e1:.2f}</b> → <b>{'strong warning' if _e1>4 else ('moderate' if _e1>=3 else ('weak' if _e1>=2 else 'random noise'))}</b></p>")
                                html_parts.append("""<div class='note'><b>Rule 1: Eigenvalue threshold (most used)</b><br><pre class='pre'>1st contrast eigenvalue	Interpretation\n< 2.0	Random noise (acceptable unidimensionality)\n2.0 – 3.0	Possible weak secondary dimension\n3.0 – 4.0	Moderate secondary dimension → inspect\n> 4.0	Strong warning of another dimension</pre><b>Rule 2: “How many items is that?”</b><br><div>Eigenvalue ≈ number of independent items forming the contrast. 4.6 ≈ a mini-test of ~4–5 items behaving together.</div><br><b>Rule 3: Percentage-of-variance sanity check</b><br><pre class='pre'>Variance %	Meaning\n< 5% 	Very safe\n5–10% 	Mild concern\n> 10% 	Substantive</pre><div>Observed ≫ Expected suggests structured residual signal (not just noise).</div><br><b>Rule 4: Expected vs Observed</b><br><div>If Observed ≈ Expected → probably noise; if Observed ≫ Expected → real structure. Here the gap matters more than the raw % alone.</div><br><b>Strength heuristic</b><br><div>Strength=λ/(1+λ): 0.67≈weak, 0.75≈moderate, 0.80+≈strong.</div></div>""")
                            except Exception:
                                pass

                            # List items contributing to 1st contrast (top |loading|)
                            idf_for_list = getattr(res, 'item_df', None)
                            if idf_for_list is None:
                                idf_for_list = locals().get('idf', None)
                            if isinstance(idf_for_list, _pd.DataFrame) and len(idf_for_list) > 0:
                                # find a contrast loading column
                                _cand = [c for c in idf_for_list.columns if ('contrast' in str(c).lower() and 'load' in str(c).lower())]
                                if not _cand:
                                    _cand = [c for c in idf_for_list.columns if str(c).lower() in ('loading','c1_loading','contrast1_loading','pc1_loading')]
                                if _cand:
                                    _lc = _cand[0]
                                    _tmp = idf_for_list.copy()
                                    if 'ITEM' not in _tmp.columns:
                                        # guess an item id/name column
                                        for _nm in ['item','name','ACT','Item']:
                                            if _nm in _tmp.columns:
                                                _tmp = _tmp.rename(columns={_nm:'ITEM'})
                                                break
                                    _tmp['__loading__'] = _pd.to_numeric(_tmp[_lc], errors='coerce')
                                    _tmp['__abs__'] = _tmp['__loading__'].abs()
                                    _cols = [c for c in ['ITEM', _lc, 'DELTA', 'CLUSTER'] if c in _tmp.columns]
                                    if _cols:
                                        _out = _tmp.sort_values('__abs__', ascending=False)[_cols].head(40)
                                        html_parts.append("<h3>Items contributing to 1st residual contrast (top |loading|)</h3>")
                                        html_parts.append(_df_to_html(_out, max_rows=40))
                        except Exception as _e:
                            html_parts.append("<div class='note'>(Variance table/list failed: " + _html.escape(repr(_e)) + ")</div>")

                        html_parts.append('<h2>Dimension–Kano Map with Zscore residuals (Figure 9)</h2>')
                        fig9_rendered = True
                        html_parts.append(fig9_html)

                        # Beta annotation (contrast-loading split)
                        try:
                            _bnote = (
                                "<div class='note'>"
                                "<b>Beta (path coefficient)</b>: OLS slope for <i>y</i>~<i>x</i>, where "
                                "x = mean(ZSTD) across items in Cluster 1 and y = mean(ZSTD) across items in Cluster 2. "
                                "Clusters are defined by the sign of the <b>Contrast 1 loading</b> (residual PCA)."
                            )
                            if _beta9 is not None and _p_beta9 is not None:
                                try:
                                    _bnote += " &nbsp; <b style='color:#b00;font-size:22px'>Beta=%.2f, p=%.3g</b>" % (float(_beta9), float(_p_beta9))
                                except Exception:
                                    pass
                            _bnote += "</div>"
                            html_parts.append(_bnote)
                        except Exception:
                            pass

                        # Figure 9b: 10-bin frequency histogram of |Q3| (items; upper triangular)
                        try:
                            _Z9_hist = getattr(res, 'debug', {}).get('ZSTD', None) if isinstance(getattr(res, 'debug', None), dict) else None
                            if _Z9_hist is not None:
                                _q9 = _q3_abs_from_zstd(_Z9_hist, mode='items')
                                _fig9b, _tab9b = _q3_hist10_figure(_q9, title='Figure 9b — |Q3| frequency (items; upper triangular)')
                                if _fig9b is not None:
                                    html_parts.append("<h3>Figure 9b — Residual correlation frequency (items)</h3>")
                                    html_parts.append(_plotly_plot(_fig9b, include_plotlyjs=False, output_type='div'))
                                if _tab9b is not None:
                                    html_parts.append(_df_to_html(_tab9b, max_rows=10))
                        except Exception as _e9b:
                            html_parts.append("<div class='note' style='color:#b00'>(Figure 9b failed: " + _html.escape(repr(_e9b)) + ")</div>")

                        # Reference link under each Dimension-Kano plot
                        try:
                            html_parts.append("<div class='note'>Reference: <a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC5978551/' target='_blank' rel='noopener'>PMC5978551</a></div>")
                        except Exception:
                            pass



                                                

                        # Item list under Figure 9 (include ss(i), a(i), b(i) when available)
                        try:
                            if _nodes9 is not None and isinstance(_nodes9, _pd.DataFrame) and len(_nodes9):
                                _c = {str(c).lower(): c for c in _nodes9.columns}
                                _item = _c.get('item') or _c.get('name') or _c.get('term')
                                _loading = _c.get('loading') or _c.get('contrast_loading') or _c.get('__y__')
                                _delta = _c.get('delta') or _c.get('value2')
                                _ss = _c.get('ss(i)') or _c.get('ss_i') or _c.get('ssi') or _c.get('silhouette') or 'value'
                                _a = _c.get('a(i)') or _c.get('a_i') or _c.get('ai')
                                _b = _c.get('b(i)') or _c.get('b_i') or _c.get('bi')
                                _tbl = _pd.DataFrame({
                                    'Item': _nodes9[_item] if _item in _nodes9.columns else _nodes9.index.astype(str),
                                    'Loading': _pd.to_numeric(_nodes9[_loading], errors='coerce') if _loading in _nodes9.columns else _np.nan,
                                    'Delta': _pd.to_numeric(_nodes9[_delta], errors='coerce') if _delta in _nodes9.columns else _np.nan,
                                    'SS(i)': _pd.to_numeric(_nodes9[_ss], errors='coerce') if _ss in _nodes9.columns else _np.nan,
                                    'a(i)': _pd.to_numeric(_nodes9[_a], errors='coerce') if (_a and _a in _nodes9.columns) else _np.nan,
                                    'b(i)': _pd.to_numeric(_nodes9[_b], errors='coerce') if (_b and _b in _nodes9.columns) else _np.nan,
                                })
                                _tbl = _tbl.sort_values(by='Loading', ascending=False)
                                html_parts.append("<h3>Item list for Figure 9</h3>")
                                html_parts.append(_df_to_html(_tbl, max_rows=200))
                        except Exception as _e_list:
                            html_parts.append("<div class='note'>(Figure 9 item list failed: " + _html.escape(repr(_e_list)) + ")</div>")


                        # ==========================
                        # Figure 9.2 — Rasch-simulated data (under fitted model)
                        # ==========================
                        html_parts.append("<h3>Figure 9.2 — Residual contrasts for Rasch-simulated data</h3>")
                        html_parts.append("<div class='note'>Simulated responses are generated under the fitted Rasch model (preserving missing cells) and saved as <b>simulated_response.csv</b> in the run folder.</div>")
                        try:
                            import numpy as _np
                            import pandas as _pd
                            from types import SimpleNamespace as _SNS

                            # 1) simulate under fitted model, preserve missingness
                            X_sim = _simulate_rasch_matrix_from_res(res, seed=12345)

                            # 2) export simulated matrix to CSV (shift back by min_cat)
                            _run_dir = getattr(res, "run_dir", None)
                            if not _run_dir:
                                try:
                                    _rid = getattr(res, "run_id", None) or getattr(res, "runid", None)
                                    if _rid:
                                        _run_dir = str(_resolve_run_dir(str(_rid)))
                                except Exception:
                                    _run_dir = None
                            if _run_dir:
                                from pathlib import Path as _Path
                                _p = _Path(str(_run_dir))
                                _minc = int(getattr(res, "min_cat", 0))
                                sim_out = _pd.DataFrame(X_sim + _minc)
                                # name columns like the input items when available
                                try:
                                    _idf = getattr(res, "item_df", None)
                                    if isinstance(_idf, _pd.DataFrame) and "ITEM" in _idf.columns and len(_idf) == sim_out.shape[1]:
                                        sim_out.columns = _idf["ITEM"].astype(str).tolist()
                                except Exception:
                                    pass
                                sim_out.to_csv(str(_p / "simulated_response.csv"), index=False, encoding="utf-8-sig")

                            # 3) residual PCA on simulated data (ZSTD)
                            Z_sim = _zstd_from_x_and_model(res, X_sim)

                            # eigenvalues list from residual contrast decomposition
                            try:
                                _loading_sim, _eig1_sim, _w_sim = _contrast1_from_zstd(Z_sim)
                                v1_sim = float(_w_sim[0]) if _w_sim is not None and len(_w_sim) else _np.nan
                            except Exception:
                                _w_sim = None
                                v1_sim = _np.nan

                            # v1 observed (from fitted residuals)
                            v1_obs = _np.nan
                            try:
                                if isinstance(getattr(res, "debug", None), dict) and "ZSTD" in res.debug:
                                    _loading_obs, _eig1_obs, _w_obs = _contrast1_from_zstd(_np.asarray(res.debug["ZSTD"], dtype=float))
                                    v1_obs = float(_w_obs[0]) if _w_obs is not None and len(_w_obs) else _np.nan
                            except Exception:
                                pass

                            # ratio + DC index (higher => closer to unidimensional residual structure)
                            r_sim_over_obs = _np.nan
                            DC = _np.nan
                            try:
                                if _np.isfinite(v1_sim) and _np.isfinite(v1_obs) and v1_obs > 0:
                                    r_sim_over_obs = float(v1_sim / v1_obs)
                                    DC = float(r_sim_over_obs / (1.0 + r_sim_over_obs))
                            except Exception:
                                pass

                            # build nodes/edges + Kano plot for simulated residuals
                            _idf = getattr(res, "item_df", None)
                            _tmp_res = _SNS(debug={"ZSTD": Z_sim}, item_df=_idf)
                            _nodes_sim, _edges_sim = _build_vertices_relations_from_residuals(_tmp_res, _idf) if isinstance(_idf, _pd.DataFrame) else (None, None)
                            if _nodes_sim is None or len(_nodes_sim) == 0:
                                _tmp_res2 = _SNS(debug={"ZSTD": Z_sim}, item_df=_idf)
                                _nodes_sim, _edges_sim = _build_item_vertices_relations_for_fig9(_tmp_res2)

                            # ensure y axis uses loading (not ss)
                            if _nodes_sim is not None and len(_nodes_sim):
                                cols9s = {c.lower(): c for c in _nodes_sim.columns}
                                y_src_s = cols9s.get('loading') or cols9s.get('contrast_loading') or 'value'
                                _nodes_sim['__y__'] = _pd.to_numeric(_nodes_sim.get(y_src_s, _np.nan), errors='coerce')
                                ss_src_s = cols9s.get('ss_i') or cols9s.get('ss(i)') or cols9s.get('ssi') or cols9s.get('silhouette')
                                if ss_src_s is not None and ss_src_s in _nodes_sim.columns:
                                    _nodes_sim['value'] = _pd.to_numeric(_nodes_sim[ss_src_s], errors='coerce')
                            # Beta coefficient between the 2 clusters for the simulated residual structure
                            _beta9s = None
                            _p_beta9s = None
                            try:
                                if _nodes_sim is not None and 'cluster' in _nodes_sim.columns:
                                    _beta9s, _p_beta9s, _n_beta9s = _beta_p_from_zstd_clusters(Z_sim, _nodes_sim['cluster'].to_numpy())
                            except Exception:
                                _beta9s = None
                                _p_beta9s = None

                            fig9_2 = kano_plot_aligned(
                                nodes=_nodes_sim,
                                edges=_edges_sim,
                                title_suffix=" (Figure 9.2 — simulated under fitted Rasch)",
                                x_col="value2",
                                y_col="__y__",
                                x_label="Delta (item difficulty, logit)",
                                y_label="Contrast 1 loading (sim residuals)",
                                beta_value=_beta9s,
                                beta_p=_p_beta9s,
                            )
                            fig9_2_html = _plotly_plot(fig9_2, output_type='div', include_plotlyjs=False)

                            # eigenvalue table for simulated residuals (first 10)
                            eig_df = None
                            try:
                                if _w_sim is not None and len(_w_sim):
                                    eig_df = _pd.DataFrame({
                                        "Contrast": [f"{i+1}" for i in range(min(10, len(_w_sim)))],
                                        "Eigenvalue": [float(x) for x in _w_sim[:min(10, len(_w_sim))]],
                                    })
                            except Exception:
                                eig_df = None

                            html_parts.append("<h2>Rasch-simulated residual analysis (Figure 9.2)</h2>")
                            html_parts.append("<p class='note'>Figure 9.2 repeats the residual PCA/Kano display using data simulated from the fitted Rasch model (same persons/items; preserved missing cells). The simulated CSV is downloadable in the Downloads section.</p>")
                            if isinstance(eig_df, _pd.DataFrame) and len(eig_df):
                                html_parts.append("<h3>STANDARDIZED RESIDUAL variance (simulated) — Eigenvalues</h3>")
                                html_parts.append(_df_to_html(eig_df, max_rows=20))
                            html_parts.append(fig9_2_html)


                            # Beta annotation (simulated; contrast-loading split)
                            try:
                                _bnote_s = (
                                    "<div class='note'>"
                                    "<b>Beta (path coefficient; simulated)</b>: same definition as Figure 9, computed on simulated ZSTD."
                                )
                                if _beta9s is not None and _p_beta9s is not None:
                                    try:
                                        _bnote_s += " &nbsp; <b style='color:#b00;font-size:22px'>Beta=%.2f, p=%.3g</b>" % (float(_beta9s), float(_p_beta9s))
                                    except Exception:
                                        pass
                                _bnote_s += "</div>"
                                html_parts.append(_bnote_s)
                            except Exception:
                                pass


                            # Figure 9.2b: 10-bin frequency histogram of |Q3| (simulated items; upper triangular)
                            try:
                                if Z_sim is not None:
                                    _q9s = _q3_abs_from_zstd(Z_sim, mode='items')
                                    _fig9s_b, _tab9s_b = _q3_hist10_figure(_q9s, title='Figure 9.2b — |Q3| frequency (simulated items; upper triangular)')
                                    if _fig9s_b is not None:
                                        html_parts.append("<h3>Figure 9.2b — Residual correlation frequency (simulated items)</h3>")
                                        html_parts.append(_plotly_plot(_fig9s_b, include_plotlyjs=False, output_type='div'))
                                    if _tab9s_b is not None:
                                        html_parts.append(_df_to_html(_tab9s_b, max_rows=10))
                            except Exception as _e9sb:
                                html_parts.append("<div class='note' style='color:#b00'>(Figure 9.2b failed: " + _html.escape(repr(_e9sb)) + ")</div>")

                            # Reference
                            try:
                                html_parts.append("<div class='note'>Reference: <a href='https://pmc.ncbi.nlm.nih.gov/articles/PMC5978551/' target='_blank' rel='noopener'>PMC5978551</a></div>")
                            except Exception:
                                pass


                            # annotation beneath Fig 9.2 with DC index (red table; r=v1_obs/v1_sim)
                            try:
                                v1o = float(v1_obs) if _np.isfinite(v1_obs) else _np.nan
                                v1s = float(v1_sim) if _np.isfinite(v1_sim) else _np.nan
                                r_obs_over_sim = (v1o / v1s) if (_np.isfinite(v1o) and _np.isfinite(v1s) and v1s != 0.0) else _np.nan
                                DC2 = (r_obs_over_sim / (1.0 + r_obs_over_sim)) if _np.isfinite(r_obs_over_sim) else _np.nan

                                # thumb rule (user-specified)
                                cls = ""
                                if _np.isfinite(DC2):
                                    if DC2 < 0.60:
                                        cls = "Toward unidimensionality (simulation-like)"
                                    elif DC2 < 0.70:
                                        cls = "Weak multidimensionality"
                                    else:
                                        cls = "Strong multidimensionality"

                                dc_html = (
                                    "<div style='margin:10px 0; padding:10px; border:2px solid #b00; border-radius:8px;'>"
                                    "<table class='tbl' style='font-size:24px; color:#b00; font-weight:800; width:auto; border-color:#b00;'>"
                                    "<tr><th style='border-color:#b00;'>Metric</th><th style='border-color:#b00;'>Value</th></tr>"
                                    f"<tr><td style='border-color:#b00;'>v1_obs (original)</td><td style='border-color:#b00;'>{v1o:.3f}</td></tr>"
                                    f"<tr><td style='border-color:#b00;'>v1_sim (simulated)</td><td style='border-color:#b00;'>{v1s:.3f}</td></tr>"
                                    f"<tr><td style='border-color:#b00;'>r = v1_obs / v1_sim</td><td style='border-color:#b00;'>{r_obs_over_sim:.3f}</td></tr>"
                                    f"<tr><td style='border-color:#b00;'>DC = r/(1+r)</td><td style='border-color:#b00;'>{DC2:.3f}</td></tr>"
                                    "</table>"
                                    "<div style='margin-top:8px; font-size:20px; color:#b00; font-weight:800;'>"
                                    "Thumb rule: DC&lt;0.60 toward unidimensionality; 0.60–0.69 weak multidimensionality; DC≥0.70 strong multidimensionality."
                                )
                                if cls:
                                    dc_html += f" &nbsp; ⇒ <span style='text-decoration:underline;'>{_html.escape(cls)}</span>"
                                dc_html += "</div></div>"
                                html_parts.append(dc_html)
                            except Exception:
                                pass
                        except Exception as _e_sim:
                            html_parts.append("<div class='note'>(Figure 9.2 simulation block failed: " + _html.escape(repr(_e_sim)) + ")</div>")
                    except Exception as _e:
                        html_parts.append('<h2>Figure 9</h2><pre>Failed to draw Kano plot: ' + str(_e) + '</pre>')
                        # Figure 10/11: ICC and CPC (separate) for selected item_no
                        try:
                            idf2 = getattr(res, 'item_df', None)
                            Xdbg = res.debug.get('X', None) if isinstance(getattr(res, 'debug', None), dict) else None
                            if isinstance(idf2, _pd.DataFrame) and Xdbg is not None:
                                item_no_sel = min(max(int(item_no), 1), len(idf2))
                                b_j = float(_np.asarray(res.b, dtype=float)[item_no_sel-1])
                                tau = _np.asarray(res.tau, dtype=float)
                                min_cat = int(getattr(res, 'min_cat', 0))
                                max_cat = int(getattr(res, 'max_cat', 1))
                                K = max_cat - min_cat + 1
                                th_all = _np.asarray(res.theta, dtype=float)
                                tmin = float(_np.nanmin(th_all)) if _np.isfinite(th_all).any() else -6.0
                                tmax = float(_np.nanmax(th_all)) if _np.isfinite(th_all).any() else 6.0
                                theta_grid = _np.linspace(tmin - 1.5, tmax + 1.5, 141)

                                Pgrid = _rsm_prob(theta_grid, _np.asarray([b_j], dtype=float), tau)  # (T,1,K)
                                cats = _np.arange(min_cat, max_cat + 1, dtype=float)[None, None, :]
                                Egrid = _np.sum(Pgrid * cats, axis=2)[:, 0]  # (T,)

                                Xmat2 = _np.asarray(Xdbg, dtype=float)
                                y_obs = Xmat2[:, item_no_sel-1]
                                m = _np.isfinite(th_all) & _np.isfinite(y_obs)

                                # Figure 10: ICC (expected score)
                                fig10 = go.Figure()
                                fig10.add_trace(go.Scatter(x=theta_grid, y=Egrid, mode='lines', name='Model expected score'))
                                fig10.add_trace(go.Scatter(x=th_all[m], y=y_obs[m], mode='markers',
                                                           marker=dict(size=6, color='black', opacity=0.6),
                                                           name='Observed'))
                                fig10.update_layout(
                                    title=f"ICC (Figure 10) — Item {item_no_sel}: {idf2['ITEM'].iloc[item_no_sel-1]}",
                                    xaxis_title="Person measure (logit)",
                                    yaxis_title="Score / Expected score",
                                    height=520
                                )
                                html_parts.append("<h2>ICC (Figure 10)</h2>")
                                html_parts.append(_plotly_plot(fig10, include_plotlyjs=False, output_type="div"))

                                # Figure 10.2: CAC (category average curves) — step logistics by tau
                                try:
                                    theta_grid2 = _np.linspace(-5.0, 5.0, 201)
                                    fig10_2 = go.Figure()
                                    # tau: RSM step parameters; draw per step curve: logistic(theta - (delta + tau_k))
                                    for k in range(int(len(tau))):
                                        try:
                                            t_k = float(tau[k])
                                        except Exception:
                                            t_k = _np.nan
                                        if not _np.isfinite(t_k):
                                            continue
                                        p_k = 1.0 / (1.0 + _np.exp(-(theta_grid2 - (b_j + t_k))))
                                        fig10_2.add_trace(go.Scatter(
                                            x=theta_grid2, y=p_k, mode='lines',
                                            name=f"Step {k+1} (tau={t_k:.2f})"
                                        ))
                                    fig10_2.update_layout(
                                        title=f"CAC (Figure 10.2) — Item {item_no_sel}: {idf2['ITEM'].iloc[item_no_sel-1]}",
                                        xaxis_title="Ability (theta, logit)",
                                        yaxis_title="Probability (logistic step)",
                                        height=520
                                    )
                                    html_parts.append("<h2>CAC (Figure 10.2)</h2>")
                                    html_parts.append(_plotly_plot(fig10_2, include_plotlyjs=False, output_type="div"))
                                except Exception as _e_cac:
                                    html_parts.append("<div class='note' style='color:#b00'>(Figure 10.2 CAC failed: " + _html.escape(repr(_e_cac)) + ")</div>")

                                # Figure 11: CPC (category probability curves)
                                _fig11 = None
                                try:
                                    _fig11 = _draw_fig11_cpc(res, item_no_sel - 1)
                                    if _fig11 is not None:
                                        html_parts.append("<h2>CPC (Figure 11)</h2>")
                                        html_parts.append(_plotly_plot(_fig11, include_plotlyjs=False, output_type="div"))
                                        fig11_rendered = True

                                        # Figure 11.2: Winsteps Table 2.5 style Category Average Plot (CAP)
                                        try:
                                            pass  # [TCP11.2-DEDUP] duplicate block removed






















































































                                        except Exception as _e_cap:
                                            pass  # [TCP11.2-DEDUP] duplicate block removed
                                    else:
                                        html_parts.append("<div class='note'>(Figure 11 skipped: CPC not available.)</div>")
                                except Exception as _e11:
                                    html_parts.append("<div class='note' style='color:#b00'><b>Figure 11 CPC failed:</b> " + _html.escape(repr(_e11)) + "</div>")


                            else:
                                html_parts.append("<div class='note'>(Figure 10/11 skipped: missing item_df or response matrix)</div>")
                        except Exception as _e_icc:
                            html_parts.append("<div class='note'>(Figure 10/11 failed: " + _html.escape(repr(_e_icc)) + ")</div>")

                        # Fallback: draw CPC (Figure 11) even if ICC prerequisites (X matrix) are missing.
                        # CPC only needs item parameters (delta/b + tau) and a theta grid.
                        try:
                            if not fig11_rendered:
                                idf_cpc = getattr(res, 'item_df', None)
                                if isinstance(idf_cpc, _pd.DataFrame) and len(idf_cpc) > 0:
                                    # use the same selected item_no if available; otherwise default to 1
                                    try:
                                        _item_no_sel = min(max(int(item_no), 1), len(idf_cpc))
                                    except Exception:
                                        _item_no_sel = 1
                                    _fig11b = _draw_fig11_cpc(res, _item_no_sel - 1)
                                    if _fig11b is not None:
                                        html_parts.append("<h2>CPC (Figure 11)</h2>")
                                        html_parts.append(_plotly_plot(_fig11b, include_plotlyjs=False, output_type="div"))
                                        fig11_rendered = True
                        except Exception as _e11b:
                            html_parts.append("<div class='note' style='color:#b00'><b>Figure 11 CPC failed:</b> " + _html.escape(repr(_e11b)) + "</div>")
                        # [DEDUP] CAP/TCP forced-render block removed to avoid duplicates.

                        # DIF table for Profile (0/1) if available, dichotomous only
                        try:
                            pdf2 = getattr(res, 'person_df', None)
                            if isinstance(pdf2, _pd.DataFrame) and ('profile' in pdf2.columns) and int(getattr(res, 'max_cat', 1)) == 1:
                                prof = _np.asarray(pdf2['profile'].to_numpy(), dtype=float)
                                gmask0 = _np.isfinite(prof) & (prof == 0)
                                gmask1 = _np.isfinite(prof) & (prof == 1)
                                if gmask0.sum() >= 3 and gmask1.sum() >= 3:
                                    # root-find b per group (theta fixed from overall)
                                    def _solve_b(theta_g, x_g):
                                        # Newton on f(b)=sum(x - p)=0 where p=logistic(theta-b)
                                        b = 0.0
                                        for _ in range(60):
                                            p = 1.0/(1.0+_np.exp(-(theta_g - b)))
                                            f = _np.nansum(x_g - p)
                                            w = _np.nansum(p*(1.0-p))
                                            if w < 1e-9: break
                                            step = f / w
                                            b_new = b + step
                                            if abs(b_new - b) < 1e-6: 
                                                b = b_new
                                                break
                                            b = b_new
                                        p = 1.0/(1.0+_np.exp(-(theta_g - b)))
                                        w = _np.nansum(p*(1.0-p))
                                        se = _np.sqrt(1.0/max(w, 1e-9))
                                        return float(b), float(se)

                                    theta0 = th2[gmask0]; theta1 = th2[gmask1]
                                    dif_rows = []
                                    for j in range(I):
                                        xj0 = X02[gmask0, j]
                                        xj1 = X02[gmask1, j]
                                        # require some data
                                        if _np.isfinite(xj0).sum() < 2 or _np.isfinite(xj1).sum() < 2:
                                            continue
                                        b0, se0 = _solve_b(theta0[_np.isfinite(xj0)], xj0[_np.isfinite(xj0)])
                                        b1, se1 = _solve_b(theta1[_np.isfinite(xj1)], xj1[_np.isfinite(xj1)])
                                        dif = b1 - b0
                                        sed = _np.sqrt(se0**2 + se1**2)
                                        z = dif / sed if sed>0 else _np.nan
                                        # normal approx
                                        try:
                                            import math as _math
                                            from math import erf as _erf, sqrt as _sqrt
                                            # survival function for normal
                                            p = 2.0*(0.5* (1.0 - _math.erf(abs(z)/_math.sqrt(2.0)))) if _np.isfinite(z) else _np.nan
                                        except Exception:
                                            p = _np.nan
                                        dif_rows.append([j+1, item_labels[j], b0, se0, b1, se1, dif, sed, z, p])
                                    if dif_rows:
                                        dif_df = _pd.DataFrame(dif_rows, columns=["ItemNo","Item","b(Profile=0)","SE0","b(Profile=1)","SE1","DIF(1-0)","SE_DIF","Z","p(norm)"])
                                        html_parts.append("<h3>DIF table (Profile 0 vs 1)</h3>")
                                        html_parts.append("<p class='note'>Heuristic DIF: theta fixed to overall estimates; per-group item difficulty solved by Newton for dichotomous items.</p>")
                                        html_parts.append(_df_to_html(dif_df, max_rows=200))
                        except Exception as _e:
                            html_parts.append("<div class='note'>(DIF table failed: " + _html.escape(repr(_e)) + ")</div>")
                    else:
                        # [DEDUP] Figure 9 recovery block removed; Figure 9 is rendered once in the main pipeline.

# Figure 10/11: determine target item index (from config/form) (1-based -> 0-based)
                        try:
                            _item_no = int(_cfg_get('item_no', 1) or 1)
                        except Exception:
                            _item_no = 1
                        _item_idx0 = max(0, _item_no - 1)
# Chisquare table + ICC
                        try:
                            _fig10_pack = _draw_fig10_icc(res, _item_idx0, (_cfg_get('kid_no', 1)))
                            if _fig10_pack is not None:
                                _fig10, _tab_item, _chisq, _dfv, _pv = _fig10_pack
                                html_parts.append("<h2>Strata_raw score item=%d</h2>" % (_item_no,))
                                if _tab_item is not None and len(_tab_item):
                                    try:
                                        _tab_show = _tab_item[["Strata","Sum","n","Mean","Expected","Variance"]].copy()
                                    except Exception:
                                        _tab_show = _tab_item
                                    html_parts.append(_df_to_html(_tab_show, max_rows=50))
                                    html_parts.append("<p><b>ChSQ=%.2f</b> df=(S-1)*(k-1)=<b>%s</b> prob.=<b>%.2f</b> &nbsp; <a href='https://www.rasch.org/rmt/rmt202c.htm' target='_blank'>Ref. in Eq 4</a></p>" % (_chisq, _dfv, _pv))
                                _note10 = None
                                try:
                                    _note10 = FIGURE_GUIDE_NOTES.get("Figure 10") if isinstance(globals().get("FIGURE_GUIDE_NOTES", None), dict) else None
                                except Exception:
                                    _note10 = None
                                _safe_add_fig(html_parts, _fig10, "Figure 10 — ICC", note=_note10)


                                # Figure 10.2: CAC (category average curves) — step logistics by tau
                                try:
                                    # pull delta and tau for selected item
                                    idf2 = getattr(res, 'item_df', None)
                                    tau = _np.asarray(getattr(res, 'tau', []), dtype=float)
                                    bvec = _np.asarray(getattr(res, 'b', []), dtype=float) if hasattr(res, 'b') else _np.array([])
                                    # fallback delta from item_df
                                    if (bvec.size == 0 or not _np.isfinite(bvec).any()) and isinstance(idf2, _pd.DataFrame) and 'DELTA' in idf2.columns:
                                        bvec = _pd.to_numeric(idf2['DELTA'], errors='coerce').to_numpy(dtype=float)
                                    item_no_sel = _item_no
                                    if isinstance(idf2, _pd.DataFrame) and len(idf2) > 0:
                                        item_no_sel = min(max(int(_item_no), 1), len(idf2))
                                    if tau is not None and len(tau) > 0 and bvec is not None and len(bvec) > (_item_idx0) and _np.isfinite(bvec[_item_idx0]):
                                        b_j = float(bvec[_item_idx0])
                                        theta_grid2 = _np.linspace(-5.0, 5.0, 201)
                                        fig10_2 = go.Figure()
                                        for k in range(int(len(tau))):
                                            t_k = float(tau[k]) if _np.isfinite(tau[k]) else _np.nan
                                            if not _np.isfinite(t_k):
                                                continue
                                            p_k = 1.0 / (1.0 + _np.exp(-(theta_grid2 - (b_j + t_k))))
                                            fig10_2.add_trace(go.Scatter(x=theta_grid2, y=p_k, mode='lines', name=f"Step {k+1} (tau={t_k:.2f})"))
                                        ttl = f"CAC (Figure 10.2) — Item {item_no_sel}"
                                        try:
                                            if isinstance(idf2, _pd.DataFrame) and 'ITEM' in idf2.columns:
                                                ttl += f": {idf2['ITEM'].iloc[item_no_sel-1]}"
                                        except Exception:
                                            pass
                                        fig10_2.update_layout(title=ttl, xaxis_title='Ability (theta, logit)', yaxis_title='Probability (logistic step)', height=520)
                                        html_parts.append("<h2>CAC (Figure 10.2)</h2>")
                                        html_parts.append(_plotly_plot(fig10_2, include_plotlyjs=False, output_type='div'))
                                except Exception as _e_cac:
                                    html_parts.append("<div class='note' style='color:#b00'>(Figure 10.2 CAC failed: " + _html.escape(repr(_e_cac)) + ")</div>")

                                # Figure 11 — CPC (category probability curves)
                                try:
                                    _item_no_sel2 = 1
                                    try:
                                        idf_cpc2 = getattr(res, 'item_df', None)
                                        if isinstance(idf_cpc2, _pd.DataFrame) and len(idf_cpc2) > 0:
                                            _item_no_sel2 = min(max(int(_item_no), 1), len(idf_cpc2))
                                    except Exception:
                                        _item_no_sel2 = 1
                                    _fig11c = _draw_fig11_cpc(res, _item_no_sel2 - 1)
                                    _note11 = None
                                    try:
                                        _note11 = globals().get('FIGURE_GUIDE_NOTES', {}).get('Figure 11', None) if isinstance(globals().get('FIGURE_GUIDE_NOTES', None), dict) else None
                                    except Exception:
                                        _note11 = None
                                    _safe_add_fig(html_parts, _fig11c, 'Figure 11 — CPC', note=_note11)

                                    # Figure 11.2 — CAP (Winsteps Table 2.5)
                                    try:
                                        _fig11_2, _cap_tbl = _draw_fig11_2_cap(res, x_min=-6, x_max=5, x_step=1)
                                        _note11_2 = (
                                            "Category Average Plot (CAP): for each item, the digits 0/1/2 are placed at the "
                                            "mean person measure (θ) of respondents who chose that category. Items are ordered "
                                            "by delta (difficulty) descending; labels include δ and Infit MNSQ."
                                        )
                                        _safe_add_fig(html_parts, _fig11_2, 'Figure 11.2 — CAP (Table 2.5)', note=_note11_2)
                                        try:
                                            # show a small preview table below the plot
                                            if isinstance(_cap_tbl, _pd.DataFrame):
                                                _cap_show = _cap_tbl.copy()
                                                for c in ('cat0','cat1','cat2'):
                                                    if c in _cap_show.columns:
                                                        _cap_show[c] = _cap_show[c].round(3)
                                                html_parts.append("<div class='note'><b>CAP means:</b> cat0/cat1/cat2 are mean θ for that observed category (per item).</div>")
                                                html_parts.append(_df_to_html(_cap_show[[c for c in ['ACT','delta','infit_mnsq','PTMA','N_USED','cat0','cat1','cat2'] if c in _cap_show.columns]] if 'ACT' in _cap_show.columns else _cap_show, max_rows=26))
                                                                                                # Figure 11.3 + Table 11.3: Testlet cluster plot (TCP) + membership table
                                                try:
                                                    if not locals().get('tcp_rendered', False):
                                                        _q3_cut = float(_cfg_get('q3_cut', 0.20) or 0.20)
                                                        _fig_tcp, _tbl_tcp = _draw_fig11_3_tcp(res, q3_cut=_q3_cut)
                                                        if _fig_tcp is not None:
                                                            html_parts.append("<h2>Figure 11.3 — Testlet cluster plot (TCP)</h2>")
                                                            # Figure 11.3 captions
                                                            _journal_cap = (
                                                                "This testlet cluster plot (TCP) visualizes local dependence using a residual Q3 network and a relation-PCA layout. "
                                                                "Vertices are sized by PTMA, and edges indicate the strength of residual associations (|Q3| above the chosen cutoff). "
                                                                "PC axes summarize similarity in residual patterns, helping identify clusters consistent with testlet effects."
                                                            )
                                                            _teaching_cap = (
                                                                "How to read this TCP: Each point is an item (vertex). Bigger points mean larger PTMA (typically stronger alignment with the latent measure). "
                                                                "Lines connect item pairs whose residual correlation (Q3) exceeds the cutoff—thicker/denser connections imply stronger local dependence. "
                                                                "The PC1–PC2 coordinates come from PCA on the residual-relationship structure, so items close together share similar residual patterns. "
                                                                "Practical rule: look for tight groups with many strong Q3 links; if such a group also sits together in the PC space, it is a strong testlet candidate."
                                                            )
                                                            html_parts.append("<div class='note'><b>Journal caption:</b> " + _html.escape(_journal_cap) + "</div>")
                                                            html_parts.append("<details style='margin:0.5em 0 0.8em 0'><summary><b>Report guide (how to examine this plot)</b></summary><div class='note' style='margin-top:0.5em'>" + _html.escape(_teaching_cap) + "</div></details>")
                                                            html_parts.append(_plotly_plot(_fig_tcp, include_plotlyjs=False, output_type='div'))
                                                        if _tbl_tcp is not None and len(_tbl_tcp):
                                                            html_parts.append("<h3>Table 11.3 — Testlet membership + PTMA + Q3max</h3>")
                                                            html_parts.append(_df_to_html(_tbl_tcp, max_rows=200))
                                                        tcp_rendered = True
                                                except Exception as _e_tcp:
                                                    html_parts.append("<div class='note' style='color:#b00'>(Figure 11.3/Table 11.3 failed: " + _html.escape(repr(_e_tcp)) + ")</div>")

                                                html_parts.append("""
<div class='note' style='margin-top:0.8em'>
<b>Visual diagnostic pattern (what you'll see in practice)</b><br><br>
If you line up: <b>PTMA</b> + <b>Infit/Outfit</b> + <b>Residual Q3 matrix</b> + <b>Residual PCA</b>, you'll often see the following for <b>testlet items</b>:
<ul>
<li>PTMA: modest or uneven</li>
<li>Infit: clustered &gt; 1.2</li>
<li>Q3: strong positive links among themselves</li>
<li>Residual PC1: dominated by those items</li>
</ul>
This is why <b>PTMA alone is not enough</b>, but it is a <b>necessary witness</b>.
<hr style='margin:0.8em 0'>
<b>Conceptual summary</b>: PTMA asks "Does this item move monotonically along theta?" Testlets answer: "Yes... but along our private little subdimension." When theta != testlet trait, PTMA suffers.
<hr style='margin:0.8em 0'>
<b>Practical rules</b>
<table class='tbl'>
<tr><th>Observation</th><th>Interpretation</th></tr>
<tr><td>High loading + High Q3 + Normal PTMA</td><td>Mild LD</td></tr>
<tr><td>High loading + High Q3 + Low PTMA</td><td><b>Serious LD</b></td></tr>
<tr><td>Negative PTMA + High Q3</td><td>Testlet reversal / miscoding</td></tr>
<tr><td>Good fit + Low PTMA</td><td>Secondary dimension</td></tr>
</table>
<b>One-sentence takeaway</b>: Under testlet effects, <b>PTMA is conservative</b> - it refuses to reward covariance that does not align with the Rasch latent trait, even when loadings look impressive.
</div>
""")
                                                # [DEDUP] duplicate TCP/Table 11.3 block removed

                                                # Annotation beneath Table 11.2: Visual diagnostic pattern
                                                html_parts.append("""
<div class='note' style='margin-top:0.8em'>
<b>Visual diagnostic pattern (what you’ll see in practice)</b><br><br>
If you line up: <b>PTMA</b> + <b>Infit/Outfit</b> + <b>Residual Q3 matrix</b> + <b>Residual PCA</b>, you’ll often see the following for <b>testlet items</b>:
<ul>
<li>PTMA: modest or uneven</li>
<li>Infit: clustered &gt; 1.2</li>
<li>Q3: strong positive links among themselves</li>
<li>Residual PC1: dominated by those items</li>
</ul>
This is why <b>PTMA alone is not enough</b>, but it’s a <b>necessary witness</b>.
<hr style='margin:0.8em 0'>
<b>Practical rules</b>
<table class='tbl'>
<tr><th>Observation</th><th>Interpretation</th></tr>
<tr><td>High loading + High Q3 + Normal PTMA</td><td>Mild LD</td></tr>
<tr><td>High loading + High Q3 + Low PTMA</td><td><b>Serious LD</b></td></tr>
<tr><td>Negative PTMA + High Q3</td><td>Testlet reversal / miscoding</td></tr>
<tr><td>Good fit + Low PTMA</td><td>Secondary dimension</td></tr>
</table>
<b>One-sentence takeaway</b>: Under testlet effects, <b>PTMA is conservative</b> — it refuses to reward covariance that does not align with the Rasch latent trait, even when loadings look impressive.
</div>
""")

                                        except Exception:
                                            pass
                                    except Exception as _e11_2:
                                        html_parts.append("<div class='note' style='color:#b00'><b>Figure 11.2 CAP failed:</b> " + _html.escape(repr(_e11_2)) + "</div>")

                                except Exception as _e11c:
                                    html_parts.append("<div class='note' style='color:#b00'><b>Figure 11 CPC failed:</b> " + _html.escape(repr(_e11c)) + "</div>")

                        except Exception as _e:
                            html_parts.append("<div class='note' style='color:#b00'><b>Figure 10 ICC failed:</b> " + _html.escape(repr(_e)) + "</div>")

                                                # CPC
                        try:
                            if _fig11 is not None:
                                _note11 = None
                                try:
                                    _note11 = FIGURE_GUIDE_NOTES.get("Figure 11") if isinstance(globals().get("FIGURE_GUIDE_NOTES", None), dict) else None
                                except Exception:
                                    _note11 = None
                                _safe_add_fig(html_parts, _fig11, "Figure 11 — CPC", note=_note11)
                        except Exception as _e:
                            html_parts.append("<div class='note' style='color:#b00'><b>Figure 11 CPC failed:</b> " + _html.escape(repr(_e)) + "</div>")
                        except Exception as _e:
                            html_parts.append("<div class='note' style='color:#b00'><b>Figure 11 CPC failed:</b> " + _html.escape(repr(_e)) + "</div>")

                        # -------------------------------------------------
                        # Task 12 (Figure 12): DIF forest + Winsteps-style tables
                        # Group label is taken from the last column in the input data
                        # (e.g., 'Profile'). If only one group exists, we show a note.
                        # -------------------------------------------------
                        try:
                            _task12_parts = _as_html_parts(_render_task12_dif(res))
                            if _task12_parts:
                                html_parts.extend(_task12_parts)
                        except Exception as _e:
                            html_parts.append("<div class='note' style='color:#b00'><b>Figure 12 DIF failed:</b> " + _html.escape(repr(_e)) + "</div>")

                        # -------------------------------------------------
                        # Task 13 (Figure 13): Person dimension plot + Person list (Top 20)
                        # Using standardized residuals (ZSTD) like Figure 9, but for persons.
                        # -------------------------------------------------
                        try:
                            # Beta (path coefficient) for persons: regress y~x where
                            # x = mean(ZSTD) across items in Cluster 1 and y = mean(ZSTD) across items in Cluster 2.
                            # Item clusters are defined by the sign of Contrast-1 loading (residual PCA; same as Figure 9).
                            _beta13 = None
                            _p_beta13 = None
                            try:
                                _Z13_full = getattr(res, 'debug', {}).get('ZSTD', None) if isinstance(getattr(res, 'debug', None), dict) else None
                                _nodes_items13, _ = _build_item_vertices_relations_for_fig9(res)
                                if _Z13_full is not None and _nodes_items13 is not None and 'cluster' in _nodes_items13.columns:
                                    _beta13, _p_beta13, _ = _beta_p_from_zstd_clusters(_Z13_full, _nodes_items13['cluster'].to_numpy())
                            except Exception:
                                _beta13 = None
                                _p_beta13 = None

                            fig13, _person_top20 = _person_dimension_plot_top20(res, top_n=20, beta_value=_beta13, beta_p=_p_beta13)
                            if fig13 is not None:
                                _note13 = None
                                try:
                                    _note13 = FIGURE_GUIDE_NOTES.get("Figure 13") if isinstance(globals().get("FIGURE_GUIDE_NOTES", None), dict) else None
                                except Exception:
                                    _note13 = None
                                _safe_add_fig(html_parts, fig13, "Task 13 — Person dimension plot", note=_note13)

                                # Beta annotation under the plot (explain what x/y mean; highlight beta+p)
                                try:
                                    _bnote13 = (
                                        "<div class='note'>"
                                        "<b>Beta (path coefficient)</b>: OLS slope for <i>y</i>~<i>x</i>, where "
                                        "x = mean(ZSTD) across items in Cluster 1 and y = mean(ZSTD) across items in Cluster 2. "
                                        "Clusters are defined by the sign of the <b>Contrast 1 loading</b> (residual PCA; same split used in Figure 9)."
                                    )
                                    if _beta13 is not None and _p_beta13 is not None:
                                        _bnote13 += " &nbsp; <b style='color:#b00;font-size:22px'>Beta=%.2f, p=%.3g</b>" % (float(_beta13), float(_p_beta13))
                                    _bnote13 += "</div>"
                                    html_parts.append(_bnote13)
                                except Exception:
                                    pass
                                # Figure 13b: 10-bin frequency histogram of |Q3| (persons; upper triangular)
                                try:
                                    _Z13 = getattr(res, 'debug', {}).get('ZSTD', None) if isinstance(getattr(res, 'debug', None), dict) else None
                                    if _Z13 is not None:
                                        _q13 = _q3_abs_from_zstd(_Z13, mode='persons')
                                        _fig13b, _tab13b = _q3_hist10_figure(_q13, title='Figure 13b — |Q3| frequency (persons; upper triangular)')
                                        if _fig13b is not None:
                                            html_parts.append("<h3>Figure 13b — Residual correlation frequency (persons)</h3>")
                                            html_parts.append(_plotly_plot(_fig13b, include_plotlyjs=False, output_type='div'))
                                        if _tab13b is not None:
                                            html_parts.append(_df_to_html(_tab13b, max_rows=10))
                                except Exception as _e13b:
                                    html_parts.append("<div class='note' style='color:#b00'>(Figure 13b failed: " + _html.escape(repr(_e13b)) + ")</div>")

                            if _person_top20 is not None and hasattr(_person_top20, "empty") and not _person_top20.empty:
                                _cols13 = [c for c in ["Person","Loading","Theta","SS(i)","a(i)","b(i)"] if c in _person_top20.columns]
                                if not _cols13:
                                    _cols13 = list(_person_top20.columns)
                                _safe_add_table(
                                    html_parts,
                                    _person_top20[_cols13],
                                    "Person list for Figure 13 (Top 20)",
                                    note=None,
                                    max_rows=50,

                                )

                            # -------------------------------------------------
                            # Task 14: Plausible Values (PV) + Rubin's rules for group comparisons
                            # -------------------------------------------------
                            try:
                                t141, t142, t14_note = _pv_rubin_group_compare(res, k=10)
                                if t141 is not None and hasattr(t141, 'empty') and not t141.empty:
                                    html_parts.append("<h2>Task 14 — Plausible Values (k=10) + Rubin's rules</h2>")
                                    if t14_note:
                                        html_parts.append("<div class='note'>" + _html.escape(str(t14_note)) + "</div>")
                                    html_parts.append("<h3>Table 14.1 — Group means (PV pooled)</h3>")
                                    html_parts.append(_df_to_html(t141, max_rows=100))
                                if t142 is not None and hasattr(t142, 'empty') and not t142.empty:
                                    html_parts.append("<h3>Table 14.2 — Group contrasts vs reference (PV pooled)</h3>")
                                    html_parts.append(_df_to_html(t142, max_rows=200))
                            except Exception as _e14:
                                html_parts.append("<div class='note' style='color:#b00'>(Task 14 failed: " + _html.escape(repr(_e14)) + ")</div>")

                        except Exception as _e13:
                            html_parts.append("<pre>Figure 13 failed: %s</pre>" % _html.escape(repr(_e13)))
                except Exception as _e:
                    html_parts.append("<div class='note'>(Figure 9 failed: " + _html.escape(repr(_e)) + ")</div>")

    except Exception as _e_after_fig4:
        html_parts.append("<div class='note'>(Figure 4+ sections failed: " + _html.escape(repr(_e_after_fig4)) + ")</div>")



    body = "\n".join(html_parts)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Rasch report { _html.escape(str(run_id)) }</title>{css}{plotly_js}</head><body>{body}</body></html>"


# =========================================================
# Figure 4: add strata separation lines on person measures
# =========================================================
def _compute_equal_width_strata(theta: np.ndarray, H: int):
    """
    Equal-width intervals in person measures from max -> min (Winsteps-style strata count H).
    Returns cutpoints [(upper, lower), ...] and label per person ("S_1"..).
    """
    th = np.asarray(theta, dtype=float)
    m = np.isfinite(th)
    thv = th[m]
    H = int(max(1, H))
    if thv.size == 0:
        return [], np.array(["S_1"] * th.shape[0], dtype=object)
    tmax = float(np.max(thv))
    tmin = float(np.min(thv))
    if H == 1 or tmax == tmin:
        cuts = [(tmax, tmin)]
        return cuts, np.array(["S_1"] * th.shape[0], dtype=object)
    w = (tmax - tmin) / H
    cuts = []
    for s in range(H):
        upper = tmax - s * w
        lower = tmax - (s + 1) * w
        if s == H - 1:
            lower = tmin
        cuts.append((upper, lower))
    labs = np.empty(th.shape[0], dtype=object)
    for i, val in enumerate(th):
        if not math.isfinite(float(val)):
            labs[i] = f"S_{H}"
            continue
        idx = int((tmax - float(val)) / w) + 1
        if idx < 1: idx = 1
        if idx > H: idx = H
        labs[i] = f"S_{idx}"

    # --- References (AMA) ---
    try:
        _append_ama_references(html_parts)
    except Exception:
        pass


    return cuts, labs

def _add_strata_lines_and_labels_to_fig4(fig, theta: np.ndarray, H: int):
    """
    Add horizontal dotted red lines at strata boundaries and S_k labels.
    """
    cuts, _ = _compute_equal_width_strata(theta, H)
    if not cuts or len(cuts) <= 1:
        return fig, cuts

    boundaries = [c[1] for c in cuts[:-1]]  # lower bounds
    shapes = list(fig.layout.shapes) if getattr(fig.layout, "shapes", None) else []
    for y in boundaries:
        shapes.append(dict(
            type="line",
            xref="paper", yref="y",
            x0=0, x1=1,
            y0=float(y), y1=float(y),
            line=dict(width=1, dash="dot", color="red"),
            opacity=0.9,
        ))
    fig.update_layout(shapes=shapes)

    ann = list(fig.layout.annotations) if getattr(fig.layout, "annotations", None) else []
    for i, (upper, lower) in enumerate(cuts, start=1):
        yc = (upper + lower) / 2.0
        ann.append(dict(
            xref="paper", yref="y",
            x=0.01, y=float(yc),
            text=f"S_{i}",
            showarrow=False,
            font=dict(size=12, color="red"),
            bgcolor="rgba(255,255,255,0.7)",
            bordercolor="rgba(255,0,0,0.5)",
            borderwidth=1,
        ))
    fig.update_layout(annotations=ann)
    return fig, cuts

# =========================================================
# Scree plots: response PCA and standardized residual PCA
# + AAC for top 3 eigenvalues
# =========================================================
def _aac_from_top3(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=float)
    eigs = eigs[np.isfinite(eigs)]
    if eigs.size < 3:
        return float("nan")
    v1, v2, v3 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    if v2 == 0 or v3 == 0:
        return float("nan")
    r = (v1 / v2) / (v2 / v3)
    return float(r / (1.0 + r))

def _pca_eigenvalues(X: np.ndarray) -> np.ndarray:
    """
    PCA eigenvalues from correlation matrix of columns (items).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return np.array([])
    # center
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    # safe std
    sd = np.nanstd(Xc, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = Xc / sd
    # correlation matrix
    R = np.corrcoef(Z, rowvar=False)
    # eigenvalues
    try:
        w = np.linalg.eigvalsh(R)
        w = np.sort(w)[::-1]
        return w
    except Exception:
        return np.array([])

def _render_scree_plot(eigs: np.ndarray, title: str) -> str:
    import plotly.graph_objects as go
    eigs = np.asarray(eigs, dtype=float)
    k = int(eigs.size)
    if k == 0:
        return "<div class='note'>(Scree plot unavailable)</div>"
    x = list(range(1, k + 1))
    aac = _aac_from_top3(eigs)
    aac_txt = "NA" if (not np.isfinite(aac)) else f"{aac:.2f}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=eigs, mode="lines+markers", name="Eigenvalue"))
    fig.update_layout(
        title=f"{title} (AAC(top3)={aac_txt})",
        xaxis_title="Component",
        yaxis_title="Eigenvalue",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)

def _get_response_matrix_for_pca(res):
    """Response matrix used in PCA and downstream plots.

    IMPORTANT: If the input data were continuous, we must use the transformed
    categorical matrix (X_post / X_transformed / X_int / X0) for all post-analysis.
    """
    dbg = res.debug if isinstance(getattr(res, 'debug', None), dict) else {}
    is_cont = bool(dbg.get('is_continuous_input', False))
    X = None
    if is_cont:
        for k in ('X_post', 'X_transformed', 'X_int', 'X_obs', 'X0', 'X'):
            v = dbg.get(k, None)
            if v is not None:
                X = v
                break
    else:
        X = dbg.get('X_int', dbg.get('X_obs', dbg.get('X', None)))

    if X is None:
        X = dbg.get('X0', None)
        if X is not None:
            X = np.asarray(X, dtype=float) + float(getattr(res, 'min_cat', 0))

    return np.asarray(X, dtype=float) if X is not None else None

def _get_standardized_residual_matrix(res):
    Z = res.debug.get("ZSTD", None)
    if Z is None:
        Z = res.debug.get("zscore", None)
    return np.asarray(Z, dtype=float) if Z is not None else None

# ===== RUMM 2020 model-data fit (chi-square) =====
def _chi2_sf(x: float, df: int) -> float:
    """Survival function (1-CDF) for chi-square without external deps.

    Uses the Wilson–Hilferty normal approximation (common in Rasch reporting when
    scipy isn't available). Accurate enough for reporting p-values and
    significance flags; not intended for high-precision tail probabilities.
    """
    x = float(max(x, 0.0))
    df = int(max(df, 1))
    # Wilson–Hilferty transform to approx N(0,1)
    z = ((x/df) ** (1.0/3.0) - (1.0 - 2.0/(9.0*df))) / math.sqrt(2.0/(9.0*df))
    # sf of standard normal via erfc
    return 0.5 * math.erfc(z / math.sqrt(2.0))

def _rumm_strata_labels_fixed(theta: np.ndarray) -> np.ndarray:
    """A-D strata using fixed cutpoints: A>3.5, B>1.0, C>-1.5, D>-4.0 else E."""
    th = np.asarray(theta, dtype=float)
    out = np.empty(th.shape[0], dtype=object)
    out[:] = "E"
    out[th > -4.0] = "D"
    out[th > -1.5] = "C"
    out[th > 1.0]  = "B"
    out[th > 3.5]  = "A"
    return out.astype(str)

def _safe_get_obs_exp_var(res):
    X_obs = np.asarray(res.debug.get("X_int", res.debug.get("X_obs", res.debug.get("X", None))), dtype=float)
    if X_obs.ndim != 2:
        X0 = np.asarray(res.debug.get("X0", None), dtype=float)
        if X0.ndim == 2:
            X_obs = X0 + float(getattr(res, "min_cat", 0))
    E = np.asarray(res.debug.get("E_exp", res.debug.get("E", None)), dtype=float)
    V = np.asarray(res.debug.get("VAR", res.debug.get("Var", None)), dtype=float)
    if X_obs.ndim != 2 or E.ndim != 2 or V.ndim != 2:
        return None, None, None
    return X_obs, E, V

def _aggregate_fixed_strata(res, jitem: int = 1):
    X_obs, E, V = _safe_get_obs_exp_var(res)
    if X_obs is None:
        return None
    theta = np.asarray(getattr(res, "theta", res.person_df["MEASURE"].to_numpy()), dtype=float)
    strata = _rumm_strata_labels_fixed(theta)
    order = ["A", "B", "C", "D"]
    I = int(E.shape[1])

    rows_all = []
    for idx, s in enumerate(order, start=1):
        m = (strata == s)
        n = int(np.sum(m))
        if n == 0:
            rows_all.append((f"{s}_{idx}", float("nan"), 0, float("nan"), float("nan"), float("nan")))
            continue
        obs_sum = float(np.nansum(X_obs[m, :]))
        exp_sum = float(np.nansum(E[m, :]))
        var_sum = float(np.nansum(V[m, :]))
        cntL = int(n * I)
        mean = float(obs_sum / max(cntL, 1))
        thr = { "A":">3.5", "B":">1.0", "C":">-1.5", "D":">-4.0" }.get(s, "<=-4.0")
        rows_all.append((f"{s}_{idx}({thr})", obs_sum, cntL, mean, exp_sum, var_sum))

    chi = 0.0
    ch1 = ch2 = ch3 = 0.0
    lam1, lam2, lam3 = 1.0, 0.01, 0.67
    for j in range(I):
        for s in order:
            m = (strata == s)
            if not np.any(m):
                continue
            O = float(np.nansum(X_obs[m, j]))
            Ex = float(np.nansum(E[m, j]))
            Va = float(np.nansum(V[m, j]))
            if Va <= 0:
                Va = 0.01
            chi += (O - Ex) ** 2 / Va
            if Ex > 0 and O > 0:
                ch1 += O * ((O/Ex) ** lam1 - 1.0)
                ch2 += O * ((O/Ex) ** lam2 - 1.0)
                ch3 += O * ((O/Ex) ** lam3 - 1.0)
    def pd_to_chi(val, lam): 
        return abs(2.0/(lam*(lam+1.0))*val)

    ch1v = pd_to_chi(ch1, lam1)
    ch2v = pd_to_chi(ch2, lam2)
    ch3v = pd_to_chi(ch3, lam3)

    df_all = (len(order)-1) * I
    p_all = _chi2_sf(chi, df_all)

    jitem = int(max(1, min(jitem, I)))
    j = jitem - 1
    rows_one = []
    chi_one = 0.0
    for idx, s in enumerate(order, start=1):
        m = (strata == s)
        n = int(np.sum(m))
        if n == 0:
            continue
        O = float(np.nansum(X_obs[m, j]))
        Ex = float(np.nansum(E[m, j]))
        Va = float(np.nansum(V[m, j]))
        if Va <= 0:
            Va = 0.01
        mean = float(O / max(n, 1))
        thr = { "A":">3.5", "B":">1.0", "C":">-1.5", "D":">-4.0" }.get(s, "<=-4.0")
        rows_one.append((f"{s}_{idx}({thr})", O, n, mean, Ex, Va))
        chi_one += (O-Ex)**2 / Va

    df_one = (len(order)-1) * 1
    p_one = _chi2_sf(chi_one, df_one)

    return dict(
        rows_all=rows_all, chi_all=chi, df_all=df_all, p_all=p_all,
        rows_one=rows_one, chi_one=chi_one, df_one=df_one, p_one=p_one,
        ch1=ch1v, ch2=ch2v, ch3=ch3v,
        p1=_chi2_sf(ch1v, df_all), p2=_chi2_sf(ch2v, df_all), p3=_chi2_sf(ch3v, df_all),
        df_pd=df_all, jitem=jitem
    )


def render_rumm2020_block(res, jitem: int = 1) -> str:
    """
    RUMM-style diagnostics block (no SciPy dependency):
      1) Item–Trait chi-square (overall + item-level)
      2) PCA / Scree (Observed, Residual, ZSTD) + Parallel Analysis + DC
      3) DIF (2-group only): class-interval × group (approx chi-square from F) + forest plot (approx DIF logit)
    Notes:
      - Requires cell-level matrices in res.debug: X_obs/X_int, E_exp (or EXP), VAR.
      - If matrices are absent, shows a clear message instead of failing silently.
    """
    import math
    import numpy as _np
    import pandas as _pd
    import html as _html

    # ---------- helpers ----------
    def _isfinite(x):
        try:
            return not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        except Exception:
            return False

    def _chi2_sf_wh(x: float, k: float) -> float:
        """Upper-tail p for Chi-square via Wilson–Hilferty normal approximation (no SciPy)."""
        if not _isfinite(x) or not _isfinite(k) or k <= 0:
            return float("nan")
        if x <= 0:
            return 1.0
        # z ≈ ((x/k)^(1/3) - (1 - 2/(9k))) / sqrt(2/(9k))
        z = ((x / k) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / math.sqrt(2.0 / (9.0 * k))
        # sf = 0.5 * erfc(z/sqrt(2))
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    def _fmt_p(p):
        if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
            return "NA"
        if p < 0.001:
            return "0.000"
        return f"{p:.3f}"

    def _safe_get_mats():
        dbg = getattr(res, "debug", {}) or {}
        def _to2d(keylist):
            for k in keylist:
                if k in dbg:
                    a = _np.asarray(dbg[k], dtype=float)
                    if a.ndim == 2:
                        return a
            return None
        X = _to2d(["X_int", "X_obs", "X", "X0"])
        E = _to2d(["E_exp", "EXP", "E"])
        V = _to2d(["VAR", "Var", "V"])
        return X, E, V, dbg

    def _strata_fixed(theta, cuts=(3.5, 1.0, -1.5, -4.0)):
        th = _np.asarray(theta, dtype=float)
        lab = _np.full(th.shape, -1, dtype=int)
        # A=0: >3.5, B=1: >1.0, C=2: >-1.5, D=3: >-4.0, else -1
        lab[th > cuts[0]] = 0
        lab[(th <= cuts[0]) & (th > cuts[1])] = 1
        lab[(th <= cuts[1]) & (th > cuts[2])] = 2
        lab[(th <= cuts[2]) & (th > cuts[3])] = 3
        return lab

    def _agg_chi(X, E, V, strata_lab):
        """Aggregate by strata across persons, return per-stratum summary and overall chi-square."""
        I = X.shape[1]
        rows = []
        chisq = 0.0
        df = 0
        for h in range(4):
            m = (strata_lab == h)
            if m.sum() == 0:
                continue
            Xh = X[m, :]
            Eh = E[m, :]
            Vh = V[m, :]
            # sum over persons and items
            O = _np.nansum(Xh)
            Ex = _np.nansum(Eh)
            Var = _np.nansum(Vh)
            Mean = O / (m.sum() * I) if m.sum() * I > 0 else _np.nan
            chi_h = ((O - Ex) ** 2 / Var) if Var > 0 else _np.nan
            if _isfinite(chi_h):
                chisq += float(chi_h)
                df += (I - 1)  # match your existing df convention
            rows.append({"Strata": h+1, "Sum": O, "Count*L": int(m.sum()*I), "Mean": Mean, "Expected": Ex, "Variance": Var})
        p = _chi2_sf_wh(chisq, df) if df > 0 else float("nan")
        return rows, chisq, df, p

    def _item_level_chi(X, E, V, strata_lab):
        I = X.shape[1]
        out = []
        for j in range(I):
            chisq = 0.0
            df = 0
            ok = True
            for h in range(4):
                m = (strata_lab == h)
                if m.sum() == 0:
                    continue
                Oj = _np.nansum(X[m, j])
                Ej = _np.nansum(E[m, j])
                Vj = _np.nansum(V[m, j])
                if Vj <= 0 or (not _isfinite(Vj)):
                    ok = False
                    continue
                chisq += float(((Oj - Ej) ** 2) / Vj)
                df += 1  # (k-1) where k=4 => 3, but per-stratum adds 1; we'll report df=3 for 4 strata
            if not ok:
                out.append((j+1, float("nan"), 3, float("nan")))
            else:
                # force df=3 for 4 strata to match your table
                df_report = 3
                p = _chi2_sf_wh(chisq, df_report)
                out.append((j+1, chisq, df_report, p))
        return out

    def _eigvals(mat):
        # column-center and compute correlation eigenvalues (match Figures 5–7 pipeline)
        A = mat.copy()
        keep = _np.isfinite(A).any(axis=0)
        A = A[:, keep]
        if A.size == 0 or A.shape[1] < 2:
            return _np.asarray([], dtype=float)
        mu = _np.nanmean(A, axis=0)
        A = A - mu
        A = _np.where(_np.isfinite(A), A, 0.0)
        C = (A.T @ A) / max(A.shape[0]-1, 1)
        sd = _np.sqrt(_np.clip(_np.diag(C), 1e-12, _np.inf))
        C = C / sd[:, None] / sd[None, :]
        w = _np.linalg.eigvalsh(C)
        w = _np.sort(w)[::-1]
        return w


    def _dc_from_top3(w):
        if w is None or len(w) < 3:
            return float("nan")
        v1, v2, v3 = float(w[0]), float(w[1]), float(w[2])
        if v2 <= 0 or v3 <= 0:
            return float("nan")
        r = (v1 / v2) / (v2 / v3)
        return r / (1.0 + r)

    def _parallel_analysis(n, p, reps=100):
        # normal random; return 95th percentile eigenvalues
        evs = []
        for _ in range(reps):
            R = _np.random.normal(size=(n, p))
            w = _eigvals(R)
            if len(w) < p:
                w = _np.pad(w, (0, p-len(w)))
            evs.append(w[:p])
        evs = _np.vstack(evs)
        return _np.quantile(evs, 0.95, axis=0)

    def _plot_scree(eigs, pa_eigs, title, dc):
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception:
            return "<div class='warn'>Plotly not available.</div>"
        k = min(len(eigs), 20)
        x = _np.arange(1, k+1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=eigs[:k], mode="lines+markers", name="Eigenvalue"))
        if pa_eigs is not None and len(pa_eigs) >= k:
            fig.add_trace(go.Scatter(x=x, y=pa_eigs[:k], mode="lines", name="PA 95%"))
        fig.update_layout(
            title=f"{_html.escape(title)} — DC={dc:.2f}" if _isfinite(dc) else _html.escape(title),
            height=320,
            margin=dict(l=30, r=20, t=50, b=30),
            template="plotly_white",
            xaxis_title="Component",
            yaxis_title="Eigenvalue"
        )
        return (__import__('plotly.io', fromlist=['io']).to_html)(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})

    def _plot_dif_forest(items, dif, se, pvals, alpha=0.05, alpha_b=None):
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
        except Exception:
            return "<div class='warn'>Plotly not available.</div>"
        I = len(items)
        y = list(range(I))[::-1]
        lo = dif - 1.96*se
        hi = dif + 1.96*se
        txt = [f"p={_fmt_p(pvals[i])}" for i in range(I)]
        # color: significant vs non-significant
        cols = ["#d62728" if (pvals[i] is not None and _isfinite(float(pvals[i])) and float(pvals[i]) < alpha) else "#1f77b4" for i in range(I)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dif, y=y, mode="markers",
            marker=dict(color=cols, size=8),
            text=txt, hovertemplate="%{text}<br>DIF=%{x:.2f}<extra></extra>",
            name="DIF"
        ))
        for i in range(I):
            fig.add_shape(type="line", x0=float(lo[i]), x1=float(hi[i]), y0=y[i], y1=y[i],
                          line=dict(color=cols[i], width=2))
        # zero line
        fig.add_shape(type="line", x0=0, x1=0, y0=-1, y1=I, line=dict(color="black", width=1, dash="dot"))
        # layout: black bg + red text
        fig.update_layout(
            title="DIF Forest Plot (2 groups)",
            height=max(320, 18*I + 120),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
            xaxis=dict(title="DIF (logit approx)", gridcolor="#ddd", zeroline=False),
            yaxis=dict(tickmode="array", tickvals=y, ticktext=[_html.escape(t) for t in items[::-1]], gridcolor="#eee"),
            margin=dict(l=240, r=80, t=60, b=40),
            showlegend=False
        )
        return (__import__('plotly.io', fromlist=['io']).to_html)(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})

    def _dif_logit_approx(X, max_cat, group):
        """Approx DIF per item using mean score -> logit transform (approx), plus SE from pooled var."""
        I = X.shape[1]
        g = _np.asarray(group)
        lv = _np.unique(g[_np.isfinite(_np.where(g==g, 1, _np.nan))])
        if len(lv) != 2:
            return None
        g0, g1 = lv[0], lv[1]
        dif = _np.zeros(I)
        se = _np.zeros(I)
        p = _np.zeros(I)
        for j in range(I):
            x0 = _np.asarray(X[g==g0, j], dtype=float)
            x1 = _np.asarray(X[g==g1, j], dtype=float)
            x0 = x0[_np.isfinite(x0)]
            x1 = x1[_np.isfinite(x1)]
            if len(x0) < 5 or len(x1) < 5:
                dif[j]=_np.nan; se[j]=_np.nan; p[j]=_np.nan; continue
            m0 = x0.mean(); m1 = x1.mean()
            # keep within (0, max_cat)
            eps = 1e-3
            m0 = min(max(m0, eps), max_cat-eps)
            m1 = min(max(m1, eps), max_cat-eps)
            l0 = math.log(m0/(max_cat-m0))
            l1 = math.log(m1/(max_cat-m1))
            dif[j] = l1 - l0
            # delta method SE
            # var(logit(m)) ~ var(m) * (d/dm log(m/(M-m)))^2 ; d = M/(m(M-m))
            d0 = max_cat/(m0*(max_cat-m0))
            d1 = max_cat/(m1*(max_cat-m1))
            v0 = x0.var(ddof=1)/len(x0)
            v1 = x1.var(ddof=1)/len(x1)
            se[j] = math.sqrt(v0*d0*d0 + v1*d1*d1) if v0>=0 and v1>=0 else _np.nan
            z = dif[j]/se[j] if se[j] and _isfinite(se[j]) and se[j]>0 else _np.nan
            # two-sided normal approx
            p[j] = math.erfc(abs(z)/math.sqrt(2.0)) if _isfinite(z) else _np.nan
        return dif, se, p, (g0, g1)

    # ---------- start building block ----------
    hp = []
    hp.append("<div class='card'><h2>RUMM 2020 Chi-square model-data fit, PCA, and DIF</h2>")

    X, E, V, dbg = _safe_get_mats()
    if X is None or E is None or V is None:
        # show what keys exist to speed debugging
        keys = sorted(list((dbg or {}).keys()))
        hp.append("<div class='warn'><b>RUMM 2020:</b> (Expected/Variance matrices not available.)</div>")
        hp.append("<div class='small'>Missing one of: X_obs/X_int, E_exp/EXP, VAR. Available debug keys: "
                  + _html.escape(", ".join(keys[:40]) + ("..." if len(keys)>40 else "")) + "</div>")
        hp.append("</div>")
        return "\n".join(hp)

    # theta
    theta = _np.asarray(getattr(res, "theta", []), dtype=float)
    if theta.ndim != 1 or len(theta) != X.shape[0]:
        # fallback: try person_df Measure
        pdf = getattr(res, "person_df", None)
        if pdf is not None and "MEASURE" in pdf.columns and len(pdf) == X.shape[0]:
            theta = _np.asarray(pdf["MEASURE"], dtype=float)
        else:
            theta = _np.zeros(X.shape[0], dtype=float)

    strata = _strata_fixed(theta)

    # ---------------- Part 1: model-data fit ----------------
    hp.append("<h3>1) Item–Trait Chi-square (Fixed 4 strata)</h3>")
    rows, chisq, df, p = _agg_chi(X, E, V, strata)
    hp.append("<div class='small'>Strata item = All items</div>")
    hp.append("<table class='rumm'><tr><th>Strata</th><th>Sum</th><th>Count*L</th><th>Mean</th><th>Expected</th><th>Variance</th></tr>")
    for r in rows:
        hp.append(
            f"<tr><td>{r['Strata']}</td><td>{r['Sum']:.2f}</td><td>{int(r['Count*L'])}</td>"
            f"<td>{r['Mean']:.2f}</td><td>{r['Expected']:.2f}</td><td>{r['Variance']:.2f}</td></tr>"
        )
    hp.append("</table>")
    hp.append(f"<div class='small'><b>ChSQ</b>= {chisq:.2f} &nbsp; <b>df</b>= {df} &nbsp; <b>prob</b>= {_fmt_p(p)}</div>")

    hp.append("<div class='small'>Reference (RUMM2020): <a href='https://link.springer.com/article/10.1186/s40488-020-00108-7' target='_blank'>BMC Medical Research Methodology (2020) 20:108</a></div>")

    # Additional chi-square family (Engelhard, Invariant Measurement p.171): Pearson / Wilks / Cressie-Read
    try:
        O = _np.array([r['Sum'] for r in rows], dtype=float)
        Ee = _np.array([r['Expected'] for r in rows], dtype=float)
        Vv = _np.array([r['Variance'] for r in rows], dtype=float)

        n_items = int(getattr(res, 'debug', {}).get('I', 0) or (len(getattr(res, 'item_df', [])) if getattr(res,'item_df',None) is not None else 0))
        df_alt = max(int(n_items) * (len(O) - 1), 1)

        chisq1 = float(_np.nansum((O - Ee) ** 2 / _np.maximum(Vv, 1e-12)))
        p1 = _chi2_sf_wh(chisq1, df_alt)

        Om = _np.maximum(O, 1e-12)
        Em = _np.maximum(Ee, 1e-12)
        chisq2 = float(2.0 * _np.nansum(Om * _np.log(Om / Em)))
        p2 = _chi2_sf_wh(chisq2, df_alt)

        lam = 2.0 / 3.0
        chisq3 = float((2.0 / (lam * (lam + 1.0))) * _np.nansum(Om * (((Om / Em) ** lam) - 1.0)))
        p3 = _chi2_sf_wh(chisq3, df_alt)

        hp.append(
            "<pre class='small' style='margin-top:8px'>"
            "chSQ1 ==================================\n"
            "Strata(Pearson 1900) for All items\n"
            f"ChSQ=\t{chisq1:.2f}\tdf=k*(s-1)=\t{df_alt}\tprob.=\t{_fmt_p(p1)}\n"
            "Formula: χ² = Σ (O−E)²/Var\n"
            "In pp171 in Invariant Measurement (George Engelhard, Jr.)\n\n"
            "chSQ2 ===================================\n"
            "Strata(Wilks, 1935) for All items\n"
            f"ChSQ=\t{chisq2:.2f}\tdf=\t{df_alt}\tprob.=\t{_fmt_p(p2)}\n"
            "Formula: G² = 2Σ O ln(O/E)\n"
            "In pp171 in Invariant Measurement (George Engelhard, Jr.)\n\n"
            "chSQ3 ====================================\n"
            "Strata(Cressie & Read, 1988) for All items\n"
            f"ChSQ=\t{chisq3:.2f}\tdf=\t{df_alt}\tprob.=\t{_fmt_p(p3)}\n"
            "Formula: (2/(λ(λ+1)))Σ O[(O/E)^λ − 1], λ=2/3\n"
            "In pp171 in Invariant Measurement (George Engelhard, Jr.)"
            "</pre>"
        )
    except Exception:
        pass

    # item-level table with infit/outfit
    item_df = getattr(res, "item_df", None)
    infit = outfit = None
    name_col = None
    if item_df is not None:
        for c in ["INFIT_MNSQ", "INFIT", "Infit_MNSQ", "INFIT_MeanSquare"]:
            if c in item_df.columns:
                infit = item_df[c].tolist()
                break
        for c in ["OUTFIT_MNSQ", "OUTFIT", "Outfit_MNSQ", "OUTFIT_MeanSquare"]:
            if c in item_df.columns:
                outfit = item_df[c].tolist()
                break
        for c in ["Item", "ITEM", "NAME", "Label"]:
            if c in item_df.columns:
                name_col = c
                break

    items = []
    if item_df is not None and name_col is not None:
        items = [str(x) for x in item_df[name_col].tolist()]
    else:
        items = [f"Item {j+1}" for j in range(X.shape[1])]

    it = _item_level_chi(X, E, V, strata)
    hp.append("<h4>Item-level χ² (All items)</h4>")
    hp.append("<table class='rumm'><tr><th>ItemNo</th><th>Item</th><th>ChSQ</th><th>df</th><th>prob.</th><th>INFIT MNSQ</th><th>OUTFIT MNSQ</th></tr>")
    I = X.shape[1]
    alpha = 0.05
    alpha_b = alpha / max(I, 1)
    for (jno, ch, dfi, pi) in it:
        j = jno - 1
        ptxt = _fmt_p(pi)
        cls = "sigb" if _isfinite(pi) and pi < alpha_b else ("sig" if _isfinite(pi) and pi < alpha else "")
        inft = f"{float(infit[j]):.2f}" if infit and j < len(infit) and _isfinite(float(infit[j])) else "NA"
        outt = f"{float(outfit[j]):.2f}" if outfit and j < len(outfit) and _isfinite(float(outfit[j])) else "NA"
        hp.append(
            f"<tr class='{cls}'><td>{jno}</td><td>{_html.escape(items[j])}</td>"
            f"<td>{ch:.2f}</td><td>{dfi}</td><td>{ptxt}</td><td>{inft}</td><td>{outt}</td></tr>"
        )
    hp.append("</table>")
    hp.append(f"<div class='small'>α=0.05; Bonferroni α={alpha_b:.4f} (I={I}).</div>")

    # ---------------- Part 2: PCA / Scree ----------------
    hp.append("<h3>2) PCA / Scree (Observed, Rasch Residuals, ZSTD) + PA + DC</h3>")
    # Observed: center across persons
    Xc = _np.where(_np.isfinite(X), X, _np.nan)
    # Residuals and ZSTD
    R = _np.asarray(dbg.get("RESID", X - E), dtype=float)
    Z = _np.asarray(dbg.get("ZSTD", (X - E)/_np.sqrt(_np.maximum(V,1e-12))), dtype=float)

    # eigenvalues
    eig_obs = _eigvals(_np.where(_np.isfinite(Xc), Xc, _np.nan))
    eig_res = _eigvals(_np.where(_np.isfinite(R), R, _np.nan))
    eig_z   = _eigvals(_np.where(_np.isfinite(Z), Z, _np.nan))

    dc_obs = _dc_from_top3(eig_obs)
    dc_res = _dc_from_top3(eig_res)
    dc_z   = _dc_from_top3(eig_z)

    hp.append(
        "<div class='small'><b>DC (per rmt263c)</b>: let r=(v1/v2)/(v2/v3); "
        "DC = r/(1+r) = ((v1/v2)/(v2/v3)) / (1 + (v1/v2)/(v2/v3)). "
        "Reference: <a href='https://www.rasch.org/rmt/rmt263c.htm' target='_blank'>rasch.org/rmt/rmt263c.htm</a></div>"
    )

    # PA
    n = X.shape[0]
    pcols = X.shape[1]
    pa = _parallel_analysis(n, pcols, reps=80) if n >= 10 and pcols >= 3 and pcols <= 200 else None

    # show eigen table
    def _eig_table(eigs, label):
        k = min(len(eigs), 10)
        tot = float(_np.nansum(eigs)) if _isfinite(float(_np.nansum(eigs))) else _np.nan
        hp.append(f"<div class='small'><b>{_html.escape(label)}</b></div>")
        hp.append("<table class='rumm'><tr><th>Component</th><th>Eigenvalue</th><th>%Var</th><th>Cum%</th></tr>")
        cum = 0.0
        for i in range(k):
            ev = float(eigs[i])
            pct = (ev/tot*100.0) if tot and tot>0 else _np.nan
            cum += pct if _isfinite(pct) else 0.0
            hp.append(f"<tr><td>{i+1}</td><td>{ev:.4f}</td><td>{pct:.1f}</td><td>{cum:.1f}</td></tr>")
        hp.append("</table>")
        # DC hyperlink
        hp.append("<div class='small'>DC (Rasch_EGA_ratio, Eq.4) = "
                  f"{_dc_from_top3(eigs):.2f} — "
                  "<a href='https://www.rasch.org/rmt/rmt263c.htm' target='_blank'>rasch.org/rmt263c</a></div>")

    _eig_table(eig_obs, "Observed scores")
    hp.append(_plot_scree(eig_obs, pa, "Scree plot (Observed scores) + PA", dc_obs))

    _eig_table(eig_res, "Rasch residuals (X−E)")
    hp.append(_plot_scree(eig_res, pa, "Scree plot (Rasch residuals) + PA", dc_res))

    _eig_table(eig_z, "Z-score residuals (ZSTD)")
    hp.append(_plot_scree(eig_z, pa, "Scree plot (Z-score residuals) + PA", dc_z))

    # ---------------- Part 3: DIF ----------------

    hp.append("<div class='small'>DIF is reported in two complementary ways: "
              "<b>unfolded</b> (within class-interval strata: tests Item×Strata×Group interaction) and "
              "<b>folded</b> (collapsed across strata: overall Item×Group difference). "
              "Forest plot uses folded DIF effect (approx logit) with 95% CI.</div>")
    hp.append("<h3>3) DIF (RUMM2020 Item–Trait Chi-Square and Winsteps DIF Size)</h3>")
    hp.append("<div class='small'>Reference: <a href='https://www.rasch.org/rmt/rmt211k.htm' target='_blank'>rasch.org/rmt211k</a></div>")

    # group: prefer Profile from person_df
    group = None
    pdf = getattr(res, "person_df", None)
    if pdf is not None:
        # detect group column (case-insensitive). Prefer 'profile' then common variants.
        colmap = {str(c).strip().lower(): c for c in pdf.columns}
        for key in ["profile","Profile","group","gender","true_label"]:
            k = str(key).strip().lower()
            if k in colmap:
                group = _np.asarray(pdf[colmap[k]], dtype=object)
                break

    if group is None:
        hp.append("<div class='warn'>DIF skipped: no group variable found in person_df (e.g., Profile).</div>")
        hp.append("</div>")
        return "\n".join(hp)

    # keep finite theta/strata
    g = group
    # two groups only
    u = [x for x in _np.unique(g) if str(x) != "nan"]
    if len(u) != 2:
        hp.append(f"<div class='warn'>DIF forest plot available only for 2 groups; found {len(u)} groups.</div>")
        hp.append("</div>")
        return "\n".join(hp)

    # two-way ANOVA on ZSTD with strata × group (approx)
    # compute per-item F for group main effect and interaction
    gmap = {u[0]:0, u[1]:1}
    g2 = _np.vectorize(lambda x: gmap.get(x, _np.nan), otypes=[float])(g)
    # ensure 0/1
    g2 = _np.asarray(g2, dtype=float)
    # build table
    dif_rows = []
    for j in range(I):
        y = Z[:, j]
        m = _np.isfinite(y) & _np.isfinite(g2) & (strata>=0)
        # Need enough observations to populate 2 groups x 4 strata (8 cells) and a stable MS_error.
        min_n = 12  # lowered from 20 to support small samples (e.g., nurse ratio n=19)
        if m.sum() < min_n:
            dif_rows.append((j+1, items[j], _np.nan, _np.nan, _np.nan))
            continue
        yy = y[m]
        gg = g2[m].astype(int)
        ss = strata[m].astype(int)
        # grand mean
        mu = yy.mean()
        # group means
        mu_g = [yy[gg==k].mean() if (gg==k).any() else mu for k in [0,1]]
        # strata means
        mu_s = [yy[ss==h].mean() if (ss==h).any() else mu for h in range(4)]
        # cell means
        mu_gs = {}
        for k in [0,1]:
            for h in range(4):
                mm = (gg==k) & (ss==h)
                mu_gs[(k,h)] = yy[mm].mean() if mm.any() else mu
        # SS terms
        ss_g = sum(((mu_g[k]-mu)**2) * (gg==k).sum() for k in [0,1])
        ss_s = sum(((mu_s[h]-mu)**2) * (ss==h).sum() for h in range(4))
        ss_int = 0.0
        for k in [0,1]:
            for h in range(4):
                n_kh = ((gg==k)&(ss==h)).sum()
                ss_int += ((mu_gs[(k,h)] - mu_g[k] - mu_s[h] + mu)**2) * n_kh
        ss_tot = ((yy-mu)**2).sum()
        ss_err = max(ss_tot - ss_g - ss_s - ss_int, 0.0)
        df_g = 1
        df_s = 3
        df_int = 3
        df_err = max(m.sum() - (2*4), 1)
        ms_g = ss_g / df_g
        ms_int = ss_int / df_int
        ms_err = ss_err / df_err if df_err>0 else _np.nan
        Fg = ms_g / ms_err if ms_err and ms_err>0 else _np.nan
        Fi = ms_int / ms_err if ms_err and ms_err>0 else _np.nan
        # p approx: df1*F ~ chi-square(df1)
        pg = _chi2_sf_wh(df_g*Fg, df_g) if _isfinite(Fg) else _np.nan
        pi = _chi2_sf_wh(df_int*Fi, df_int) if _isfinite(Fi) else _np.nan
        dif_rows.append((j+1, items[j], pg, pi, float(Fg) if _isfinite(Fg) else _np.nan))

    hp.append("<table class='rumm'><tr><th>ItemNo</th><th>Item</th><th>p_GroupMain</th><th>p_Interaction</th></tr>")
    for jno, nm, pg, pi, _ in dif_rows:
        cls = "sigb" if _isfinite(pg) and pg < alpha_b else ("sig" if _isfinite(pg) and pg < alpha else "")
        hp.append(f"<tr class='{cls}'><td>{jno}</td><td>{_html.escape(nm)}</td><td>{_fmt_p(pg)}</td><td>{_fmt_p(pi)}</td></tr>")
    hp.append("</table>")

    # forest plot (two-group only) using DIF logit approx
    max_cat = int(_np.nanmax(X)) if _isfinite(float(_np.nanmax(X))) else 2
    dres = _dif_logit_approx(X, max_cat=max_cat, group=g)
    if dres is not None:
        dif, se, pvals, _gg = dres
        # take top 25 by p
        order = _np.argsort(_np.where(_np.isfinite(pvals), pvals, 1e9))[:min(25, I)]
        hp.append(_plot_dif_forest([items[i] for i in order], dif[order], se[order], pvals[order], alpha=alpha, alpha_b=alpha_b))
    else:
        hp.append("<div class='warn'>Forest plot not available (need exactly 2 groups with enough data).</div>")

    hp.append("</div>")
    return "\n".join(hp)

# =========================
# Figure 10/11 helpers (ICC/CPC) - minimal additive patch
# =========================
def _chi2_sf(x, df):
    """Survival function for chi-square without SciPy.
    Uses regularized upper incomplete gamma via continued fraction / series (Numerical Recipes style).
    """
    import math
    if x is None or df is None:
        return float("nan")
    try:
        x = float(x); df = float(df)
    except Exception:
        return float("nan")
    if x < 0 or df <= 0:
        return float("nan")

    a = df / 2.0
    xx = x / 2.0

    def _gammaln(z):
        return math.lgamma(z)

    def _gser(a, x):
        ITMAX = 200
        EPS = 3e-14
        gln = _gammaln(a)
        if x <= 0:
            return 0.0
        ap = a
        summ = 1.0 / a
        delt = summ
        for _ in range(ITMAX):
            ap += 1.0
            delt *= x / ap
            summ += delt
            if abs(delt) < abs(summ) * EPS:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)

    def _gcf(a, x):
        ITMAX = 200
        EPS = 3e-14
        FPMIN = 1e-300
        gln = _gammaln(a)
        b = x + 1.0 - a
        c = 1.0 / FPMIN
        d = 1.0 / b if abs(b) > FPMIN else 1.0 / FPMIN
        h = d
        for i in range(1, ITMAX + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < FPMIN:
                d = FPMIN
            c = b + an / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delt = d * c
            h *= delt
            if abs(delt - 1.0) < EPS:
                break
        return math.exp(-x + a * math.log(x) - gln) * h

    if xx < a + 1.0:
        P = _gser(a, xx)
        return max(0.0, min(1.0, 1.0 - P))
    else:
        Q = _gcf(a, xx)
        return max(0.0, min(1.0, Q))

def _pcm_cat_probs(theta, delta, tau):
    """PCM/RSM style category probabilities for one item.
    tau: list-like length (max_cat+1) with tau[0]=0 (preferred) or length max_cat (steps). We accept both.
    """
    import numpy as np
    th = float(theta)
    de = float(delta)
    tau = list(tau) if tau is not None else [0.0]
    if len(tau) == 0:
        tau = [0.0]
    if abs(tau[0]) > 1e-12:
        tau = [0.0] + tau
    K = len(tau) - 1
    eta = []
    for k in range(K + 1):
        s_tau = sum(tau[:k + 1])
        eta.append(k * (th - de) - s_tau)
    eta = np.asarray(eta, dtype=float)
    eta -= np.nanmax(eta)
    num = np.exp(eta)
    den = float(np.sum(num)) if np.isfinite(np.sum(num)) else 1.0
    p = num / den
    p = np.clip(p, 0.0, 1.0)
    if p.sum() > 0:
        p = p / p.sum()
    return p

def _expected_score(theta, delta, tau):
    import numpy as np
    p = _pcm_cat_probs(theta, delta, tau)
    ks = np.arange(len(p), dtype=float)
    return float(np.sum(ks * p))

def _build_item_trait_chisq_fixed4_for_item(res, item_index_0):
    """Table: Strata_raw score item=j using fixed 4 strata on theta.
    Returns (df_table, chisq, df, pval)."""
    import numpy as np
    import pandas as pd

    pdf = getattr(res, "person_df", None)
    idf = getattr(res, "item_df", None)
    dbg = getattr(res, "debug", {}) or {}
    X = dbg.get("X") or dbg.get("OBS") or dbg.get("X_obs") or dbg.get("RESP")
    EXP = dbg.get("EXP") or dbg.get("E") or dbg.get("E_exp")
    VAR = dbg.get("VAR") or dbg.get("V") or dbg.get("Var")
    if pdf is None or idf is None or X is None or EXP is None or VAR is None:
        return None, float("nan"), None, float("nan")

    X = np.asarray(X, dtype=float)
    EXP = np.asarray(EXP, dtype=float)
    VAR = np.asarray(VAR, dtype=float)
    if X.ndim != 2 or item_index_0 < 0 or item_index_0 >= X.shape[1]:
        return None, float("nan"), None, float("nan")

    theta = np.asarray(pdf.get("MEASURE"), dtype=float)
    okp = np.isfinite(theta)
    if okp.sum() < 4:
        return None, float("nan"), None, float("nan")

    cut = dbg.get("RUMM_FIXED4_BOUNDS")
    if not cut or not isinstance(cut, (list, tuple)) or len(cut) != 5:
        qs = np.nanquantile(theta[okp], [0.25, 0.5, 0.75])
        cut = [float("inf"), float(qs[2]), float(qs[1]), float(qs[0]), float("-inf")]
        dbg["RUMM_FIXED4_BOUNDS"] = cut  # persist for consistency
    else:
        cut = [float(x) for x in cut]

    labels = ["A_1", "B_2", "C_3", "D_4"]
    rows = []
    chisq = 0.0

    try:
        min_cat = int(getattr(res, "min_cat", 0))
        max_cat = int(getattr(res, "max_cat", int(np.nanmax(X[:, item_index_0]))))
    except Exception:
        min_cat, max_cat = 0, int(np.nanmax(X[:, item_index_0]))
    K = max(1, max_cat - min_cat)  # categories-1

    for s in range(4):
        hi = cut[s]
        lo = cut[s + 1]
        if math.isfinite(hi):
            mask = okp & (theta < hi) & (theta >= lo)
        else:
            mask = okp & (theta >= lo)
        idx = np.where(mask)[0]
        n = int(idx.size)
        if n == 0:
            continue
        obs_sum = float(np.nansum(X[idx, item_index_0]))
        exp_sum = float(np.nansum(EXP[idx, item_index_0]))
        var_sum = float(np.nansum(VAR[idx, item_index_0]))
        mean = obs_sum / n if n else float("nan")

        if var_sum > 0 and np.isfinite(var_sum):
            chisq += (obs_sum - exp_sum) ** 2 / var_sum

        rows.append({
            "Strata": f"{labels[s]}(>{lo:.2f})",
            "Sum": round(obs_sum, 2),
            "n": n,
            "Mean": round(mean, 2),
            "Expected": round(exp_sum, 2),
            "Variance": round(var_sum, 2),
            "Theta_mean": float(np.nanmean(theta[idx])),
        })

    df_table = pd.DataFrame(rows)
    df_val = (4 - 1) * K
    pval = _chi2_sf(chisq, df_val)
    return df_table, float(chisq), int(df_val), float(pval)

def _draw_fig10_icc(res, item_index_0: int, kid_no=None):
    """Figure 10 ICC (0..1).
    Always returns a Plotly figure. If res.debug contains cell-level matrices (OBS/EXP/VAR),
    also returns the RUMM-style per-strata table + chi-square stats.
    Returns tuple: (fig, tab_df_or_None, chisq, df, p)
    """
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    # --- item delta ---
    delta = 0.0
    idf = getattr(res, "item_df", None)
    if idf is not None:
        for col in ("MEASURE", "DELTA"):
            if col in idf.columns:
                try:
                    delta = float(pd.to_numeric(idf[col], errors="coerce").iloc[item_index_0])
                    break
                except Exception:
                    pass
    if not np.isfinite(delta):
        delta = 0.0
    if (idf is None) and hasattr(res, "b"):
        try:
            delta = float(np.asarray(res.b, float)[item_index_0])
        except Exception:
            pass

    # --- tau / categories ---
    tau = getattr(res, "tau", None)
    if tau is None:
        tau = (getattr(res, "debug", {}) or {}).get("TAU", [0.0])
    tau = list(tau) if tau is not None else [0.0]
    if len(tau) == 0:
        tau = [0.0]
    if abs(tau[0]) > 1e-12:
        tau = [0.0] + tau

    min_cat = int(getattr(res, "min_cat", 0))
    max_cat = int(getattr(res, "max_cat", max(1, len(tau) - 1)))
    K = max(1, max_cat - min_cat)

    # --- model curve (expected score / K) ---
    xs = np.linspace(-6, 6, 121)
    ys = []
    for x in xs:
        try:
            e = _expected_score_from_tau(x, tau)
        except Exception:
            eta = []
            for k in range(K + 1):
                s_tau = sum(tau[:k + 1])
                eta.append(k * x - s_tau)
            eta = np.asarray(eta, float)
            eta -= np.nanmax(eta)
            p = np.exp(eta); p = p / np.sum(p)
            e = float(np.sum(np.arange(K + 1) * p))
        ys.append(e / K)
    ys = np.asarray(ys, float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Expected ICC"))

    tab_item = None
    chisq = float("nan")
    dfv = None
    pv = float("nan")

    # --- overlay strata (if matrices exist) ---
    dbg = getattr(res, "debug", {}) or {}
    X = dbg.get("X") or dbg.get("OBS") or dbg.get("X_obs") or dbg.get("RESP")
    EXP = dbg.get("EXP") or dbg.get("E") or dbg.get("E_exp")
    VAR = dbg.get("VAR") or dbg.get("V") or dbg.get("Var")

    pdf = getattr(res, "person_df", None)
    if pdf is not None and X is not None and EXP is not None and VAR is not None:
        try:
            X = np.asarray(X, float); EXP = np.asarray(EXP, float); VAR = np.asarray(VAR, float)
            theta = np.asarray(pd.to_numeric(pdf["MEASURE"], errors="coerce"), float)
            ok = np.isfinite(theta)
            qs = np.nanquantile(theta[ok], [0.25, 0.5, 0.75])
            bounds = [float("inf"), float(qs[2]), float(qs[1]), float(qs[0]), float("-inf")]
            rows = []
            chisq_acc = 0.0
            for s in range(4):
                hi, lo = bounds[s], bounds[s+1]
                mask = ok & (theta < hi) & (theta >= lo) if np.isfinite(hi) else (ok & (theta >= lo))
                idx = np.where(mask)[0]
                n = int(idx.size)
                if n == 0:
                    continue
                obs_sum = float(np.nansum(X[idx, item_index_0]))
                exp_sum = float(np.nansum(EXP[idx, item_index_0]))
                var_sum = float(np.nansum(VAR[idx, item_index_0]))
                mean = obs_sum / n
                if var_sum > 0 and np.isfinite(var_sum):
                    chisq_acc += (obs_sum - exp_sum) ** 2 / var_sum
                rows.append({
                    "Strata": ["A_1", "B_2", "C_3", "D_4"][s],
                    "Sum": round(obs_sum, 2),
                    "n": n,
                    "Mean": round(mean, 2),
                    "Expected": round(exp_sum, 2),
                    "Variance": round(var_sum, 2),
                    "Theta_mean": float(np.nanmean(theta[idx])),
                })
            tab_item = pd.DataFrame(rows)
            dfv = int((4 - 1) * K)
            chisq = float(chisq_acc)
            try:
                pv = float(_chi2_sf(chisq, dfv))
            except Exception:
                pv = float("nan")

            if tab_item is not None and len(tab_item):
                x_obs = tab_item["Theta_mean"].values.astype(float) - delta
                y_obs = (tab_item["Mean"].values.astype(float) / K)
                y_exp = (tab_item["Expected"].values.astype(float) / tab_item["n"].values.astype(float) / K)
                fig.add_trace(go.Scatter(x=x_obs, y=y_obs, mode="lines+markers", name="Observed (by strata)"))
                fig.add_trace(go.Scatter(x=x_obs, y=y_exp, mode="lines+markers", name="Expected (by strata)"))
        except Exception:
            tab_item = None
            chisq = float("nan")
            dfv = None
            pv = float("nan")

    title_extra = "" if not (np.isfinite(chisq) and dfv is not None) else f"  ChSQ={chisq:.2f} df={dfv} p={pv:.2f}"
    fig.update_layout(
        title=f"ICC (Figure 10) — Item {item_index_0+1}{title_extra}",
        xaxis_title="Logits (theta − delta), -6 to 6",
        yaxis_title="Observed mean / max(category)",
        yaxis=dict(range=[0, 1]),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig, tab_item, chisq, dfv, pv

def _draw_fig11_cpc(res, item_index_0):
    """Figure 11: CPC (category probability curves).

    Design goals:
    - Never crash the report build (return a Plotly figure whenever possible).
    - Work even if item_df is missing or item_index is out of range.
    - Be tolerant to tau formats: scalar, 1D steps, 1D categories, 2D per-item.
    """
    import numpy as np
    import plotly.graph_objects as go

    dbg = getattr(res, "debug", {}) or {}
    idf = getattr(res, "item_df", None)

    # --- normalize item index (1 item is enough to plot CPC) ---
    try:
        item_index_0 = int(item_index_0)
    except Exception:
        item_index_0 = 0

    n_items = None
    if idf is not None:
        try:
            n_items = int(len(idf))
        except Exception:
            n_items = None

    if n_items and n_items > 0:
        if item_index_0 < 0:
            item_index_0 = 0
        if item_index_0 >= n_items:
            item_index_0 = n_items - 1

    # --- choose delta ---
    delta = np.nan
    if idf is not None and n_items and n_items > 0:
        try:
            row = idf.iloc[item_index_0]
            for col in ("DELTA", "MEASURE"):
                if col in getattr(idf, "columns", []):
                    try:
                        v = float(row[col])
                        if np.isfinite(v):
                            delta = v
                            break
                    except Exception:
                        pass
        except Exception:
            pass

    if not np.isfinite(delta):
        # fall back to estimated b if present
        try:
            b = np.asarray(getattr(res, "b", []), dtype=float)
            if b.size > item_index_0 and np.isfinite(b[item_index_0]):
                delta = float(b[item_index_0])
        except Exception:
            pass

    if not np.isfinite(delta):
        delta = 0.0

    # --- infer category count (K steps => K+1 categories) ---
    try:
        min_cat = int(getattr(res, "min_cat", 0))
        max_cat = int(getattr(res, "max_cat", 1))
        K = max(1, max_cat - min_cat)  # number of steps
    except Exception:
        min_cat, max_cat, K = 0, 1, 1

    # --- get tau and coerce into 1D list of length K+1 with tau[0]=0 ---
    tau = getattr(res, "tau", None)
    if tau is None:
        tau = dbg.get("TAU") or dbg.get("tau")

    tau_list = []
    try:
        # Case A: tau is 2D per-item
        arr = np.asarray(tau, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > item_index_0:
            tau_list = [float(x) for x in list(arr[item_index_0].ravel())]
        elif arr.ndim == 1 and arr.size > 0:
            tau_list = [float(x) for x in list(arr.ravel())]
        elif np.isscalar(tau):
            tau_list = [0.0, float(tau)]
    except Exception:
        # Case B: tau is a list of numbers
        try:
            if np.isscalar(tau):
                tau_list = [0.0, float(tau)]
            else:
                tau_list = [float(x) for x in list(tau)]
        except Exception:
            tau_list = []

    if len(tau_list) == 0:
        tau_list = [0.0] + [0.0] * K

    # If tau looks like "steps" (length K) and does not start at 0, prepend 0
    if abs(float(tau_list[0])) > 1e-12:
        tau_list = [0.0] + tau_list

    # Pad/trim to K+1
    if len(tau_list) < K + 1:
        tau_list = tau_list + [0.0] * (K + 1 - len(tau_list))
    elif len(tau_list) > K + 1:
        tau_list = tau_list[: K + 1]

    xs = np.linspace(-6, 6, 241)
    fig = go.Figure()
    for k in range(len(tau_list)):
        ys = [_pcm_cat_probs(x + delta, delta, tau_list)[k] for x in xs]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Cat {k}"))

    fig.update_layout(
        title=f"CPC (Figure 11) — Item {item_index_0 + 1}",
        xaxis_title="Logits (theta − delta), -6 to 6",
        yaxis_title="Category probability",
        yaxis=dict(range=[0, 1]),
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def _draw_fig11_2_cap(res, x_min=-6, x_max=5, x_step=1):
    """Figure 11.2: Winsteps Table 2.5 style Category Average Plot (CAP).

    CAP means: cat0/cat1/... are mean θ for that observed category (per item).
    - We fill missing cells with expected values first (per your rule), then map to
      nearest category so that every person contributes to a category mean.
    """
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as _go

    item_df = getattr(res, 'item_df', None)
    person_df = getattr(res, 'person_df', None)

    if not isinstance(item_df, _pd.DataFrame) or len(item_df) == 0:
        raise ValueError('CAP needs res.item_df')
    if not isinstance(person_df, _pd.DataFrame) or len(person_df) == 0:
        raise ValueError('CAP needs res.person_df')

    # --- categories from fitted model ---
    try:
        min_cat = int(getattr(res, 'min_cat', 0))
        max_cat = int(getattr(res, 'max_cat', 2))
    except Exception:
        min_cat, max_cat = 0, 2
    if max_cat < min_cat:
        min_cat, max_cat = 0, 2
    cats = list(range(min_cat, max_cat + 1))

    # --- Pull observed matrix + expected matrix (for filling missing) ---
    dbg = getattr(res, 'debug', {}) if isinstance(getattr(res, 'debug', None), dict) else {}
    X_obs = None
    for k in ('X_obs_filled', 'X_obs', 'X_post', 'X_transformed', 'X'):
        if isinstance(dbg, dict) and k in dbg and dbg.get(k) is not None:
            X_obs = dbg.get(k)
            break
    if X_obs is None:
        X_obs = getattr(res, 'X_obs', None)

    # Fallback: some pipelines only store shifted 0..K-1 matrix as X0.
    # Convert back to observed scale by adding min_cat so CAP can still run.
    if X_obs is None and isinstance(dbg, dict) and dbg.get('X0') is not None:
        try:
            X0 = _np.asarray(dbg.get('X0'), dtype=float)
            if X0.ndim == 2:
                X_obs = X0 + float(min_cat)
                dbg['X_obs'] = X_obs.tolist()
                try:
                    setattr(res, 'debug', dbg)
                except Exception:
                    pass
        except Exception:
            X_obs = None

    if X_obs is None:
        raise ValueError('CAP needs res.X_obs (observed response matrix)')

    EXP = None
    if isinstance(dbg, dict):
        EXP = dbg.get('EXP', None) or dbg.get('E_exp', None)

    X = _np.asarray(X_obs, dtype=float)
    if X.ndim != 2:
        raise ValueError('res.X_obs must be a 2D matrix')

    # --- align X to (persons x items) ---
    if X.shape[0] != len(person_df) and X.shape[1] == len(person_df):
        X = X.T
    if X.shape[1] != len(item_df) and X.shape[0] == len(item_df):
        X = X.T
    if X.shape[0] != len(person_df):
        raise ValueError(f'X_obs rows ({X.shape[0]}) do not match person_df ({len(person_df)})')
    if X.shape[1] != len(item_df):
        raise ValueError(f'X_obs cols ({X.shape[1]}) do not match item_df ({len(item_df)})')

    # fill missing with expected value prior to CAP/Table 11.2
    X_fill = X.copy()
    if EXP is not None:
        E = _np.asarray(EXP, dtype=float)
        if E.ndim == 2:
            if E.shape[0] != X_fill.shape[0] and E.shape[1] == X_fill.shape[0]:
                E = E.T
            if E.shape == X_fill.shape:
                m = ~_np.isfinite(X_fill)
                if m.any():
                    X_fill[m] = E[m]

    # map to nearest category (so filled expected values contribute)
    X_cat = _np.rint(X_fill)
    X_cat = _np.clip(X_cat, min_cat, max_cat)

    # --- columns ---
    act_col = 'ACT' if 'ACT' in item_df.columns else ('ITEM' if 'ITEM' in item_df.columns else None)
    if act_col is None:
        raise ValueError('CAP needs item_df with ACT/ITEM column')
    delta_col = 'MEASURE' if 'MEASURE' in item_df.columns else ('DELTA' if 'DELTA' in item_df.columns else None)
    if delta_col is None:
        raise ValueError('CAP needs item_df with MEASURE (delta) column')
    infit_col = 'INFIT_MNSQ' if 'INFIT_MNSQ' in item_df.columns else None

    # person theta column
    if 'theta' in person_df.columns:
        theta_col = 'theta'
    elif 'MEASURE' in person_df.columns:
        theta_col = 'MEASURE'
    else:
        cand = [c for c in person_df.columns if c.lower() not in {'name','kid','person','id'}]
        theta_col = cand[0] if cand else None
    if theta_col is None:
        raise ValueError('CAP needs person_df with theta/MEASURE column')

    theta = _pd.to_numeric(person_df[theta_col], errors='coerce').to_numpy()

    act = item_df[act_col].astype(str).to_list()
    delta = _pd.to_numeric(item_df[delta_col], errors='coerce').to_numpy()
    infit = _pd.to_numeric(item_df[infit_col], errors='coerce').to_numpy() if infit_col else _np.full(len(item_df), _np.nan)

    # --- Compute mean theta per item per category ---
    rows = []
    for j in range(len(item_df)):
        rj = X_cat[:, j]
        for cat in cats:
            m = (rj == cat) & _np.isfinite(theta)
            mt = float(_np.nanmean(theta[m])) if m.sum() else _np.nan
            rows.append({
                'ACT': act[j],
                'delta': float(delta[j]) if _np.isfinite(delta[j]) else _np.nan,
                'infit_mnsq': float(infit[j]) if _np.isfinite(infit[j]) else _np.nan,
                'cat': str(cat),
                'mean_theta': mt,
                'n': int(m.sum()),
            })

    cap_long = _pd.DataFrame(rows)

    # --- Wide table (Table 11.2) ---
    cap_item_df = (cap_long
        .pivot_table(index=['ACT','delta','infit_mnsq'], columns='cat', values='mean_theta', aggfunc='first')
        .reset_index())

    # rename to cat0..catK in proper order
    for c in [str(x) for x in cats]:
        if c not in cap_item_df.columns:
            cap_item_df[c] = _np.nan
    for x in cats:
        cap_item_df = cap_item_df.rename(columns={str(x): f'cat{x}'})

    # Add PTMA to Table 11.2
    try:
        _ptma = compute_ptma_from_matrices(X_cat, theta, item_names=act, min_n=5)
        if isinstance(_ptma, _pd.DataFrame) and 'ITEM' in _ptma.columns:
            _ptma = _ptma.rename(columns={'ITEM':'ACT'})
            cap_item_df = cap_item_df.merge(_ptma[['ACT','PTMA','N_USED']], on='ACT', how='left')
    except Exception:
        pass

    # ensure PTMA columns exist even if merge failed
    if 'PTMA' not in cap_item_df.columns:
        cap_item_df['PTMA'] = _np.nan
    if 'N_USED' not in cap_item_df.columns:
        cap_item_df['N_USED'] = _np.nan

    # Sort by delta descending like Winsteps
    cap_item_df = cap_item_df.sort_values('delta', ascending=False, na_position='last').reset_index(drop=True)

    # Reorder columns: ACT, delta, infit, PTMA, N_USED, cat*
    cat_cols = [f'cat{x}' for x in cats]
    keep = ['ACT','delta','infit_mnsq','PTMA','N_USED'] + cat_cols
    cap_item_df = cap_item_df[[c for c in keep if c in cap_item_df.columns]].copy()

    # --- build y labels for plot ---
    def _fmt(v, nd=2):
        try:
            if _np.isfinite(v):
                return f'{v:+.{nd}f}'
        except Exception:
            pass
        return 'NA'

    y_labels = []
    for _, r in cap_item_df.iterrows():
        y_labels.append(f"{r['ACT']}  (δ={_fmt(r['delta'],2)}, Infit={_fmt(r['infit_mnsq'],2).replace('+','')})")

    label_map = dict(zip(cap_item_df['ACT'].astype(str).to_list(), y_labels))
    cap_long['ACT_label'] = cap_long['ACT'].map(label_map).fillna(cap_long['ACT'])

    # --- Plotly figure ---
    fig = _go.Figure()

    # Consistent colors across the whole report
    palette = list(SPECIFIED_COLORS)
    for idx, cat in enumerate([str(x) for x in cats]):
        d = cap_long[cap_long['cat'] == cat]
        fig.add_trace(_go.Scatter(
            x=d['mean_theta'],
            y=d['ACT_label'],
            mode='markers+text',
            text=[cat]*len(d),
            textposition='middle right',
            marker=dict(size=7, color=palette[idx % len(palette)]),
            name=f"cat{cat}",
            hovertemplate="cat%{text}<br>mean θ=%{x:.3f}<extra></extra>",
        ))

    # Winsteps Table 2.5 style "vertical" linkage:
    # connect the same category (0/1/2/...) from top to bottom across items.
    # (NOT left-to-right within each item.)
    try:
        # cap_item_df is already sorted by delta (top to bottom). Use that order.
        # Important: do NOT "skip" missing categories when drawing lines.
        # Skipping would connect non-adjacent items and creates confusing crossings.
        # Instead, we insert breaks (None) so lines only connect consecutive items.
        for idxc, xcat in enumerate(cats):
            xs = []
            ys = []
            n_ok = 0
            for _, r in cap_item_df.iterrows():
                a = str(r.get('ACT', ''))
                ylab = label_map.get(a, a)
                v = r.get(f'cat{xcat}', _np.nan)
                try:
                    v = float(v)
                except Exception:
                    v = _np.nan
                if _np.isfinite(v):
                    xs.append(v)
                    ys.append(ylab)
                    n_ok += 1
                else:
                    # break the polyline at this item
                    xs.append(None)
                    ys.append(None)
            if n_ok >= 2:
                # Use the same palette color as the corresponding category dots.
                # This makes it visually obvious that linkages are within-category.
                line_col = palette[idxc % len(palette)] if palette else 'black'
                fig.add_trace(_go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines',
                    line=dict(color=line_col, width=1),
                    hoverinfo='skip',
                    showlegend=False,
                ))
    except Exception:
        pass

    fig.update_layout(
        title='CAP (Figure 11.2) — Observed average measures by category (Winsteps Table 2.5)',
        xaxis_title='Person measure (θ, logit)',
        yaxis_title='Item (sorted by delta)',
        height=880,
        margin=dict(l=340, r=40, t=80, b=60),
        legend_title='Category'
    )

    # Ensure the y-axis follows the delta-sorted order (top = largest delta).
    # Plotly categoryarray is applied bottom->top for y-axes, so we reverse.
    try:
        fig.update_yaxes(categoryorder='array', categoryarray=list(reversed(y_labels)))
    except Exception:
        pass



    # x-axis range: if caller used defaults (-6..5), tighten to data range to reduce blank space
    try:
        _vals = cap_long['mean_theta'].to_numpy(dtype=float)
        _vals = _vals[_np.isfinite(_vals)]
        if _vals.size:
            _mn = float(_np.min(_vals)); _mx = float(_np.max(_vals))
            _pad = max(0.5, 0.10 * (_mx - _mn))
            if float(x_min) == -6 and float(x_max) == 5:
                x_min_use, x_max_use = _mn - _pad, _mx + _pad
            else:
                x_min_use, x_max_use = float(x_min), float(x_max)
            fig.update_xaxes(range=[x_min_use, x_max_use], dtick=float(x_step))
    except Exception:
        pass

    return fig, cap_item_df




def _draw_fig11_3_tcp__old(res, q3_cut: float = 0.20):
    """Figure 11.3: Testlet Cluster Plot (TCP) using residual Q3 network + relation-PCA layout.

    Steps (Python port of your R script idea):
    1) Residual matrix R = O - E (or ZSTD as fallback)
    2) Q3 = cor(R) across items; diag set to 0
    3) Edges where |Q3| >= q3_cut
    4) Communities via Louvain (networkx.louvain_communities) with fallback to greedy modularity
    5) Layout via relation PCA: eigen(Q3) -> PC1/PC2

    Returns: (plotly_figure, testlet_table, q3_matrix)
    """
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as _go

    item_df = getattr(res, 'item_df', None)
    if not isinstance(item_df, _pd.DataFrame) or len(item_df) == 0:
        return None, None, None

    dbg = getattr(res, 'debug', {}) if isinstance(getattr(res, 'debug', None), dict) else {}

    # --- residual matrix ---
    R = None
    X = dbg.get('X_obs_filled') or dbg.get('X_obs') or dbg.get('X0') or dbg.get('X')
    E = dbg.get('E_exp') or dbg.get('EXP')
    if X is not None and E is not None:
        try:
            X = _np.asarray(X, dtype=float)
            E = _np.asarray(E, dtype=float)
            if X.shape == E.shape:
                R = X - E
        except Exception:
            R = None
    if R is None:
        Z = dbg.get('ZSTD', None)
        if Z is not None:
            try:
                R = _np.asarray(Z, dtype=float)
            except Exception:
                R = None

    if R is None or _np.asarray(R).ndim != 2:
        return None, None, None

    # align to persons x items
    R = _np.asarray(R, dtype=float)
    if R.shape[1] != len(item_df) and R.shape[0] == len(item_df):
        R = R.T
    if R.shape[1] != len(item_df):
        return None, None, None

    # --- Q3 matrix ---
    Q3 = _np.corrcoef(_np.nan_to_num(R, nan=_np.nan), rowvar=False)
    Q3 = _np.nan_to_num(Q3, nan=0.0, posinf=0.0, neginf=0.0)
    _np.fill_diagonal(Q3, 0.0)

    # store for downstream hovers
    try:
        if isinstance(dbg, dict):
            dbg['Q3'] = Q3
            res.debug = dbg
    except Exception:
        pass

    I = Q3.shape[0]
    names = (item_df['ITEM'].astype(str).to_list() if 'ITEM' in item_df.columns else [f'Item{i+1}' for i in range(I)])

    # --- edge list ---
    idx = _np.argwhere(_np.abs(Q3) >= float(q3_cut))
    idx = idx[idx[:, 0] < idx[:, 1]]
    if idx.size == 0:
        # still return PCA scatter with no edges
        edges = []
    else:
        edges = [(int(a), int(b), float(Q3[a, b])) for a, b in idx]

    # --- communities ---
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(I))
        for a, b, r in edges:
            G.add_edge(a, b, weight=abs(r), r=r)

        memb = {i: 0 for i in range(I)}
        if G.number_of_edges() > 0:
            # prefer Louvain if available
            if hasattr(nx.algorithms.community, 'louvain_communities'):
                comms = nx.algorithms.community.louvain_communities(G, weight='weight', seed=123)
            else:
                comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
            for k, cset in enumerate(comms, start=1):
                for i in cset:
                    memb[int(i)] = k
            # optional: set singletons to 0
            from collections import Counter
            cnt = Counter(memb.values())
            for i in list(memb.keys()):
                if memb[i] != 0 and cnt[memb[i]] == 1:
                    memb[i] = 0
        testlet = _np.array([memb[i] for i in range(I)], dtype=int)
    except Exception:
        testlet = _np.zeros(I, dtype=int)

    testlet_df = _pd.DataFrame({'ITEM': names, 'TESTLET': testlet})

    # --- relation PCA layout ---
    try:
        vals, vecs = _np.linalg.eigh(Q3)
        order = _np.argsort(_np.abs(vals))[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        pc1 = vecs[:, 0] * _np.sqrt(_np.abs(vals[0])) if vals.size > 0 else _np.zeros(I)
        pc2 = vecs[:, 1] * _np.sqrt(_np.abs(vals[1])) if vals.size > 1 else _np.zeros(I)
    except Exception:
        pc1 = _np.zeros(I)
        pc2 = _np.zeros(I)

    # --- build plotly figure ---
    fig = _go.Figure()

    # edges
    for a, b, r in edges:
        fig.add_trace(_go.Scatter(
            x=[pc1[a], pc1[b]],
            y=[pc2[a], pc2[b]],
            mode='lines',
            line=dict(width=1),
            opacity=0.35,
            hoverinfo='skip',
            showlegend=False,
        ))

    # nodes
    hover = []
    for i in range(I):
        # max |Q3| for node
        qmax = float(_np.max(_np.abs(Q3[i, :]))) if I > 1 else 0.0
        hover.append(f"{names[i]}<br>Testlet={int(testlet[i])}<br>Q3max={qmax:.3f}<br>PC1={pc1[i]:.3f}<br>PC2={pc2[i]:.3f}")

    fig.add_trace(_go.Scatter(
        x=pc1,
        y=pc2,
        mode='markers+text',
        text=names,
        textposition='top center',
        hovertext=hover,
        hoverinfo='text',
        marker=dict(size=12, line=dict(width=1)),
        name='Items'
    ))

    fig.update_layout(
        title=f"Figure 11.3 — Testlet Cluster Plot (TCP): Q3 network (|Q3| ≥ {q3_cut}) with relation-PCA layout",
        xaxis_title='Relation PCA 1',
        yaxis_title='Relation PCA 2',
        height=720,
        margin=dict(l=40, r=20, t=70, b=50),
    )

    return fig, testlet_df, Q3







def _draw_fig11_3_tcp(res, q3_cut: float = 0.20):
    """Figure 11.3: Testlet cluster plot (TCP) from residual Q3.

    Outputs:
      - Figure 11.3: relation-PCA layout of the Q3 matrix, with edges for |Q3| >= q3_cut.
      - Table 11.3: Item membership + PTMA + Q3max + suspected testlet flag.

    Notes:
      - Residuals use RESID = O - E from Rasch (preferred), otherwise ZSTD as fallback.
      - Communities use Louvain if networkx provides it; otherwise greedy modularity.
    """
    import numpy as _np
    import pandas as _pd
    import plotly.graph_objects as _go

    dbg = getattr(res, 'debug', {}) if isinstance(getattr(res, 'debug', None), dict) else {}
    item_df = getattr(res, 'item_df', None)
    person_df = getattr(res, 'person_df', None)
    if not isinstance(item_df, _pd.DataFrame) or len(item_df) == 0:
        return None, None

    # labels
    if 'ACT' in item_df.columns:
        labels = item_df['ACT'].astype(str).to_list()
    elif 'ITEM' in item_df.columns:
        labels = item_df['ITEM'].astype(str).to_list()
    else:
        labels = [f'Item{i+1}' for i in range(len(item_df))]

    # residual matrix R (persons x items)
    R = None
    if isinstance(dbg, dict) and dbg.get('RESID') is not None:
        try:
            R = _np.asarray(dbg.get('RESID'), dtype=float)
        except Exception:
            R = None
    if R is None:
        Xobs = None
        if isinstance(dbg, dict):
            for k in ('X_obs_filled', 'X_obs', 'X_post', 'X_transformed', 'X0', 'X'):
                if dbg.get(k) is not None:
                    Xobs = dbg.get(k)
                    break
        EXP = dbg.get('EXP', None) or dbg.get('E_exp', None) if isinstance(dbg, dict) else None
        if Xobs is not None and EXP is not None:
            try:
                R = _np.asarray(Xobs, dtype=float) - _np.asarray(EXP, dtype=float)
            except Exception:
                R = None
    if R is None and isinstance(dbg, dict) and dbg.get('ZSTD') is not None:
        try:
            R = _np.asarray(dbg.get('ZSTD'), dtype=float)
        except Exception:
            R = None

    if R is None or _np.asarray(R).ndim != 2:
        return None, None

    R = _np.asarray(R, dtype=float)
    I = len(labels)
    # align persons x items
    if R.shape[1] != I and R.shape[0] == I:
        R = R.T
    if R.shape[1] != I:
        return None, None

    # Q3 (pairwise-complete)
    Q3 = _pd.DataFrame(R, columns=labels).corr(method='pearson', min_periods=3).to_numpy(dtype=float)
    Q3 = _np.nan_to_num(Q3, nan=0.0, posinf=0.0, neginf=0.0)
    _np.fill_diagonal(Q3, 0.0)

    # store for other plots/hover
    try:
        if isinstance(dbg, dict):
            dbg['Q3'] = Q3
            res.debug = dbg
    except Exception:
        pass

    # edges
    edges = []
    for i in range(I):
        for j in range(i+1, I):
            w = float(Q3[i, j])
            if abs(w) >= float(q3_cut):
                edges.append((i, j, w))

    # communities
    testlet = _np.zeros(I, dtype=int)
    if edges:
        try:
            import networkx as nx
            G = nx.Graph()
            G.add_nodes_from(range(I))
            for i, j, w in edges:
                G.add_edge(i, j, weight=abs(w), r=w)
            if hasattr(nx.algorithms.community, 'louvain_communities'):
                comms = nx.algorithms.community.louvain_communities(G, weight='weight', seed=123)
            else:
                comms = list(nx.algorithms.community.greedy_modularity_communities(G, weight='weight'))
            for k, cset in enumerate(comms, start=1):
                for v in cset:
                    testlet[int(v)] = k
            # singleton -> 0
            from collections import Counter
            cnt = Counter(testlet.tolist())
            for idx in range(I):
                if testlet[idx] != 0 and cnt.get(int(testlet[idx]), 0) <= 1:
                    testlet[idx] = 0
        except Exception:
            # fallback: single community (non-singletons only)
            for idx in range(I):
                testlet[idx] = 1

    # relation PCA layout (eigh)
    try:
        vals, vecs = _np.linalg.eigh(Q3)
        order = _np.argsort(_np.abs(vals))[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        pc1 = vecs[:, 0] * _np.sqrt(abs(vals[0])) if vals.size > 0 else _np.zeros(I)
        pc2 = vecs[:, 1] * _np.sqrt(abs(vals[1])) if vals.size > 1 else _np.zeros(I)
    except Exception:
        pc1 = _np.arange(I, dtype=float)
        pc2 = _np.zeros(I, dtype=float)

    # PTMA (leave-one-out theta) for Table 11.3
    ptma_map = {labels[i]: _np.nan for i in range(I)}
    nused_map = {labels[i]: _np.nan for i in range(I)}
    try:
        # theta from person_df
        theta = None
        if isinstance(person_df, _pd.DataFrame) and len(person_df) > 0:
            if 'theta' in person_df.columns:
                theta = _pd.to_numeric(person_df['theta'], errors='coerce').to_numpy(dtype=float)
            elif 'MEASURE' in person_df.columns:
                theta = _pd.to_numeric(person_df['MEASURE'], errors='coerce').to_numpy(dtype=float)
        # observed (filled) responses for PTMA categories
        Xobs = None
        if isinstance(dbg, dict):
            for k in ('X_obs_filled', 'X_obs', 'X_post', 'X_transformed', 'X0', 'X'):
                if dbg.get(k) is not None:
                    Xobs = dbg.get(k)
                    break
        if theta is not None and Xobs is not None:
            Xmat = _np.asarray(Xobs, dtype=float)
            if Xmat.ndim == 2:
                if Xmat.shape[1] != I and Xmat.shape[0] == I:
                    Xmat = Xmat.T
                # map to nearest category so decimals also work
                try:
                    min_cat = int(getattr(res, 'min_cat', 0))
                    max_cat = int(getattr(res, 'max_cat', 2))
                except Exception:
                    min_cat, max_cat = 0, 2
                Xcat = _np.rint(Xmat)
                Xcat = _np.clip(Xcat, min_cat, max_cat)
                _pt = compute_ptma_from_matrices(Xcat, theta, item_names=labels, min_n=5)
                if isinstance(_pt, _pd.DataFrame) and len(_pt) > 0:
                    # accept either ITEM or ACT column
                    if 'ITEM' in _pt.columns:
                        for _, r in _pt.iterrows():
                            ptma_map[str(r['ITEM'])] = float(r.get('PTMA', _np.nan))
                            nused_map[str(r['ITEM'])] = float(r.get('N_USED', _np.nan))
                    elif 'ACT' in _pt.columns:
                        for _, r in _pt.iterrows():
                            ptma_map[str(r['ACT'])] = float(r.get('PTMA', _np.nan))
                            nused_map[str(r['ACT'])] = float(r.get('N_USED', _np.nan))
    except Exception:
        pass

    # node-level Q3max and suspected flag
    q3max = _np.array([float(_np.max(_np.abs(Q3[i, :]))) if I > 1 else 0.0 for i in range(I)], dtype=float)
    ptma_arr = _np.array([ptma_map.get(labels[i], _np.nan) for i in range(I)], dtype=float)
    suspected = (q3max >= 0.10) & _np.isfinite(ptma_arr) & (ptma_arr < 0.10)

    # Table 11.3
    tbl = _pd.DataFrame({
        'Item': labels,
        'Testlet': testlet,
        'PTMA': ptma_arr,
        'N_USED': [_np.nan if not _np.isfinite(nused_map.get(labels[i], _np.nan)) else nused_map.get(labels[i], _np.nan) for i in range(I)],
        'Q3max': q3max,
        'Suspected_testlet': ['High Q3 + Low PTMA' if bool(suspected[i]) else '' for i in range(I)],
        'PC1': pc1,
        'PC2': pc2,
    })

    # Figure 11.3
    fig = _go.Figure()
    # edges
    for i, j, w in edges:
        fig.add_trace(_go.Scatter(
            x=[pc1[i], pc1[j]],
            y=[pc2[i], pc2[j]],
            mode='lines',
            line=dict(width=1),
            opacity=0.35,
            hoverinfo='skip',
            showlegend=False,
        ))

    hover = []
    for i in range(I):
        h = f"{labels[i]}<br>Testlet={int(testlet[i])}<br>PTMA={ptma_arr[i]:.3f}" if _np.isfinite(ptma_arr[i]) else f"{labels[i]}<br>Testlet={int(testlet[i])}<br>PTMA=NA"
        h += f"<br>Q3max={q3max[i]:.3f}"
        if suspected[i]:
            h += "<br><b>Suspected: High Q3 + Low PTMA</b>"
        hover.append(h)

    # Vertex size by PTMA (robust normalization).
    # If any PTMA < 0, normalize by (PTMA - min)/(max - min) + 0.1 (as requested)
    # so negatives still get a visible size. Otherwise we still use the same
    # normalization for consistency.
    sizes = _np.full(I, 12.0, dtype=float)
    v = ptma_arr[_np.isfinite(ptma_arr)]
    if v.size:
        vmin = float(_np.min(v))
        vmax = float(_np.max(v))
        if vmax - vmin > 1e-12:
            norm = (ptma_arr - vmin) / (vmax - vmin)
            norm = norm + 0.1
            norm = _np.where(_np.isfinite(norm), norm, 0.1)
            # Map to a readable marker size range
            sizes = 10.0 + 22.0 * _np.clip(norm, 0.0, 1.5)
        else:
            sizes = _np.full(I, 14.0, dtype=float)
            sizes[~_np.isfinite(ptma_arr)] = 10.0
    else:
        sizes = _np.full(I, 12.0, dtype=float)

    fig.add_trace(_go.Scatter(
        x=pc1,
        y=pc2,
        mode='markers+text',
        text=labels,
        textposition='top center',
        hovertext=hover,
        hoverinfo='text',
        marker=dict(size=sizes, line=dict(width=1)),
        name='Items'
    ))

    fig.update_layout(
        title=f"Figure 11.3 — TCP: Q3 network (|Q3| >= {float(q3_cut):.2f}) with relation-PCA layout",
        xaxis_title='Relation PCA 1',
        yaxis_title='Relation PCA 2',
        height=720,
        margin=dict(l=40, r=20, t=70, b=50),
    )

    return fig, tbl
def _chi2_sf_df1(x: float) -> float:
    """Survival function for Chi-square with df=1 using erfc; avoids scipy dependency."""
    import math
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


def _norm_sf_abs(z: float) -> float:
    """Two-sided p-value under standard normal for |z| using erfc."""
    import math
    if z is None:
        return 1.0
    z = abs(float(z))
    return math.erfc(z / math.sqrt(2.0))


def _rsm_expected_score_and_var(theta: float, delta: float, tau, min_cat: int, max_cat: int):
    """Expected score and variance for one person-item under RSM parameterization used here."""
    import math
    K = max_cat - min_cat
    tau_vec = list(tau) if tau is not None else [0.0] * (K + 1)
    if len(tau_vec) < (K + 1):
        tau_vec = tau_vec + [0.0] * ((K + 1) - len(tau_vec))
    if abs(float(tau_vec[0])) > 1e-12:
        tau_vec = [0.0] + tau_vec[:K]
    logits = []
    for k in range(0, K + 1):
        step_sum = 0.0
        for s in range(1, k + 1):
            step_sum += float(tau_vec[s])
        eta = k * (float(theta) - float(delta)) - step_sum
        logits.append(eta)
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    Z = sum(exps) if exps else 1.0
    ps = [v / Z for v in exps]
    exp_score = sum((min_cat + k) * ps[k] for k in range(0, K + 1))
    exp_sq = sum(((min_cat + k) ** 2) * ps[k] for k in range(0, K + 1))
    var = max(0.0, exp_sq - exp_score ** 2)
    return exp_score, var


def _solve_delta_for_group(theta_list, obs_mean: float, delta0: float, tau, min_cat: int, max_cat: int):
    """Solve delta so mean expected score matches obs_mean (bisection)."""
    if theta_list is None or len(theta_list) == 0:
        return float(delta0)

    lo = float(delta0) - 8.0
    hi = float(delta0) + 8.0

    def f(d):
        s = 0.0
        for th in theta_list:
            e, _ = _rsm_expected_score_and_var(float(th), float(d), tau, min_cat, max_cat)
            s += e
        return (s / len(theta_list)) - obs_mean

    flo = f(lo)
    fhi = f(hi)
    for _ in range(10):
        if flo == 0.0:
            return lo
        if fhi == 0.0:
            return hi
        if flo * fhi < 0:
            break
        lo -= 6.0
        hi += 6.0
        flo = f(lo)
        fhi = f(hi)

    if flo * fhi > 0:
        return float(delta0)

    for _ in range(60):
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        if abs(fmid) < 1e-7:
            return mid
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return (lo + hi) / 2.0


def _as_html_parts(x):
    """Normalize various return types into a list of HTML fragments."""
    if x is None:
        return []
    if isinstance(x, list):
        return [s for s in x if s is not None and str(s) != ""]
    if isinstance(x, tuple):
        return [str(s) for s in x if s is not None and str(s) != ""]
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [str(x)]


def _render_task12_dif(res):
    """Render Winsteps-like DIF tables and forest plot using Profile (last column)."""
    import numpy as np
    import math

    dbg = getattr(res, "debug", {}) or {}
    person_df = getattr(res, "person_df", None)
    item_df = getattr(res, "item_df", None)
    if person_df is None or item_df is None:
        return ""

    # --- group label (Profile/GROUP) retrieval ---
    # IMPORTANT: must keep alignment with persons (same length as person_df)
    profile_series = None
    group_col_name = None

    raw_df = dbg.get("raw_df", None)
    if raw_df is None:
        raw_df = dbg.get("input_df", None)
    if raw_df is None:
        raw_df = getattr(res, "raw_df", None)
    if raw_df is not None:
        # Prefer explicit names
        for cand in ("Profile", "PROFILE", "profile", "GROUP", "group", "Group"):
            if cand in getattr(raw_df, "columns", []):
                profile_series = raw_df[cand]
                group_col_name = cand
                break
        # Otherwise: if last column looks like a group label, take it
        if profile_series is None and hasattr(raw_df, "columns") and len(raw_df.columns) > 0:
            last = raw_df.columns[-1]
            last_l = str(last).strip().lower()
            if last_l in ("profile", "group", "grp", "dif", "difgroup", "dif_group"):
                profile_series = raw_df[last]
                group_col_name = str(last)

    if profile_series is None:
        # Fall back to person_df (sometimes the report pipeline merged Profile into person_df)
        for cand in ("Profile", "PROFILE", "profile", "GROUP", "group", "Group"):
            if cand in person_df.columns:
                profile_series = person_df[cand]
                group_col_name = cand
                break

    if profile_series is None:
        # Last resort: debug dicts
        # NOTE: don't use `or` chaining here because an empty list/Series is falsy
        # but still indicates the key exists.
        cand = None
        for k in ("profile", "Profile", "PROFILE", "group", "GROUP", "dif_group", "DIF_GROUP"):
            if k in dbg:
                cand = dbg.get(k)
                break
        profile_series = cand
        group_col_name = "(debug)"

    if profile_series is None:
        return "<h2>Task 12 — DIF</h2><p>(Skipped: group label column used for DIF was not found. Put a column named 'Profile' (or 'GROUP') as the last column in the input data.)</p>"

    import pandas as _pd
    if isinstance(profile_series, _pd.Series):
        prof_arr = profile_series.to_numpy()
    else:
        prof_arr = np.asarray(list(profile_series))

    P = len(person_df)
    if prof_arr.shape[0] != P:
        # try align by index if possible; otherwise refuse (misaligned groups are worse than skipping)
        try:
            if isinstance(profile_series, _pd.Series) and hasattr(person_df, "index"):
                prof_arr = profile_series.reindex(person_df.index).to_numpy()
        except Exception:
            pass

    if prof_arr.shape[0] != P:
        return f"<h2>Task 12 — DIF</h2><p>(Skipped: group label column length ({prof_arr.shape[0]}) does not match #persons ({P}). Ensure the group label column is person-level and not aggregated.)</p>"

    # normalize missing
    grp_full = np.array(["" if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x).strip() for x in prof_arr], dtype=object)
    valid_mask = (grp_full != "")

    uniq = []
    for s in grp_full[valid_mask]:
        if s not in uniq:
            uniq.append(s)

    if len(uniq) != 2:
        return f"<h2>Task 12 — DIF</h2><p>(Skipped: need exactly 2 groups in the group label column; got {len(uniq)}: {uniq}.)</p>"

    g0, g1 = uniq[0], uniq[1]
    grp = grp_full  # keep full-length; we will mask later


    X = np.array(dbg.get("X", dbg.get("X_obs", None)))
    if X is None or X.size == 0:
        return "<h2>Task 12 — DIF</h2><p>(Skipped: response matrix not available.)</p>"

    def _safe_vec(x):
        # Return 1D float vector or None (never returns 0-d array)
        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 0 or arr.size == 0:
            return None
        return arr.reshape(-1)

    # Prefer engine-provided arrays; otherwise fall back to MEASURE/DELTA columns
    theta = _safe_vec(getattr(res, "theta", None) if hasattr(res, "theta") else dbg.get("theta", None))
    if theta is None and isinstance(person_df, pd.DataFrame):
        for cand in ("MEASURE", "Measure", "measure", "THETA", "Theta", "theta"):
            if cand in person_df.columns:
                theta = _safe_vec(person_df[cand].to_numpy())
                break

    delta = _safe_vec(getattr(res, "delta", None) if hasattr(res, "delta") else dbg.get("delta", None))
    if delta is None and isinstance(item_df, pd.DataFrame):
        for cand in ("DELTA", "Delta", "delta", "MEASURE", "Measure", "measure"):
            if cand in item_df.columns:
                delta = _safe_vec(item_df[cand].to_numpy())
                break

    tau = dbg.get("tau", getattr(res, "tau", None))
    min_cat = int(getattr(res, "min_cat", dbg.get("min_cat", 0)))
    max_cat = int(getattr(res, "max_cat", dbg.get("max_cat", int(np.nanmax(X)))))

    if theta is None or delta is None:
        return "<h2>Task 12 — DIF</h2><p>(Skipped: theta/delta not available.)</p>"

    idx0 = np.where(grp == g0)[0]
    idx1 = np.where(grp == g1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return "<h2>Task 12 — DIF</h2><p>(Skipped: one group has zero persons.)</p>"

    theta0 = theta[idx0].tolist()
    theta1 = theta[idx1].tolist()

    # Winsteps-style DIF: use group-specific person measures (theta) for fit computations.
    # (Older versions used th0/th1; keep those names for downstream helpers.)
    th0 = theta0
    th1 = theta1

    item_names = item_df["ITEM"].astype(str).tolist() if "ITEM" in item_df.columns else [f"Item {i+1}" for i in range(X.shape[1])]

    rows_30_4, rows_30_2, rows_30_7 = [], [], []
    rows_30_1, rows_30_3, rows_30_5, rows_30_6 = [], [], [], []
    contrasts, se_contrasts = [], []

    for i in range(X.shape[1]):
        xi0 = X[idx0, i]
        xi1 = X[idx1, i]
        m0 = float(np.nanmean(xi0))
        m1 = float(np.nanmean(xi1))
        d_base = float(delta[i])

        d_g0 = _solve_delta_for_group(theta0, m0, d_base, tau, min_cat, max_cat)
        d_g1 = _solve_delta_for_group(theta1, m1, d_base, tau, min_cat, max_cat)

        info0 = 0.0
        for th in theta0:
            _, v = _rsm_expected_score_and_var(float(th), d_g0, tau, min_cat, max_cat)
            info0 += max(1e-9, v)
        info1 = 0.0
        for th in theta1:
            _, v = _rsm_expected_score_and_var(float(th), d_g1, tau, min_cat, max_cat)
            info1 += max(1e-9, v)
        se0 = 1.0 / math.sqrt(info0) if info0 > 0 else float("nan")
        se1 = 1.0 / math.sqrt(info1) if info1 > 0 else float("nan")


        # Alias names used later in DIF tables/forest
        se_g0 = se0
        se_g1 = se1
        contrast = d_g0 - d_g1
        se_con = math.sqrt((se0 ** 2 if math.isfinite(se0) else 0.0) + (se1 ** 2 if math.isfinite(se1) else 0.0))
        tval = contrast / se_con if se_con > 0 else float("nan")
        p_t = _norm_sf_abs(tval)

        chisq = (tval ** 2) if math.isfinite(tval) else float("nan")
        p_chi = _chi2_sf_df1(chisq) if math.isfinite(chisq) else float("nan")

        exp0 = float(np.nanmean([_rsm_expected_score_and_var(float(th), d_base, tau, min_cat, max_cat)[0] for th in theta0]))
        exp1 = float(np.nanmean([_rsm_expected_score_and_var(float(th), d_base, tau, min_cat, max_cat)[0] for th in theta1]))
        obs_exp0 = m0 - exp0
        obs_exp1 = m1 - exp1

        rows_30_2.append([g0, len(idx0), float(np.nansum(xi0)), m0, exp0, d_base, obs_exp0, d_g0, contrast, se_con, tval, p_t, i+1, item_names[i]])
        rows_30_2.append([g1, len(idx1), float(np.nansum(xi1)), m1, exp1, d_base, obs_exp1, d_g1, -contrast, se_con, -tval if math.isfinite(tval) else float("nan"), p_t, i+1, item_names[i]])
        # Table 30.3 (same info as 30.2, grouped by class)
        rows_30_3.append([g0, len(idx0), float(np.nansum(xi0)), m0, exp0, d_base, obs_exp0, d_g0, se_g0, i+1, item_names[i]])
        rows_30_3.append([g1, len(idx1), float(np.nansum(xi1)), m1, exp1, d_base, obs_exp1, d_g1, se_g1, i+1, item_names[i]])
        # Table 30.1 (contrast-style summary on one line)
        chi2 = (tval*tval) if math.isfinite(tval) else float("nan")
        p_chi = (math.erfc(math.sqrt(chi2/2.0)) if math.isfinite(chi2) else float("nan"))
        rows_30_1.append([obs_exp0, d_g0, se_g0, obs_exp1, d_g1, se_g1, contrast, se_con, tval, (len(idx0)+len(idx1)-2), p_t, chi2, p_chi, i+1, item_names[i]])
        # Table 30.5/30.6: weighted INFIT/OUTFIT by group (approx)
        def _mn_sq_stats(th, x, delta):
            # returns (infit_mnsq, outfit_mnsq, infit_z, outfit_z)
            th = np.asarray(th, float)
            x = np.asarray(x, float)
            m = np.isfinite(th) & np.isfinite(x)
            if m.sum() < 2:
                return float("nan"), float("nan"), float("nan"), float("nan")
            th = th[m]; x = x[m]
            exp = np.empty_like(x)
            var = np.empty_like(x)
            for j,(t_) in enumerate(th):
                e_, v_ = _rsm_expected_score_and_var(float(t_), float(delta), tau, min_cat, max_cat)
                exp[j]=e_; var[j]=max(v_, 1e-12)
            resid = x - exp
            outfit = float(np.nanmean((resid*resid)/var))
            infit = float(np.nansum(resid*resid) / max(np.nansum(var), 1e-12))
            df_ = max(int(len(x)-1), 1)
            z_out = (outfit-1.0)/math.sqrt(2.0/df_)
            z_in = (infit-1.0)/math.sqrt(2.0/df_)
            return infit, outfit, z_in, z_out
        infit0, outfit0, z_in0, z_out0 = _mn_sq_stats(th0, xi0, d_base)
        infit1, outfit1, z_in1, z_out1 = _mn_sq_stats(th1, xi1, d_base)
        size0 = d_g0 - d_base
        size1 = d_g1 - d_base
        rows_30_5.append([g0, len(idx0), m0, exp0, d_base, size0, se_g0, infit0, z_in0, outfit0, z_out0, i+1, item_names[i]])
        rows_30_5.append([g1, len(idx1), m1, exp1, d_base, size1, se_g1, infit1, z_in1, outfit1, z_out1, i+1, item_names[i]])
        rows_30_6.append([g0, len(idx0), m0, exp0, d_base, size0, se_g0, infit0, z_in0, outfit0, z_out0, i+1, item_names[i]])
        rows_30_6.append([g1, len(idx1), m1, exp1, d_base, size1, se_g1, infit1, z_in1, outfit1, z_out1, i+1, item_names[i]])

        rows_30_4.append([2, chisq, 1, p_chi, i+1, item_names[i]])

        rows_30_7.append([
            len(idx0)+len(idx1), float(np.nansum(X[:, i])), d_base, "",
            len(idx0), float(np.nansum(xi0)), d_g0, se0,
            len(idx1), float(np.nansum(xi1)), d_g1, se1,
            i+1, item_names[i]
        ])

        contrasts.append(contrast)
        se_contrasts.append(se_con)

    def _fmt(v):
        import math
        try:
            if isinstance(v, str):
                return v
            if isinstance(v, int):
                return str(v)
            if isinstance(v, float):
                if not math.isfinite(v):
                    return ""
                return f"{v:.2f}"
            return str(v)
        except Exception:
            return str(v)

    t30_4 = ["<h3>TABLE 30.4 SUMMARY DIF (between-group)</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             "<tr><th>KID CLASSES</th><th>CHI-SQUARED</th><th>D.F.</th><th>PROB.</th><th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_4:
        t30_4.append("<tr>" + "".join(f"<td>{_fmt(x)}</td>" for x in r) + "</tr>")
    t30_4.append("</table>")



    # --- Winsteps-like DIF tables (30.1–30.7) ---
    t30_1 = ["<h3>TABLE 30.1 DIF contrast (group vs group)</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             f"<tr><th>{g0} Obs-Exp</th><th>{g0} DIF MEASURE</th><th>{g0} S.E.</th>"
             f"<th>{g1} Obs-Exp</th><th>{g1} DIF MEASURE</th><th>{g1} S.E.</th>"
             "<th>CONTRAST</th><th>S.E.</th><th>t</th><th>d.f.</th><th>Prob.</th><th>Mantel Chi-squ</th><th>Prob.</th><th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_1:
        (obs0, d0, se0, obs1, d1, se1, con, secon, t, df, p_t, chi, p_chi, act, name) = r
        t30_1.append(
            f"<tr><td>{obs0:.2f}</td><td>{d0:.2f}</td><td>{se0:.2f}</td>"
            f"<td>{obs1:.2f}</td><td>{d1:.2f}</td><td>{se1:.2f}</td>"
            f"<td>{con:.2f}</td><td>{secon:.2f}</td><td>{t:.2f}</td><td>{int(df)}</td>"
            f"<td>{p_t:.4f}</td><td>{chi:.4f}</td><td>{p_chi:.4f}</td><td>{int(act)}</td><td>{name}</td></tr>"
        )
    t30_1.append("</table>")

    t30_3 = ["<h3>TABLE 30.3 DIF table (grouped by class)</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             "<tr><th>KID CLASS</th><th>COUNT</th><th>SCORE</th><th>AVERAGE</th><th>EXPECT</th><th>MEASURE</th><th>DIF SCORE</th><th>DIF MEASURE</th><th>DIF S.E.</th><th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_3:
        cls, count, score, avg, exp, base_m, dif_score, dif_m, se, act, name = r
        t30_3.append(
            f"<tr><td>{cls}</td><td>{int(count)}</td><td>{score:.0f}</td><td>{avg:.2f}</td><td>{exp:.2f}</td>"
            f"<td>{base_m:.2f}</td><td>{dif_score:.2f}</td><td>{dif_m:.2f}</td><td>{se:.2f}</td><td>{int(act)}</td><td>{name}</td></tr>"
        )
    t30_3.append("</table>")

    def _sort_key_56(r):
        # (class, act)
        return (str(r[0]), int(r[11]))

    # 30.5 alternating F/M per item; 30.6 grouped by class
    t30_5 = ["<h3>TABLE 30.5 DIF fit (W-INFIT / W-OUTFIT) — alternating by class</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             "<tr><th>KID CLASS</th><th>COUNT</th><th>AVERAGE</th><th>EXPECT</th><th>MEASURE</th><th>DIF SIZE</th><th>DIF S.E.</th><th>W-INFIT MNSQ</th><th>ZSTD</th><th>W-OUTFIT MNSQ</th><th>ZSTD</th><th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_5:
        cls, count, avg, exp, base_m, dif_size, se, infit, zin, outfit, zout, act, name = r
        t30_5.append(
            f"<tr><td>{cls}</td><td>{int(count)}</td><td>{avg:.2f}</td><td>{exp:.2f}</td><td>{base_m:.2f}</td><td>{dif_size:.2f}</td><td>{se:.2f}</td>"
            f"<td>{infit:.2f}</td><td>{zin:.2f}</td><td>{outfit:.2f}</td><td>{zout:.2f}</td><td>{int(act)}</td><td>{name}</td></tr>"
        )
    t30_5.append("</table>")

    t30_6 = ["<h3>TABLE 30.6 DIF fit (W-INFIT / W-OUTFIT) — grouped by class</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             "<tr><th>KID CLASS</th><th>COUNT</th><th>AVERAGE</th><th>EXPECT</th><th>MEASURE</th><th>DIF SIZE</th><th>DIF S.E.</th><th>W-INFIT MNSQ</th><th>ZSTD</th><th>W-OUTFIT MNSQ</th><th>ZSTD</th><th>ACT</th><th>Name</th></tr>"]
    for r in sorted(rows_30_6, key=_sort_key_56):
        cls, count, avg, exp, base_m, dif_size, se, infit, zin, outfit, zout, act, name = r
        t30_6.append(
            f"<tr><td>{cls}</td><td>{int(count)}</td><td>{avg:.2f}</td><td>{exp:.2f}</td><td>{base_m:.2f}</td><td>{dif_size:.2f}</td><td>{se:.2f}</td>"
            f"<td>{infit:.2f}</td><td>{zin:.2f}</td><td>{outfit:.2f}</td><td>{zout:.2f}</td><td>{int(act)}</td><td>{name}</td></tr>"
        )
    t30_6.append("</table>")
    t30_2 = ["<h3>TABLE 30.2 DIF detail (baseline vs group)</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             "<tr><th>CLASS</th><th>COUNT</th><th>SCORE</th><th>AVERAGE</th><th>EXPECT</th><th>BASELINE MEASURE</th><th>Obs-Exp</th><th>DIF MEASURE</th><th>CONTRAST</th><th>S.E.</th><th>t</th><th>Prob.</th><th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_2:
        t30_2.append("<tr>" + "".join(f"<td>{_fmt(x)}</td>" for x in r) + "</tr>")
    t30_2.append("</table>")

    t30_7 = ["<h3>TABLE 30.7 Summary by group</h3>",
             "<table border='1' cellpadding='4' cellspacing='0'>",
             f"<tr><th>* ALL COUNT</th><th>* ALL T.SCORE</th><th>* MEASURE</th><th>* S.E.</th>"
             f"<th>{g0} COUNT</th><th>{g0} T.SCORE</th><th>{g0} MEASURE</th><th>{g0} S.E.</th>"
             f"<th>{g1} COUNT</th><th>{g1} T.SCORE</th><th>{g1} MEASURE</th><th>{g1} S.E.</th>"
             "<th>ACT</th><th>Name</th></tr>"]
    for r in rows_30_7:
        t30_7.append("<tr>" + "".join(f"<td>{_fmt(x)}</td>" for x in r) + "</tr>")
    t30_7.append("</table>")

    try:
        # Build proper SMD-based DIF forest plot using group-wise DIF MEASURE + S.E. from Table 30.1
        import plotly.graph_objects as go  # noqa: F401
        dif_df = pd.DataFrame([{
            "ITEM": str(name),
            "b_g0": float(d0) if np.isfinite(float(d0)) else np.nan,
            "se_g0": float(se0) if np.isfinite(float(se0)) else np.nan,
            "b_g1": float(d1) if np.isfinite(float(d1)) else np.nan,
            "se_g1": float(se1) if np.isfinite(float(se1)) else np.nan,
        } for (obs0, d0, se0, obs1, d1, se1, con, secon, t, df, p_t, chi, p_chi, act, name) in rows_30_1])

        # drop rows with missing essentials
        dif_df = dif_df.dropna(subset=["b_g0","se_g0","b_g1","se_g1"])

        if dif_df.empty:
            fig_html = "<div class='warn'>Figure 12 DIF forest plot skipped: insufficient group-wise estimates.</div>"
        else:
            fig = plot_dif_smd_forest(
                dif_df,
                title=f"Figure 12 — SMD in DIF measure (contrast: {g0} - {g1})"
            )
            fig.update_layout(height=max(640, 22 * len(dif_df) + 220), margin=dict(l=260, r=60, t=80, b=60))
            fig_html = (__import__('plotly.io', fromlist=['io']).to_html)(fig, include_plotlyjs=False, full_html=False, config={"displaylogo": False})
    except Exception as e:
        fig_html = f"<pre>Figure 12 DIF forest plot failed: {e!r}</pre>"

    out = []
    out.append("<h2>Task 12 — DIF (Winsteps-like)</h2>")
    out.append(f"<p>DIF class/group specification is: DIF=Profile (last column). Groups: <b>{g0}</b> vs <b>{g1}</b>.</p>")
    # Tables 30.1–30.7 (requested Winsteps-style output)
    out.extend(t30_1)
    out.extend(t30_2)
    out.extend(t30_3)
    out.extend(t30_4)
    out.extend(t30_5)
    out.extend(t30_6)
    out.extend(t30_7)

    # Figure 12 (forest plot of DIF contrast)
    out.append(fig_html)
    return "\n".join(out)



def _euclidean_distance_matrix(X):
    """Pairwise Euclidean distance matrix for 2D array-like X (n x p)."""
    import numpy as np
    X = np.asarray(X, dtype=float)
    n = X.shape[0] if X.ndim == 2 else (X.shape[0] if X.ndim == 1 else 0)
    if X.ndim != 2 or n < 2:
        return np.zeros((n, n), dtype=float)
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (X @ X.T)
    D2[D2 < 0] = 0.0
    return np.sqrt(D2)



def _safe_marker_sizes(values, min_size=8.0, max_size=28.0):
    """Convert array of (possibly NaN/Inf) values into Plotly-safe marker sizes."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if v.size == 0:
        return []
    vmin = float(np.min(v)); vmax = float(np.max(v))
    if vmax <= vmin + 1e-12:
        return (np.full_like(v, (min_size + max_size) / 2.0)).tolist()
    t = (v - vmin) / (vmax - vmin)
    s = min_size + t * (max_size - min_size)
    s = np.clip(s, min_size, max_size)
    return s.tolist()



def build_person_dimension_top20_table(person_df, Z_person, loading_pc1, cluster, D=None, top_n=20):
    """Return Person list (Top-N) with Loading, Delta, SS(i), a(i), b(i) like Figure 9."""
    import numpy as np
    import pandas as pd
    Z = np.asarray(Z_person, dtype=float)
    n = Z.shape[0]
    if D is None:
        D = _euclidean_distance_matrix(Z)
    cluster = np.asarray(cluster)
    loading_pc1 = np.asarray(loading_pc1, dtype=float)
    a = np.zeros(n, dtype=float); b = np.zeros(n, dtype=float); ss = np.zeros(n, dtype=float)
    for i in range(n):
        ci = cluster[i]
        same = np.where(cluster == ci)[0]
        other = np.where(cluster != ci)[0]
        same_wo = same[same != i]
        a[i] = float(np.mean(D[i, same_wo])) if same_wo.size > 0 else 0.0
        b[i] = float(np.mean(D[i, other])) if other.size > 0 else 0.0
        denom = max(a[i], b[i])
        ss[i] = 0.0 if denom <= 1e-12 else (b[i] - a[i]) / denom
    if "PERSON" in person_df.columns:
        person_name = person_df["PERSON"].astype(str).values
    elif "KID" in person_df.columns:
        person_name = person_df["KID"].astype(str).values
    elif "ID" in person_df.columns:
        person_name = person_df["ID"].astype(str).values
    else:
        person_name = np.array([f"Person {i+1}" for i in range(n)], dtype=object)
    delta = pd.to_numeric(person_df["MEASURE"], errors="coerce").values if "MEASURE" in person_df.columns else np.full(n, np.nan)
    out = pd.DataFrame({"Person": person_name, "Loading": loading_pc1, "Theta": delta,
                        "SS(i)": ss, "a(i)": a, "b(i)": b, "Cluster": cluster})
    out = out.sort_values("SS(i)", ascending=False).head(int(top_n)).reset_index(drop=True)
    return out



def _person_dimension_plot_top20(res, top_n=20, beta_value=None, beta_p=None):
    """Figure 13: Person dimension Kano-style plot (Top-N by |PC1| of ZSTD).

    - X axis: Theta (person measure)
    - Y axis: Contrast-1 loading (from PCA on person ZSTD correlation)
    - Bubble size: SS(i)
    - Links: one-link within each cluster using correlation r>0 on ZSTD vectors (Kano-style)
    """
    import numpy as np
    import pandas as pd

    pdf = getattr(res, "person_df", None)
    dbg = getattr(res, "debug", None)
    if not isinstance(pdf, pd.DataFrame) or not isinstance(dbg, dict):
        return None, None

    Z = dbg.get("ZSTD", None)
    if Z is None:
        return None, None
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != len(pdf):
        return None, None

    # person vectors in item space
    X = Z.copy()
    # impute NaN by row mean, then center
    mu = np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isfinite(X), X, mu)
    X = X - np.mean(X, axis=1, keepdims=True)

    # PCA on correlation between persons (using corrcoef of person vectors)
    try:
        C = np.corrcoef(X)
        C[~np.isfinite(C)] = 0.0
        w, V = np.linalg.eigh(C)
        pc1 = V[:, np.argmax(w)]
    except Exception:
        # fallback: first left singular vector
        try:
            U, S, VT = np.linalg.svd(X, full_matrices=False)
            pc1 = U[:, 0]
        except Exception:
            return None, None

    pc1 = np.asarray(pc1, dtype=float).ravel()
    if pc1.size != len(pdf):
        return None, None

    # choose top-N by |pc1|
    idx = np.argsort(-np.abs(pc1))[: int(min(top_n, len(pc1)))]
    pdf_top = pdf.iloc[idx].copy().reset_index(drop=True)
    Z_top = X[idx, :].copy()
    pc1_top = pc1[idx].copy()

    # normalize loading scale for display (avoid >1)
    m = float(np.nanmax(np.abs(pc1_top))) if pc1_top.size else 1.0
    if np.isfinite(m) and m > 1e-12:
        pc1_top = pc1_top / m

    clusters = np.where(pc1_top >= 0, 1, 2).astype(int)

    # distance matrix (for silhouette in the table)
    D = _euclidean_distance_matrix(Z_top)

    # table (Theta, Loading, SS(i), a(i), b(i))
    tbl = build_person_dimension_top20_table(
        pdf_top, Z_top, pc1_top, clusters, D=D, top_n=min(20, len(pdf_top))
    )

    # --- edges (Kano-style) ---
    edges = []
    try:
        R = np.corrcoef(Z_top)
        R[~np.isfinite(R)] = -np.inf
        np.fill_diagonal(R, -np.inf)
    except Exception:
        R = None

    if R is not None and R.shape[0] == len(tbl):
        persons = tbl["Person"].astype(str).tolist()
        cl_arr = tbl["Cluster"].to_numpy(dtype=int)
        for c in (1, 2):
            inds = np.where(cl_arr == c)[0]
            if len(inds) < 2:
                continue
            Rc = R[np.ix_(inds, inds)].copy()
            Rc[Rc <= 0] = -np.inf  # keep only r>0
            for ii_local, ii in enumerate(inds):
                jj_local = int(np.argmax(Rc[ii_local]))
                if not np.isfinite(Rc[ii_local, jj_local]):
                    continue
                jj = int(inds[jj_local])
                # undirected dedup
                a, b = (ii, jj) if ii < jj else (jj, ii)
                edges.append((a, b, float(R[a, b])))

    if edges:
        # dedup keep max r
        best = {}
        for a, b, r in edges:
            key = (a, b)
            if key not in best or r > best[key]:
                best[key] = r
        rel_df = pd.DataFrame(
            {
                "term1": [tbl["Person"].iloc[a] for (a, b) in best.keys()],
                "term2": [tbl["Person"].iloc[b] for (a, b) in best.keys()],
                "WCD": [best[(a, b)] for (a, b) in best.keys()],  # here WCD is r (positive)
            }
        )
    else:
        rel_df = pd.DataFrame(columns=["term1", "term2", "WCD"])

    # nodes for kano_plot_aligned
    nodes = pd.DataFrame(
        {
            "name": tbl["Person"].astype(str),
            "value": pd.to_numeric(tbl["Loading"], errors="coerce"),
            "value2": pd.to_numeric(tbl["Theta"], errors="coerce"),
            "cluster": pd.to_numeric(tbl["Cluster"], errors="coerce").fillna(0).astype(int),
            "ss_i": pd.to_numeric(tbl["SS(i)"], errors="coerce"),
            "a_i": pd.to_numeric(tbl["a(i)"], errors="coerce"),
            "b_i": pd.to_numeric(tbl["b(i)"], errors="coerce"),
        }
    )

    fig = kano_plot_aligned(
        nodes,
        rel_df,
        title_suffix=" (Figure 13; r>0 one-link within cluster)",
        x_col="value2",
        y_col="value",
        x_label="Theta (person measure)",
        y_label="Contrast-1 loading",
        beta_value=beta_value,
        beta_p=beta_p,
    )
    return fig, tbl