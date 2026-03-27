"""
=============================================================
  CRPA ANTENNA — Advanced Null Steering Dashboard
  Controlled Reception Pattern Antenna (N+1 Circular Array)
  Projection / LCMV-style constrained beamforming + diagnostics
=============================================================
"""

import math
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

# -----------------------------------------------------------------------------
# APP CONFIG + THEME
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CRPA Null Steering • Advanced", page_icon="📡", layout="wide")

st.markdown(
    """
<style>
  .block-container { padding-top: 0.9rem; }
  .metric-box {
    background:#1a1d2e;
    border-radius:10px;
    padding:10px 12px;
    text-align:center;
    margin-bottom:6px;
    border:1px solid #2a3150;
  }
  .metric-val { font-size:1.2rem; font-weight:700; color:#00d4ff; }
  .metric-lbl { font-size:0.72rem; color:#adb5bd; margin-top:2px; }
  .small-note { font-size:0.82rem; color:#adb5bd; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📡 CRPA Antenna — Advanced Null Steering Dashboard")
st.caption(
    "Controlled Reception Pattern Antenna · Circular N+1 array · "
    "Projection-matrix null steering with stability diagnostics"
)

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
C = 299_792_458.0
DARK_BG = "#0f1117"
PANEL_BG = "#1a1d2e"
GRID_COL = "#2e3150"
NULL_COLS = ["#e74c3c", "#e67e22", "#9b59b6", "#1abc9c"]
DES_COL = "#2ecc71"
PAT_COL = "#00d4ff"


# -----------------------------------------------------------------------------
# DATA MODELS
# -----------------------------------------------------------------------------
@dataclass
class Scenario:
    name: str
    freq_mhz: float
    n_outer: int
    spacing_frac: float
    steer_az: float
    steer_el: float
    patch_eff_pct: int
    dyn_range_db: int
    nulls: List[Tuple[float, float]]
    quality_mode: str
    show_3d: bool

SCENARIOS: Dict[str, Scenario] = {
    "Custom": Scenario(
        name="Custom",
        freq_mhz=1575.42,
        n_outer=7,
        spacing_frac=0.50,
        steer_az=0.0,
        steer_el=90.0,
        patch_eff_pct=80,
        dyn_range_db=40,
        nulls=[(-60, 30), (120, 60)],
        quality_mode="Normal",
        show_3d=False,
    ),
    "GPS L1 (2 jammer)": Scenario(
        name="GPS L1 (2 jammer)",
        freq_mhz=1575.42,
        n_outer=7,
        spacing_frac=0.50,
        steer_az=15.0,
        steer_el=70.0,
        patch_eff_pct=82,
        dyn_range_db=45,
        nulls=[(-55, 25), (130, 50)],
        quality_mode="Normal",
        show_3d=False,
    ),
    "Stress test (4 jammer)": Scenario(
        name="Stress test (4 jammer)",
        freq_mhz=1575.42,
        n_outer=8,
        spacing_frac=0.45,
        steer_az=0.0,
        steer_el=75.0,
        patch_eff_pct=80,
        dyn_range_db=50,
        nulls=[(-70, 25), (40, 35), (130, 45), (-145, 60)],
        quality_mode="Fast",
        show_3d=False,
    ),
}

QUALITY_GRIDS = {
    "Fast": {"az_pts": 361, "el_pts": 181, "n_az_3d": 61, "n_el_3d": 31},
    "Normal": {"az_pts": 721, "el_pts": 361, "n_az_3d": 91, "n_el_3d": 46},
    "High": {"az_pts": 1441, "el_pts": 721, "n_az_3d": 121, "n_el_3d": 61},
}


# -----------------------------------------------------------------------------
# PLOT STYLE HELPERS
# -----------------------------------------------------------------------------
def _ax_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color=GRID_COL, linestyle="--", alpha=0.5)

def _fig_dark(fig):
    fig.patch.set_facecolor(DARK_BG)


# -----------------------------------------------------------------------------
# NUMERICS (PURE FUNCTIONS)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_array(n_outer: int, spacing_frac: float, freq_mhz: float):
    wl = C / (freq_mhz * 1e6)
    radius = spacing_frac * wl
    ang = 2 * np.pi * np.arange(n_outer) / n_outer
    ring = np.zeros((n_outer, 3))
    ring[:, 0] = radius * np.cos(ang)
    ring[:, 1] = radius * np.sin(ang)
    pos = np.vstack([np.zeros((1, 3)), ring])
    return tuple(map(tuple, pos)), wl, radius

@st.cache_data(show_spinner=False)
def steering_vec(pos_t, az_deg: float, el_deg: float, wl: float):
    pos = np.array(pos_t)
    az, el = np.radians(az_deg), np.radians(el_deg)
    k = np.array([np.cos(el) * np.sin(az), np.cos(el) * np.cos(az), np.sin(el)])
    return np.exp(1j * 2 * np.pi / wl * (pos @ k))

@st.cache_data(show_spinner=False)
def compute_weights_projection(
    pos_t,
    wl: float,
    steer_az: float,
    steer_el: float,
    null_t,
    loading_eps: float,
):
    """Projection-matrix nulling with adaptive diagonal loading diagnostics."""
    n = len(pos_t)
    a_d = steering_vec(pos_t, steer_az, steer_el, wl)

    if len(null_t) == 0:
        w = a_d / max(np.linalg.norm(a_d), 1e-12)
        return tuple(w.real), tuple(w.imag), 1.0, 0.0, 0.0

    A_n = np.column_stack([steering_vec(pos_t, az, el, wl) for az, el in null_t])
    AHA = A_n.conj().T @ A_n
    cond_num = float(np.linalg.cond(AHA)) if AHA.size else 1.0

    # Adaptive loading when ill-conditioned
    adaptive_eps = loading_eps
    if cond_num > 1e8:
        adaptive_eps = max(loading_eps, 1e-3)
    elif cond_num > 1e6:
        adaptive_eps = max(loading_eps, 1e-4)

    try:
        inv_term = np.linalg.solve(AHA + adaptive_eps * np.eye(AHA.shape[0]), A_n.conj().T)
        P = np.eye(n) - A_n @ inv_term
    except np.linalg.LinAlgError:
        P = np.eye(n)

    w = P @ a_d
    nm = np.linalg.norm(w)
    if nm < 1e-12:
        w = a_d / max(np.linalg.norm(a_d), 1e-12)
    else:
        w = w / nm

    # Constraint residual: ||A^H w||_2
    residual = float(np.linalg.norm(A_n.conj().T @ w)) if len(null_t) else 0.0

    return tuple(w.real), tuple(w.imag), cond_num, adaptive_eps, residual

@st.cache_data(show_spinner=False)
def compute_az_pattern(pos_t, wr, wi, wl: float, eff: float, n_pts: int):
    w = np.array(wr) + 1j * np.array(wi)
    azs = np.linspace(-180, 180, n_pts)
    out = np.zeros(n_pts)
    for i, az in enumerate(azs):
        a = steering_vec(pos_t, az, 90.0, wl)
        out[i] = (np.abs(np.dot(w.conj(), a)) * np.sqrt(eff)) ** 2
    return tuple(azs), tuple(out)

@st.cache_data(show_spinner=False)
def compute_el_pattern(pos_t, wr, wi, wl: float, az_deg: float, eff: float, n_pts: int):
    w = np.array(wr) + 1j * np.array(wi)
    els = np.linspace(0, 90, n_pts)
    out = np.zeros(n_pts)
    for i, el in enumerate(els):
        a = steering_vec(pos_t, az_deg, el, wl)
        el_r = np.radians(90 - el)
        elem_pat = max(np.cos(el_r), 0.0) ** 1.5
        out[i] = (np.abs(np.dot(w.conj(), a)) * elem_pat * np.sqrt(eff)) ** 2
    return tuple(els), tuple(out)

@st.cache_data(show_spinner=False)
def compute_3d_pattern(pos_t, wr, wi, wl: float, eff: float, n_az: int, n_el: int):
    w = np.array(wr) + 1j * np.array(wi)
    az_v = np.linspace(-180, 180, n_az)
    el_v = np.linspace(0, 90, n_el)
    pwr = np.zeros((n_el, n_az))

    for ei, el in enumerate(el_v):
        el_r = np.radians(90 - el)
        elem_pat = max(np.cos(el_r), 0.0) ** 1.5
        for ai, az in enumerate(az_v):
            a = steering_vec(pos_t, az, el, wl)
            pwr[ei, ai] = (np.abs(np.dot(w.conj(), a)) * elem_pat * np.sqrt(eff)) ** 2

    return tuple(az_v), tuple(el_v), tuple(map(tuple, pwr))

def to_db(arr, ref, floor_db):
    arr_np = np.array(arr)
    return 10 * np.log10(np.maximum(arr_np / max(ref, 1e-30), 10 ** (floor_db / 10)))

def beamwidth_3db(az_deg, db_vals):
    az = np.array(az_deg)
    db = np.array(db_vals)
    imax = int(np.argmax(db))
    thr = db[imax] - 3.0

    left = imax
    while left > 0 and db[left] >= thr:
        left -= 1
    right = imax
    while right < len(db) - 1 and db[right] >= thr:
        right += 1

    return float(abs(az[right] - az[left])) if right > left else 0.0

def sidelobe_level(db_vals):
    db = np.array(db_vals)
    i0 = int(np.argmax(db))
    guard = max(3, len(db) // 60)
    mask = np.ones_like(db, dtype=bool)
    mask[max(i0 - guard, 0):min(i0 + guard + 1, len(db))] = False
    if np.any(mask):
        return float(np.max(db[mask]))
    return float(np.max(db))

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Scenario")
    preset_name = st.selectbox("Preset", list(SCENARIOS.keys()), index=0)
    preset = SCENARIOS[preset_name]

    st.markdown("---")
    st.header("🔧 Array Parameters")
    freq_mhz = st.number_input("Frequency (MHz)", 100.0, 6000.0, float(preset.freq_mhz), 0.5)
    n_outer = st.number_input("Outer Elements (N)", 2, 16, int(preset.n_outer), 1)
    spacing_frac = st.slider("Element Spacing (× λ)", 0.30, 1.00, float(preset.spacing_frac), 0.01)
    steer_az = st.slider("Desired Azimuth (°)", -180, 180, int(preset.steer_az), 1)
    steer_el = st.slider("Desired Elevation (°)", 0, 90, int(preset.steer_el), 1)
    patch_eff = st.slider("Patch Efficiency (%)", 50, 100, int(preset.patch_eff_pct), 1)
    dyn_range = st.slider("Pattern Dynamic Range (dB)", 20, 70, int(preset.dyn_range_db), 5)

    st.markdown("---")
    st.header("🎯 Null Steering")
    n_nulls = st.selectbox("Number of Nulls", [0, 1, 2, 3, 4], index=min(len(preset.nulls), 4))

    default_nulls = preset.nulls + [(-120, 35), (60, 30), (145, 50), (-40, 20)]
    null_dirs = []
    for i in range(n_nulls):
        st.markdown(f"**Null {i + 1}**")
        c1, c2 = st.columns(2)
        with c1:
            naz = st.slider(f"Az {i + 1} (°)", -180, 180, int(default_nulls[i][0]), 1, key=f"naz{i}")
        with c2:
            nel = st.slider(f"El {i + 1} (°)", 0, 90, int(default_nulls[i][1]), 1, key=f"nel{i}")
        null_dirs.append((float(naz), float(nel)))

    st.markdown("---")
    st.header("🧮 Solver & Rendering")
    quality_mode = st.selectbox("Quality Mode", ["Fast", "Normal", "High"], index=["Fast", "Normal", "High"].index(preset.quality_mode))
    loading_eps = st.number_input("Diagonal Loading ε", 1e-8, 1e-1, 1e-6, format="%.1e")
    show_3d = st.checkbox("Show 3D Pattern (slower)", value=bool(preset.show_3d))

# -----------------------------------------------------------------------------
# HEADER INFO
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📖 Project Overview")
st.markdown(
    "This dashboard models a **Controlled Reception Pattern Antenna (CRPA)** used for GNSS anti-jamming. "
    "It computes complex element weights via projection-matrix null steering and visualizes array response, "
    "null depth, stability metrics, and beam quality."
)

# -----------------------------------------------------------------------------
# COMPUTE PIPELINE
# -----------------------------------------------------------------------------
null_t = tuple(null_dirs)
pos_t, wl, radius = build_array(int(n_outer), float(spacing_frac), float(freq_mhz))
n_total = len(pos_t)

if len(null_t) >= n_total:
    st.error(
        f"Invalid configuration: null count ({len(null_t)}) must be less than total elements ({n_total})."
    )
    st.stop()

if float(spacing_frac) < 0.40:
    st.warning("Spacing < 0.40 λ can increase coupling effects; model may be optimistic.")

eff = patch_eff / 100.0
grids = QUALITY_GRIDS[quality_mode]

wr_t, wi_t, cond_num, eps_used, residual = compute_weights_projection(
    pos_t, wl, float(steer_az), float(steer_el), null_t, float(loading_eps)
)

w = np.array(wr_t) + 1j * np.array(wi_t)
wm = np.abs(w)
wp = np.angle(w, deg=True)
max_w = float(max(np.max(wm), 1e-30))

az_t, p_az_t = compute_az_pattern(pos_t, wr_t, wi_t, wl, eff, grids["az_pts"])
el_t, p_el_t = compute_el_pattern(pos_t, wr_t, wi_t, wl, float(steer_az), eff, grids["el_pts"])

peak = float(max(np.max(p_az_t), 1e-30))
db_az_t = tuple(to_db(p_az_t, peak, -dyn_range).tolist())
db_el_t = tuple(to_db(p_el_t, peak, -dyn_range).tolist())

# Null diagnostics
null_depths = []
null_az_err = []
for naz, nel in null_t:
    a_n = steering_vec(pos_t, naz, nel, wl)
    el_r = np.radians(90 - nel)
    pn = (np.abs(np.dot(w.conj(), a_n)) * max(np.cos(el_r), 0.0) ** 1.5 * np.sqrt(eff)) ** 2
    d = float(to_db(np.array([pn]), peak, -dyn_range)[0])
    null_depths.append(d)

    idx = int(np.argmin(np.abs(np.array(az_t) - naz)))
    null_az_err.append(float(db_az_t[idx] - d))

a_des = steering_vec(pos_t, float(steer_az), float(steer_el), wl)
el_des_r = np.radians(90 - float(steer_el))
p_des = (np.abs(np.dot(w.conj(), a_des)) * max(np.cos(el_des_r), 0.0) ** 1.5 * np.sqrt(eff)) ** 2
gain_des = float(to_db(np.array([p_des]), peak, -dyn_range)[0])

peak_gain_dbi = (
    10 * np.log10(n_total)
    + 10 * np.log10(max(0.01 * patch_eff * 4 * math.pi * 0.25, 1e-10))
    + 5.0
)

bw3 = beamwidth_3db(az_t, db_az_t)
sll = sidelobe_level(db_az_t)

# -----------------------------------------------------------------------------
# TOP METRICS
# -----------------------------------------------------------------------------
st.markdown("---")
metric_items = [
    ("Frequency", f"{freq_mhz:.2f} MHz"),
    ("Wavelength", f"{wl * 100:.2f} cm"),
    ("Total Elements", str(n_total)),
    ("Array Diameter", f"{2 * radius * 100:.2f} cm"),
    ("Est. Peak Gain", f"{peak_gain_dbi:.1f} dBi"),
    ("Gain @ Desired", f"{gain_des:.1f} dB rel."),
    ("3 dB BW", f"{bw3:.1f}°"),
    ("Max Sidelobe", f"{sll:.1f} dB"),
]

for row_start in [0, 4]:
    cols = st.columns(4)
    for c, (lbl, val) in zip(cols, metric_items[row_start:row_start + 4]):
        c.markdown(
            f'<div class="metric-box"><div class="metric-val">{val}</div><div class="metric-lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

# Solver health
colh1, colh2, colh3 = st.columns(3)
health = "✅ Stable"
if cond_num > 1e8:
    health = "❌ Ill-conditioned"
elif cond_num > 1e6:
    health = "⚠️ Marginal"

colh1.info(f"**Constraint Matrix Cond. #**: `{{cond_num:.2e}}`")
colh2.info(f"**Loading ε Used**: `{{eps_used:.1e}}`")
colh3.info(f"**Null Residual ‖Aᴴw‖₂**: `{{residual:.2e}}` · {{health}}")

if cond_num > 1e8:
    st.warning(
        "Constraint matrix is highly ill-conditioned. Consider fewer nulls, larger spacing, or higher loading ε."
    )

# -----------------------------------------------------------------------------
# FIGURE BUILDERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fig_polar(az_t, db_t, steer_az, null_t, depths_t, dyn):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.4, 5.4))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)
    th = np.radians(np.array(az_t))
    rr = np.clip(np.array(db_t) + dyn, 0, None)
    ax.plot(th, rr, color=PAT_COL, lw=1.8)
    ax.fill(th, rr, color=PAT_COL, alpha=0.14)
    ax.axvline(np.radians(steer_az), color=DES_COL, lw=2, ls="--")
    for i, ((naz, _), _) in enumerate(zip(null_t, depths_t)):
        ax.axvline(np.radians(naz), color=NULL_COLS[i % 4], lw=1.8, ls=":")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ticks = np.linspace(0, dyn, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{{int(t - dyn)}} dB" for t in ticks], color="white", fontsize=7)
    ax.tick_params(colors="white")
    ax.grid(color=GRID_COL, alpha=0.5)
    ax.set_title("Azimuth Cut (El = 90°)", color="white", pad=12, fontsize=11)
    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def fig_cart(az_t, db_t, steer_az, null_t, depths_t, dyn):
    fig, ax = plt.subplots(figsize=(5.9, 4.4))
    _fig_dark(fig)
    _ax_dark(ax)

    az = np.array(az_t)
    db = np.array(db_t)
    ax.plot(az, db, color=PAT_COL, lw=1.8, label="Pattern")
    ax.axvline(steer_az, color=DES_COL, lw=2, ls="--", label=f"Desired {{steer_az:.0f}}°")

    for i, ((naz, _), d) in enumerate(zip(null_t, depths_t)):
        c = NULL_COLS[i % 4]
        ax.axvline(naz, color=c, lw=1.8, ls=":", label=f"N{{i+1}} {{naz:.0f}}° ({d:.1f} dB)")

    ax.set_ylim([-dyn - 2, 3])
    ax.set_xlim([-180, 180])
    ax.set_xlabel("Azimuth (°)", color="white", fontsize=9)
    ax.set_ylabel("Relative Gain (dB)", color="white", fontsize=9)
    ax.set_title("Azimuth Pattern (Cartesian)", color="white", fontsize=11)
    ax.legend(fontsize=7.3, facecolor=PANEL_BG, labelcolor="white", framealpha=0.85)
    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def fig_el(el_t, db_t, steer_az, steer_el, null_t, dyn):
    fig, ax = plt.subplots(figsize=(11.6, 3.8))
    _fig_dark(fig)
    _ax_dark(ax)
    ax.plot(np.array(el_t), np.array(db_t), color="#f39c12", lw=1.8, label="Pattern")
    ax.axvline(steer_el, color=DES_COL, lw=2, ls="--", label=f"Desired El={{steer_el:.0f}}°")

    for i, (naz, nel) in enumerate(null_t):
        if abs(naz - steer_az) < 20:
            ax.axvline(nel, color=NULL_COLS[i % 4], lw=1.8, ls=":", label=f"N{{i+1}} El={{nel:.0f}}°")

    ax.set_ylim([-dyn - 2, 3])
    ax.set_xlim([0, 90])
    ax.set_xlabel("Elevation (°)", color="white", fontsize=9)
    ax.set_ylabel("Relative Gain (dB)", color="white", fontsize=9)
    ax.set_title(f"Elevation Pattern (Az = {{steer_az:.0f}}°)", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white", framealpha=0.85)
    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def fig_layout(pos_t, wl, steer_az, null_t, wm_t, wp_t, max_w):
    pos = np.array(pos_t)
    pos_cm = pos * 100
    patch = 0.5 * (wl * 100) * 0.85

    fig, ax = plt.subplots(figsize=(5.3, 5.3))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)
    ax.set_aspect("equal")

    cmap = plt.cm.plasma
    for i, (x, y, _) in enumerate(pos_cm):
        col = cmap(wm_t[i] / max_w)
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (x - patch / 2, y - patch / 2), patch, patch,
                boxstyle="round,pad=0.05", lw=1.2, edgecolor="white", facecolor=col, alpha=0.9
            )
        )
        ax.text(x, y, f"{{wp_t[i]:.0f}}°", ha="center", va="center", color="white", fontsize=6, fontweight="bold")

    r = max(np.max(np.abs(pos_cm[:, :2])), 2) * 2.4

    dr = np.radians(steer_az)
    ax.annotate("", xy=(r * np.sin(dr), r * np.cos(dr)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=DES_COL, lw=2.1, mutation_scale=14))
    ax.text(r * 1.12 * np.sin(dr), r * 1.12 * np.cos(dr), "Des.", color=DES_COL, fontsize=8, fontweight="bold")

    for i, (naz, _) in enumerate(null_t):
        c = NULL_COLS[i % 4]
        nr = np.radians(naz)
        ax.annotate("", xy=(r * np.sin(nr), r * np.cos(nr)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.7, mutation_scale=12))
        ax.text(r * 1.12 * np.sin(nr), r * 1.12 * np.cos(nr), f"N{{i+1}}", color=c, fontsize=8, fontweight="bold")

    lim = r * 1.35
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xlabel("x (cm)", color="white", fontsize=9)
    ax.set_ylabel("y (cm)", color="white", fontsize=9)
    ax.set_title("Element Layout (color=|w|, text=phase°)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color=GRID_COL, alpha=0.3, ls="--")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04)
    cb.set_label("Norm. |w|", color="white", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def fig_phasors(wr_t, wi_t, max_w, n_total):
    w = np.array(wr_t) + 1j * np.array(wi_t)
    mags = np.abs(w)
    phs = np.angle(w, deg=True)

    fig, ax = plt.subplots(figsize=(5.3, 5.3))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)

    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color="#444", lw=1)
    ax.axhline(0, color="#444", lw=0.5)
    ax.axvline(0, color="#444", lw=0.5)

    colors_e = plt.cm.cool(np.linspace(0, 1, n_total))
    for i, wi in enumerate(w):
        wn = wi / max_w
        ax.quiver(0, 0, wn.real, wn.imag, angles="xy", scale_units="xy", scale=1,
                  color=colors_e[i], alpha=0.9, width=0.012)
        ax.scatter(wn.real, wn.imag, color=colors_e[i], s=52, zorder=5)
        ax.text(wn.real * 1.09, wn.imag * 1.09, str(i), color=colors_e[i], fontsize=8, ha="center")

    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_aspect("equal")
    ax.set_xlabel("Real", color="white", fontsize=9)
    ax.set_ylabel("Imaginary", color="white", fontsize=9)
    ax.set_title("Weight Phasors (normalized)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color=GRID_COL, alpha=0.3, ls="--")

    if n_total <= 12:
        patches = [
            mpatches.Patch(color=plt.cm.cool(i / max(n_total - 1, 1)),
                           label=f"E{{i}}: |w|={{mags[i]/max_w:.2f}} ∠{{phs[i]:.0f}}°")
            for i in range(n_total)
        ]
        ax.legend(handles=patches, fontsize=6.3, facecolor=PANEL_BG, labelcolor="white", framealpha=0.85, loc="lower right")

    fig.tight_layout()
    return fig

@st.cache_data(show_spinner=False)
def fig_3d(pos_t, wr_t, wi_t, wl, eff, dyn, n_az, n_el):
    az_t, el_t, pwr_t = compute_3d_pattern(pos_t, wr_t, wi_t, wl, eff, n_az, n_el)
    pwr = np.array(pwr_t)
    peak_local = max(np.max(pwr), 1e-30)
    db = to_db(pwr, peak_local, -dyn)
    R = np.clip(db + dyn, 0, None) / dyn

    az_v = np.array(az_t)
    el_v = np.array(el_t)
    AZ_r, EL_r = np.radians(np.meshgrid(az_v, el_v))

    X = R * np.cos(EL_r) * np.sin(AZ_r)
    Y = R * np.cos(EL_r) * np.cos(AZ_r)
    Z = R * np.sin(EL_r)

    fig = plt.figure(figsize=(7.7, 5.6))
    _fig_dark(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(R), alpha=0.85, linewidth=0, antialiased=True)
    ax.set_title("3D Pattern (upper hemisphere)", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=7)

    for p in [ax.xaxis, ax.yaxis, ax.zaxis]:
        p.pane.fill = False
        p.pane.set_edgecolor("#333")

    fig.tight_layout()
    return fig

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
st.markdown("---")
depths_t = tuple(null_depths)
wm_t = tuple(wm.tolist())
wp_t = tuple(wp.tolist())

c1, c2 = st.columns(2)
with c1:
    st.subheader("Azimuth Pattern — Polar")
    f1 = fig_polar(az_t, db_az_t, float(steer_az), null_t, depths_t, int(dyn_range))
    st.pyplot(f1, use_container_width=True)

with c2:
    st.subheader("Azimuth Pattern — Cartesian")
    f2 = fig_cart(az_t, db_az_t, float(steer_az), null_t, depths_t, int(dyn_range))
    st.pyplot(f2, use_container_width=True)

st.subheader(f"Elevation Pattern (Az = {{steer_az:.0f}}°)")
f3 = fig_el(el_t, db_el_t, float(steer_az), float(steer_el), null_t, int(dyn_range))
st.pyplot(f3, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.subheader("Array Element Layout")
    f4 = fig_layout(pos_t, wl, float(steer_az), null_t, wm_t, wp_t, max_w)
    st.pyplot(f4, use_container_width=True)

with c4:
    st.subheader("Weight Phasors")
    f5 = fig_phasors(wr_t, wi_t, max_w, n_total)
    st.pyplot(f5, use_container_width=True)

if show_3d:
    st.subheader("3D Radiation Pattern")
    with st.spinner("Rendering 3D…"):
        f6 = fig_3d(
            pos_t, wr_t, wi_t, wl, eff, int(dyn_range),
            grids["n_az_3d"], grids["n_el_3d"]
        )
        st.pyplot(f6, use_container_width=True)

# -----------------------------------------------------------------------------
# TABLES + DOWNLOADS
# -----------------------------------------------------------------------------
st.markdown("---")
t1, t2 = st.columns(2)

with t1:
    st.subheader("🎯 Null Summary")
    if len(null_t) > 0:
        df_null = pd.DataFrame([
            {
                "Null": i + 1,
                "Az (°)": naz,
                "El (°)": nel,
                "Depth (dB)": f"{{d:.1f}}",
                "Status": "✅ Deep" if d < -20 else ("⚠️ Shallow" if d < -10 else "❌ Weak"),
            }
            for i, ((naz, nel), d) in enumerate(zip(null_t, null_depths))
        ])
        st.dataframe(df_null, use_container_width=True, hide_index=True)
    else:
        st.info("No nulls configured.")

with t2:
    st.subheader("⚖️ Element Weights")
    pos_arr = np.array(pos_t)
    df_w = pd.DataFrame(
        {
            "Elem": [f"E{i}{'*' if i == 0 else ''}" for i in range(n_total)],
            "x (cm)": [f"{{pos_arr[i, 0] * 100:.2f}}" for i in range(n_total)],
            "y (cm)": [f"{{pos_arr[i, 1] * 100:.2f}}" for i in range(n_total)],
            "|w| norm": [f"{{wm_t[i] / max_w:.3f}}" for i in range(n_total)],
            "Phase (°)": [f"{{wp_t[i]:.1f}}" for i in range(n_total)],
            "w_real": [float(np.real(w[i])) for i in range(n_total)],
            "w_imag": [float(np.imag(w[i])) for i in range(n_total)],
        }
    )
    st.dataframe(df_w, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("📦 Export")
e1, e2, e3 = st.columns(3)

weights_csv = df_w.to_csv(index=False).encode("utf-8")
scenario_obj = Scenario(
    name="Exported Scenario",
    freq_mhz=float(freq_mhz),
    n_outer=int(n_outer),
    spacing_frac=float(spacing_frac),
    steer_az=float(steer_az),
    steer_el=float(steer_el),
    patch_eff_pct=int(patch_eff),
    dyn_range_db=int(dyn_range),
    nulls=list(map(tuple, null_t)),
    quality_mode=quality_mode,
    show_3d=bool(show_3d),
)
scenario_json = json.dumps(asdict(scenario_obj), indent=2).encode("utf-8")

with e1:
    st.download_button("Download Weights CSV", data=weights_csv, file_name="crpa_weights.csv", mime="text/csv")
with e2:
    st.download_button("Download Scenario JSON", data=scenario_json, file_name="crpa_scenario.json", mime="application/json")
with e3:
    st.caption("Use CSV for DSP chain integration and JSON for reproducible testing.")

st.markdown("---")
st.caption(
    "**Method:** Projection null steering with diagonal loading. "
    "Weights: **w = P·a_d**,  **P = I − A(AᴴA + εI)⁻¹Aᴴ**. "
    "Element model: cos¹·⁵(θ), square patch approximation. ★ = center element."
)