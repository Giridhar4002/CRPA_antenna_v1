"""
=============================================================
  CRPA ANTENNA — Null Steering Dashboard  (optimised)
  Controlled Reception Pattern Antenna
  Square Patch Elements · Circular Array · Adaptive Nulling
=============================================================
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import streamlit as st

# ──────────────────────────────────────────────────────
# PAGE STYLE
# ──────────────────────────────────────────────────────
st.set_page_config(page_title="CRPA Null Steering", page_icon="📡", layout="wide")
st.markdown("""
<style>
  .block-container { padding-top: 1rem; }
  .metric-box { background:#1a1d2e; border-radius:8px; padding:10px 14px; text-align:center; margin-bottom:6px; }
  .metric-val { font-size:1.25rem; font-weight:700; color:#00d4ff; }
  .metric-lbl { font-size:0.72rem; color:#aaa; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

st.title("📡 CRPA Antenna — Null Steering Dashboard")
st.caption("Controlled Reception Pattern Antenna · Square Patch Elements · Circular Array · Projection-Matrix Null Steering")

C = 299_792_458.0
DARK_BG  = "#0f1117"
PANEL_BG = "#1a1d2e"
GRID_COL = "#2e3150"
NULL_COLS = ["#e74c3c", "#e67e22", "#9b59b6", "#1abc9c"]
DES_COL  = "#2ecc71"
PAT_COL  = "#00d4ff"


def _ax_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color=GRID_COL, linestyle="--", alpha=0.5)


def _fig_dark(fig):
    fig.patch.set_facecolor(DARK_BG)


# ──────────────────────────────────────────────────────
# SIDEBAR — collect all inputs before any computation
# ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Array Parameters")
    freq_mhz     = st.number_input("Frequency (MHz)", 100.0, 6000.0, 1575.42, 0.5)
    N_elem       = st.number_input("Outer Elements (N)", 2, 16, 7, 1,
                                   help="Ring elements — a centre element is always added")
    spacing_frac = st.slider("Element Spacing (× λ)", 0.30, 1.00, 0.50, 0.01)
    steer_az     = st.slider("Desired Azimuth (°)",   -180, 180, 0,  1)
    steer_el     = st.slider("Desired Elevation (°)",    0,  90, 90, 1)
    patch_eff    = st.slider("Patch Efficiency (%)",    50, 100, 80, 1)
    dyn_range    = st.slider("Pattern Dynamic Range (dB)", 20, 60, 40, 5)

    st.markdown("---")
    st.header("🎯 Null Steering")
    n_nulls = st.selectbox("Number of Nulls", [0, 1, 2, 3, 4], index=2)

    null_dirs = []
    _def_az = [-60, 120, -130, 45]
    _def_el = [30,   60,   45,  20]
    for i in range(n_nulls):
        st.markdown(f"**Null {i+1}**")
        c1, c2 = st.columns(2)
        with c1:
            naz = st.slider(f"Az {i+1} (°)", -180, 180, _def_az[i], 1, key=f"naz{i}")
        with c2:
            nel = st.slider(f"El {i+1} (°)",   0,  90, _def_el[i], 1, key=f"nel{i}")
        null_dirs.append((naz, nel))

    st.markdown("---")
    show_3d = st.checkbox("Show 3D Pattern (slower)", value=False)


# ──────────────────────────────────────────────────────
# PURE CACHED PHYSICS  (re-runs only when inputs change)
# ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_array(N_elem, spacing_frac, freq_mhz):
    wl     = C / (freq_mhz * 1e6)
    radius = spacing_frac * wl
    ang    = 2 * np.pi * np.arange(N_elem) / N_elem
    ring   = np.zeros((N_elem, 3))
    ring[:, 0] = radius * np.cos(ang)
    ring[:, 1] = radius * np.sin(ang)
    pos = np.vstack([np.zeros((1, 3)), ring])
    return tuple(map(tuple, pos)), wl, radius


@st.cache_data(show_spinner=False)
def _sv(pos_t, az_deg, el_deg, wl):
    pos = np.array(pos_t)
    az, el = np.radians(az_deg), np.radians(el_deg)
    k = np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])
    return np.exp(1j * 2*np.pi/wl * pos @ k)


@st.cache_data(show_spinner=False)
def compute_weights(pos_t, wl, s_az, s_el, null_t):
    N   = len(pos_t)
    a_d = _sv(pos_t, s_az, s_el, wl)
    if not null_t:
        return tuple(a_d.real), tuple(a_d.imag)
    A_n = np.column_stack([_sv(pos_t, az, el, wl) for az, el in null_t])
    AHA = A_n.conj().T @ A_n
    try:
        P = np.eye(N) - A_n @ np.linalg.solve(AHA + 1e-6*np.eye(len(null_t)), A_n.conj().T)
    except np.linalg.LinAlgError:
        P = np.eye(N)
    w = P @ a_d
    nm = np.linalg.norm(w)
    w  = w / nm if nm > 1e-10 else a_d / N
    return tuple(w.real), tuple(w.imag)


@st.cache_data(show_spinner=False)
def compute_az_pattern(pos_t, wr, wi, wl, eff, n_pts=721):
    w   = np.array(wr) + 1j*np.array(wi)
    azs = np.linspace(-180, 180, n_pts)
    out = np.zeros(n_pts)
    for i, az in enumerate(azs):
        a     = _sv(pos_t, az, 90, wl)
        out[i] = (np.abs(np.dot(w.conj(), a)) * np.sqrt(eff))**2
    return tuple(azs), tuple(out)


@st.cache_data(show_spinner=False)
def compute_el_pattern(pos_t, wr, wi, wl, az_deg, eff, n_pts=361):
    w   = np.array(wr) + 1j*np.array(wi)
    els = np.linspace(0, 90, n_pts)
    out = np.zeros(n_pts)
    for i, el in enumerate(els):
        a        = _sv(pos_t, az_deg, el, wl)
        el_r     = np.radians(90 - el)
        elem_pat = max(np.cos(el_r), 0)**1.5
        out[i]   = (np.abs(np.dot(w.conj(), a)) * elem_pat * np.sqrt(eff))**2
    return tuple(els), tuple(out)


@st.cache_data(show_spinner=False)
def compute_3d_pattern(pos_t, wr, wi, wl, eff, n_az=91, n_el=46):
    w    = np.array(wr) + 1j*np.array(wi)
    az_v = np.linspace(-180, 180, n_az)
    el_v = np.linspace(0, 90, n_el)
    POW  = np.zeros((n_el, n_az))
    for ei, el in enumerate(el_v):
        el_r     = np.radians(90 - el)
        elem_pat = max(np.cos(el_r), 0)**1.5
        for ai, az in enumerate(az_v):
            a          = _sv(pos_t, az, el, wl)
            POW[ei,ai] = (np.abs(np.dot(w.conj(), a)) * elem_pat * np.sqrt(eff))**2
    return tuple(az_v), tuple(el_v), tuple(map(tuple, POW))


def to_db(arr, ref, floor_db):
    return 10*np.log10(np.maximum(np.array(arr)/max(ref, 1e-30), 10**(floor_db/10)))


# ──────────────────────────────────────────────────────
# CACHED FIGURE BUILDERS  (re-render only when data changes)
# ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _fig_polar(az_t, db_t, steer_az, null_t, depths_t, dyn):
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.2, 5.2))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)
    th = np.radians(az_t)
    r  = np.clip(np.array(db_t) + dyn, 0, None)
    ax.plot(th, r, color=PAT_COL, lw=1.8)
    ax.fill(th, r, color=PAT_COL, alpha=0.12)
    ax.axvline(np.radians(steer_az), color=DES_COL, lw=2, ls="--")
    for i, ((naz, _), d) in enumerate(zip(null_t, depths_t)):
        ax.axvline(np.radians(naz), color=NULL_COLS[i % 4], lw=1.8, ls=":")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ticks = np.linspace(0, dyn, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t - dyn)}dB" for t in ticks], color="white", fontsize=7)
    ax.tick_params(colors="white")
    ax.grid(color=GRID_COL, alpha=0.5)
    ax.set_title("Azimuth Cut  (El = 90°)", color="white", pad=12, fontsize=11)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _fig_cart(az_t, db_t, steer_az, null_t, depths_t, dyn):
    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    _fig_dark(fig)
    _ax_dark(ax)
    azs = np.array(az_t)
    dbs = np.array(db_t)
    ax.plot(azs, dbs, color=PAT_COL, lw=1.8, label="Pattern")
    ax.axvline(steer_az, color=DES_COL, lw=2, ls="--", label=f"Desired {steer_az}°")
    for i, ((naz, nel), d) in enumerate(zip(null_t, depths_t)):
        c = NULL_COLS[i % 4]
        ax.axvline(naz, color=c, lw=1.8, ls=":", label=f"N{i+1} Az={naz}° [{d:.1f}dB]")
        idx = int(np.argmin(np.abs(azs - naz)))
        ax.annotate(f"N{i+1}\n{d:.0f}dB", xy=(naz, dbs[idx]),
                    xytext=(naz + 10, -dyn * 0.55), color=c, fontsize=7.5,
                    arrowprops=dict(arrowstyle="->", color=c, lw=1.1))
    ax.set_ylim([-dyn - 2, 3])
    ax.set_xlim([-180, 180])
    ax.set_xlabel("Azimuth (°)", color="white", fontsize=9)
    ax.set_ylabel("Rel. Gain (dB)", color="white", fontsize=9)
    ax.set_title("Azimuth Pattern (Cartesian)", color="white", fontsize=11)
    ax.legend(fontsize=7.5, facecolor=PANEL_BG, labelcolor="white", framealpha=0.85)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _fig_el(el_t, db_t, steer_az, steer_el, null_t, dyn):
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    _fig_dark(fig)
    _ax_dark(ax)
    ax.plot(np.array(el_t), np.array(db_t), color="#f39c12", lw=1.8, label="Pattern")
    ax.axvline(steer_el, color=DES_COL, lw=2, ls="--", label=f"Desired El={steer_el}°")
    for i, (naz, nel) in enumerate(null_t):
        if abs(naz - steer_az) < 15:
            ax.axvline(nel, color=NULL_COLS[i % 4], lw=1.8, ls=":", label=f"N{i+1} El={nel}°")
    ax.set_ylim([-dyn - 2, 3])
    ax.set_xlim([0, 90])
    ax.set_xlabel("Elevation (°)", color="white", fontsize=9)
    ax.set_ylabel("Rel. Gain (dB)", color="white", fontsize=9)
    ax.set_title(f"Elevation Pattern  (Az = {steer_az}°)", color="white", fontsize=11)
    ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor="white", framealpha=0.85)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _fig_layout(pos_t, wl, steer_az, null_t, wm_t, wp_t, max_w, radius):
    pos    = np.array(pos_t)
    pos_cm = pos * 100
    wl_cm  = wl * 100
    patch  = 0.5 * wl_cm * 0.85
    cmap   = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)
    ax.set_aspect("equal")
    for i, (xi, yi, _) in enumerate(pos_cm):
        col = cmap(wm_t[i] / max_w)
        ax.add_patch(mpatches.FancyBboxPatch(
            (xi - patch / 2, yi - patch / 2), patch, patch,
            boxstyle="round,pad=0.05", lw=1.4, edgecolor="white", facecolor=col, alpha=0.9))
        ax.text(xi, yi, f"{wp_t[i]:.0f}°", ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")
    R = max(np.max(np.abs(pos_cm[:, 0])), 2) * 2.4
    dr = np.radians(steer_az)
    ax.annotate("", xy=(R * np.sin(dr), R * np.cos(dr)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=DES_COL, lw=2.2, mutation_scale=14))
    ax.text(R * 1.12 * np.sin(dr), R * 1.12 * np.cos(dr), "Des.",
            color=DES_COL, fontsize=8, ha="center", va="center", fontweight="bold")
    for i, (naz, _) in enumerate(null_t):
        c  = NULL_COLS[i % 4]
        nr = np.radians(naz)
        ax.annotate("", xy=(R * np.sin(nr), R * np.cos(nr)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8, mutation_scale=12))
        ax.text(R * 1.12 * np.sin(nr), R * 1.12 * np.cos(nr), f"N{i+1}",
                color=c, fontsize=8, ha="center", va="center", fontweight="bold")
    lim = R * 1.35
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xlabel("x (cm)", color="white", fontsize=9)
    ax.set_ylabel("y (cm)", color="white", fontsize=9)
    ax.set_title("Element Layout\n(color=|w|, text=phase°)", color="white", fontsize=10)
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
def _fig_phasors(wr_t, wi_t, max_w, N_total):
    w      = np.array(wr_t) + 1j * np.array(wi_t)
    mags   = np.abs(w)
    phases = np.angle(w, deg=True)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    _fig_dark(fig)
    ax.set_facecolor(PANEL_BG)
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color="#444", lw=1)
    ax.axhline(0, color="#444", lw=0.5)
    ax.axvline(0, color="#444", lw=0.5)
    colors_e = plt.cm.cool(np.linspace(0, 1, N_total))
    for i, wi in enumerate(w):
        wn = wi / max_w
        ax.quiver(0, 0, wn.real, wn.imag, angles="xy", scale_units="xy", scale=1,
                  color=colors_e[i], alpha=0.85, width=0.013)
        ax.scatter(wn.real, wn.imag, color=colors_e[i], s=55, zorder=5)
        ax.text(wn.real * 1.1, wn.imag * 1.1, str(i),
                color=colors_e[i], fontsize=8, ha="center")
    ax.set_xlim([-1.4, 1.4])
    ax.set_ylim([-1.4, 1.4])
    ax.set_aspect("equal")
    ax.set_xlabel("Real", color="white", fontsize=9)
    ax.set_ylabel("Imaginary", color="white", fontsize=9)
    ax.set_title("Weight Phasors (normalised)", color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.grid(color=GRID_COL, alpha=0.3, ls="--")
    if N_total <= 10:
        patches = [mpatches.Patch(color=plt.cm.cool(i / (max(N_total - 1, 1))),
                                  label=f"E{i}  |w|={mags[i]/max_w:.2f} ∠{phases[i]:.0f}°")
                   for i in range(N_total)]
        ax.legend(handles=patches, fontsize=6.5, facecolor=PANEL_BG,
                  labelcolor="white", framealpha=0.85, loc="lower right")
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner=False)
def _fig_3d(pos_t, wr_t, wi_t, wl, eff, dyn):
    az_t, el_t, pow_t = compute_3d_pattern(pos_t, wr_t, wi_t, wl, eff)
    POW  = np.array(pow_t)
    peak = max(np.max(POW), 1e-30)
    DB   = to_db(POW, peak, -dyn)
    R    = np.clip(DB + dyn, 0, None) / dyn
    az_v = np.array(az_t)
    el_v = np.array(el_t)
    AZ_r = np.radians(np.meshgrid(az_v, el_v)[0])
    EL_r = np.radians(np.meshgrid(az_v, el_v)[1])
    X = R * np.cos(EL_r) * np.sin(AZ_r)
    Y = R * np.cos(EL_r) * np.cos(AZ_r)
    Z = R * np.sin(EL_r)
    fig = plt.figure(figsize=(7.5, 5.5))
    _fig_dark(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(PANEL_BG)
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.plasma(R),
                    alpha=0.85, linewidth=0, antialiased=True)
    ax.set_title("3D Pattern (upper hemisphere)", color="white", fontsize=11)
    ax.tick_params(colors="white", labelsize=7)
    for p in [ax.xaxis, ax.yaxis, ax.zaxis]:
        p.pane.fill = False
        p.pane.set_edgecolor("#333")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────
# COMPUTE
# ──────────────────────────────────────────────────────
pos_t, wl, radius = build_array(int(N_elem), spacing_frac, freq_mhz)
null_t  = tuple(null_dirs)
N_total = len(pos_t)
eff     = patch_eff / 100.0

wr_t, wi_t = compute_weights(pos_t, wl, steer_az, steer_el, null_t)
w = np.array(wr_t) + 1j * np.array(wi_t)

weight_mags   = tuple(np.abs(w).tolist())
weight_phases = tuple(np.angle(w, deg=True).tolist())
max_w = max(max(weight_mags), 1e-30)

az_t, pow_az_t = compute_az_pattern(pos_t, wr_t, wi_t, wl, eff)
el_t, pow_el_t = compute_el_pattern(pos_t, wr_t, wi_t, wl, steer_az, eff)

peak    = max(max(pow_az_t), 1e-30)
db_az_t = tuple(to_db(pow_az_t, peak, -dyn_range).tolist())
db_el_t = tuple(to_db(pow_el_t, peak, -dyn_range).tolist())

# Null depths
null_depths = []
for naz, nel in null_dirs:
    a_n  = _sv(pos_t, naz, nel, wl)
    el_r = np.radians(90 - nel)
    pn   = (np.abs(np.dot(w.conj(), a_n)) * max(np.cos(el_r), 0)**1.5 * np.sqrt(eff))**2
    null_depths.append(float(to_db(np.array([pn]), peak, -dyn_range)[0]))

a_des    = _sv(pos_t, steer_az, steer_el, wl)
el_des_r = np.radians(90 - steer_el)
p_des    = (np.abs(np.dot(w.conj(), a_des)) * max(np.cos(el_des_r), 0)**1.5 * np.sqrt(eff))**2
gain_des = float(to_db(np.array([p_des]), peak, -dyn_range)[0])

peak_gain_dbi = (10 * np.log10(N_total)
                 + 10 * np.log10(max(0.01 * patch_eff * 4 * math.pi * 0.25, 1e-10))
                 + 5.0)

# ──────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────
st.markdown("---")
mcols = st.columns(6)
for col, (lbl, val) in zip(mcols, [
    ("Frequency",      f"{freq_mhz:.2f} MHz"),
    ("Wavelength",     f"{wl * 100:.2f} cm"),
    ("Total Elements", str(N_total)),
    ("Array Diameter", f"{2 * radius * 100:.2f} cm"),
    ("Est. Peak Gain", f"{peak_gain_dbi:.1f} dBi"),
    ("Gain @ Desired", f"{gain_des:.1f} dB rel."),
]):
    col.markdown(
        f'<div class="metric-box"><div class="metric-val">{val}</div>'
        f'<div class="metric-lbl">{lbl}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────
depths_t = tuple(null_depths)
wm_t     = weight_mags
wp_t     = weight_phases

col1, col2 = st.columns(2)
with col1:
    st.subheader("Azimuth Pattern — Polar")
    f = _fig_polar(az_t, db_az_t, steer_az, null_t, depths_t, dyn_range)
    st.pyplot(f, use_container_width=True)
    plt.close(f)

with col2:
    st.subheader("Azimuth Pattern — Cartesian")
    f = _fig_cart(az_t, db_az_t, steer_az, null_t, depths_t, dyn_range)
    st.pyplot(f, use_container_width=True)
    plt.close(f)

st.subheader(f"Elevation Pattern  (Az = {steer_az}°)")
f = _fig_el(el_t, db_el_t, steer_az, steer_el, null_t, dyn_range)
st.pyplot(f, use_container_width=True)
plt.close(f)

col3, col4 = st.columns(2)
with col3:
    st.subheader("Array Element Layout")
    f = _fig_layout(pos_t, wl, steer_az, null_t, wm_t, wp_t, max_w, radius)
    st.pyplot(f, use_container_width=True)
    plt.close(f)

with col4:
    st.subheader("Weight Phasors")
    f = _fig_phasors(wr_t, wi_t, max_w, N_total)
    st.pyplot(f, use_container_width=True)
    plt.close(f)

if show_3d:
    st.subheader("3D Radiation Pattern")
    with st.spinner("Rendering 3D…"):
        f = _fig_3d(pos_t, wr_t, wi_t, wl, eff, dyn_range)
        st.pyplot(f, use_container_width=True)
        plt.close(f)

# ──────────────────────────────────────────────────────
# TABLES
# ──────────────────────────────────────────────────────
st.markdown("---")
tc1, tc2 = st.columns(2)

with tc1:
    st.subheader("🎯 Null Summary")
    if n_nulls:
        st.dataframe(pd.DataFrame([{
            "Null": i + 1,
            "Az (°)": naz,
            "El (°)": nel,
            "Depth (dB)": f"{d:.1f}",
            "Status": ("✅ Deep" if d < -20 else ("⚠️ Shallow" if d < -10 else "❌ Weak"))
        } for i, ((naz, nel), d) in enumerate(zip(null_dirs, null_depths))]),
        use_container_width=True, hide_index=True)
    else:
        st.info("No nulls configured.")

with tc2:
    st.subheader("⚖️ Element Weights")
    pos_arr = np.array(pos_t)
    st.dataframe(pd.DataFrame({
        "Elem":      [f"E{i}{'★' if i == 0 else ''}" for i in range(N_total)],
        "x (cm)":    [f"{pos_arr[i, 0] * 100:.2f}" for i in range(N_total)],
        "y (cm)":    [f"{pos_arr[i, 1] * 100:.2f}" for i in range(N_total)],
        "|w| norm":  [f"{wm_t[i] / max_w:.3f}" for i in range(N_total)],
        "Phase (°)": [f"{wp_t[i]:.1f}" for i in range(N_total)],
    }), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "**Method:** Projection-matrix null steering — "
    "**w = P·a_d**  where  **P = I − Aₙ(AₙᴴAₙ)⁻¹Aₙᴴ**. "
    "Element pattern: cos¹·⁵(θ) (square microstrip patch). ★ = centre element."
)
