import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import google.generativeai as genai
import math

# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GT3 · Cockpit Analytics",
    layout="wide",
    page_icon="🏎",
    initial_sidebar_state="collapsed"
)

# ── MATPLOTLIB DEFAULTS ──────────────────────────────────────
plt.rcParams.update({
    'axes.facecolor':    '#080707',
    'figure.facecolor':  '#060505',
    'text.color':        '#E8E0D0',
    'axes.labelcolor':   '#3A3228',
    'xtick.color':       '#3A3228',
    'ytick.color':       '#3A3228',
    'grid.color':        '#141210',
    'grid.linewidth':    0.6,
    'font.family':       'monospace',
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.edgecolor':    '#2A2218',
})
C_GOLD  = '#C8952A'
C_AMBER = '#E8A020'
C_CREAM = '#E8E0D0'
C_RED   = '#CC2200'
C_GREEN = '#3DAA55'
C_BLUE  = '#4488CC'
C_DIM   = '#2A2218'
C_MID   = '#3A3228'

def ax_s(ax, yg=True):
    ax.set_facecolor('#080707')
    ax.spines['bottom'].set_color('#2A2218')
    for s in ['left','top','right']: ax.spines[s].set_visible(False)
    ax.tick_params(colors='#3A3228', length=3, pad=5)
    if yg: ax.grid(True, axis='y', color='#141210', lw=0.6, zorder=0)

# ── SESSION STATE ────────────────────────────────────────────
defaults = dict(
    screen=0,
    df=None, target_col=None, target_event=None,
    evidence_col=None, evidence_name=None, evidence_condition=None,
    numeric_cols=[], datetime_cols=[],
)
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

SCREENS = ['START', 'DATOS', 'BAYES', 'MODELO', 'CHARTS', 'IA']
MAX_SCREEN = 5

def go(n):
    st.session_state.screen = max(0, min(n, MAX_SCREEN))
    st.rerun()

# ── BACKEND — Metodología exacta del reporte (Listings 8 y 9) ──
def compute_bayes():
    s = st.session_state
    if s.df is None or s.target_col is None or s.evidence_condition is None:
        return None

    df, tc, te = s.df, s.target_col, s.target_event
    econd, nc  = s.evidence_condition, s.numeric_cols

    # ── Listing 8: Cálculo vectorizado de probabilidades (PDF pág. 15-16) ──
    total      = len(df)
    fallos_mask = df[tc] == te          # A: evento objetivo

    n_a  = int(fallos_mask.sum())       # |A|
    n_b  = int(econd.sum())             # |B|
    n_ab = int((fallos_mask & econd).sum())  # |A∩B|

    p_a   = fallos_mask.sum() / total                        # P(A) = |A|/N
    p_b   = econd.sum()       / total                        # P(B) = |B|/N
    p_b_a = n_ab / n_a        if n_a > 0 else 0.0           # P(B|A) = |A∩B|/|A|
    p_a_b = (p_b_a * p_a) / p_b if p_b > 0 else 0.0        # Bayes: P(B|A)·P(A)/P(B)

    acc = sens = spec = prec = tn = fp = fn = tp = None
    mv  = False
    used_cols = []

    if nc:
        # ── Listing 9: Entrenamiento GNB (PDF pág. 17) ──
        # features = df[numeric_cols].dropna()
        # Se excluye la columna objetivo si quedó entre las numéricas
        # Se excluyen columnas binarias 0/1 que son sub-tipos del evento (data leakage)
        feats_raw = df[nc].dropna()

        # Excluir target si es numérica
        cols_model = [c for c in feats_raw.columns if c != tc]

        # Excluir columnas 100% binarias (0/1) que derivan del target — data leakage
        # (ej: TWF, HDF, PWF, OSF, RNF en AI4I 2020)
        cols_model = [c for c in cols_model
                      if not (feats_raw[c].dropna().isin([0, 1]).all()
                              and feats_raw[c].nunique() <= 2)]

        if cols_model:
            feats = feats_raw[cols_model]
            y     = (df[tc].loc[feats.index] == te).astype(int)

            if len(feats) > 10 and len(y.unique()) > 1:
                try:
                    # nb = GaussianNB()
                    # nb.fit(features, y)       — resustitución: mismos datos
                    # y_pred = nb.predict(features)
                    nb = GaussianNB()
                    nb.fit(feats, y)
                    yp = nb.predict(feats)

                    # confusion_matrix con labels=[0,1] para garantizar orden
                    cm_r = confusion_matrix(y, yp, labels=[0, 1])
                    tn = int(cm_r[0, 0])   # TN: real 0, pred 0
                    fp = int(cm_r[0, 1])   # FP: real 0, pred 1
                    fn = int(cm_r[1, 0])   # FN: real 1, pred 0
                    tp = int(cm_r[1, 1])   # TP: real 1, pred 1

                    acc  = accuracy_score(y, yp)                            # (TP+TN)/N
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0        # TP/(TP+FN)
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0        # TN/(TN+FP)
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0        # TP/(TP+FP)
                    used_cols = cols_model
                    mv   = True
                except Exception:
                    mv = False

    return dict(p_a=p_a, p_b=p_b, p_b_a=p_b_a, p_a_b=p_a_b,
                n_a=n_a, n_b=n_b, n_ab=n_ab, n_total=total,
                acc=acc, sens=sens, spec=spec, prec=prec,
                tn=tn, fp=fp, fn=fn, tp=tp, mv=mv,
                used_cols=used_cols)

# ── SVG HELPERS ──────────────────────────────────────────────
def build_rpm_svg(gear, max_gear=6):
    """Build RPM arc segments as SVG string - NO f-string nesting issues"""
    pct = (gear - 1) / (max_gear - 1) if max_gear > 1 else 0
    gear_colors = {1:'#504438', 2:'#2A5A30', 3:'#2A4870',
                   4:'#705A10', 5:'#802020', 6:'#C8952A'}
    col = gear_colors.get(gear, '#504438')
    active = int(pct * 12)
    parts = []
    cx, cy, r = 60, 55, 44
    for i in range(12):
        a1 = math.radians(200 - i * (180/12))
        a2 = math.radians(200 - (i + 0.6) * (180/12))
        sx = cx + r * math.cos(a1); sy = cy - r * math.sin(a1)
        ex = cx + r * math.cos(a2); ey = cy - r * math.sin(a2)
        sc = col if i < active else '#1C1814'
        op = '1.0' if i < active else '0.35'
        parts.append(
            '<path d="M %.2f %.2f A %d %d 0 0 0 %.2f %.2f" '
            'fill="none" stroke="%s" stroke-width="6" '
            'stroke-linecap="round" opacity="%s"/>' % (sx, sy, r, r, ex, ey, sc, op)
        )
    gear_col = col
    parts.append(
        '<text x="60" y="50" text-anchor="middle" '
        'font-family="serif" font-size="26" '
        'fill="%s" letter-spacing="2">%d</text>' % (gear_col, gear)
    )
    parts.append(
        '<text x="60" y="62" text-anchor="middle" '
        'font-family="monospace" font-size="7" '
        'fill="#3A3228" letter-spacing="2">MARCHA</text>'
    )
    return '\n'.join(parts)

def build_mini_gauge(pct, color, label):
    """Build small semicircular gauge SVG - returns clean HTML string"""
    cx, cy, r = 45, 40, 30
    # background arc 200 -> -20 (220 deg sweep)
    def pt(deg):
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy - r * math.sin(rad)
    bsx, bsy = pt(200); bex, bey = pt(-20)
    val_deg = 200 - (min(pct, 100) / 100) * 220
    vex, vey = pt(val_deg)
    bg_path = "M %.2f %.2f A %d %d 0 1 0 %.2f %.2f" % (bsx, bsy, r, r, bex, bey)
    val_path = "M %.2f %.2f A %d %d 0 1 0 %.2f %.2f" % (bsx, bsy, r, r, vex, vey)
    val_str = "%.1f" % pct
    return (
        '<svg width="90" height="52" viewBox="0 0 90 52" xmlns="http://www.w3.org/2000/svg">'
        '<path d="%s" fill="none" stroke="#1C1814" stroke-width="5" stroke-linecap="round"/>'
        '<path d="%s" fill="none" stroke="%s" stroke-width="5" stroke-linecap="round" opacity="0.9"/>'
        '<text x="45" y="38" text-anchor="middle" font-family="serif" font-size="14" fill="%s">%s</text>'
        '<text x="45" y="48" text-anchor="middle" font-family="monospace" font-size="6" fill="#3A3228" letter-spacing="2">%s</text>'
        '</svg>'
    ) % (bg_path, val_path, color, color, val_str, label.upper())


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --leather-deep:   #0A0806;
  --leather-main:   #100C08;
  --leather-mid:    #181410;
  --leather-light:  #201A14;
  --leather-rim:    #2C241C;
  --chrome-dark:    #1C1814;
  --chrome-mid:     #2A2218;
  --chrome-light:   #3A3228;
  --chrome-bright:  #504438;
  --gold:           #C8952A;
  --gold-light:     #E8B040;
  --gold-dim:       #786040;
  --amber:          #E8A020;
  --cream:          #E8E0D0;
  --ivory:          #C0B898;
  --red:            #CC2200;
  --green:          #3DAA55;
  --blue:           #4488CC;
}
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--leather-deep) !important;
  color: var(--cream) !important;
}
#MainMenu, footer, header { visibility: hidden; }
* { box-sizing: border-box; }

.stApp {
  background:
    radial-gradient(ellipse 120% 60% at 50% -5%, rgba(200,149,42,0.04) 0%, transparent 55%),
    repeating-linear-gradient(17deg, transparent 0px, transparent 2.5px,
      rgba(255,255,255,0.007) 2.5px, rgba(255,255,255,0.007) 3px) !important;
  background-color: var(--leather-deep) !important;
}
.stApp::after {
  content: ''; position: fixed; inset: 0; pointer-events: none; z-index: 1;
  background: radial-gradient(ellipse 110% 110% at 50% 50%,
    transparent 30%, rgba(0,0,0,0.55) 100%);
}
.main .block-container { padding: 0 !important; max-width: 100% !important; }
/* Kill ALL default Streamlit vertical gaps */
.block-container > div:first-child { gap: 0 !important; }
div[data-testid="stVerticalBlock"] { gap: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
section[data-testid="stSidebar"] { display: none !important; }
/* Horizontal block columns - keep gap for content readability */
div[data-testid="stHorizontalBlock"] { padding: 0 !important; }
div[data-testid="column"] { padding: 0 8px !important; }
div[data-testid="column"]:first-child { padding-left: 0 !important; }
div[data-testid="column"]:last-child  { padding-right: 0 !important; }
/* Remove default element spacing only for top-level items */
.element-container { margin-bottom: 0 !important; }
.stButton { margin: 0 !important; padding: 0 !important; }
/* Kill Streamlit's built-in top padding on main content */
.main > div:first-child { padding-top: 0 !important; }
[data-testid="stAppViewContainer"] > section > div { padding: 0 !important; }
.stMarkdown { margin: 0 !important; }
[data-testid="stMarkdownContainer"] { margin: 0 !important; }

/* ── COCKPIT HEADER ── */
.cockpit-header {
  position: relative; width: 100%;
  background: linear-gradient(180deg, #060504 0%, var(--leather-main) 100%);
  border-bottom: 1px solid rgba(200,149,42,0.12);
  overflow: hidden;
}
.cockpit-header::before {
  content: ''; position: absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg, transparent 0%, var(--gold-dim) 20%,
    var(--gold) 50%, var(--gold-dim) 80%, transparent 100%);
}
.header-inner {
  display: grid; grid-template-columns: 260px 1fr 260px;
  align-items: center; padding: 18px 40px 14px; gap: 0; position: relative; z-index: 2;
}
.h-left { display: flex; align-items: center; gap: 14px; }
.gear-readout { display: flex; flex-direction: column; }
.gr-gear {
  font-family: 'Bebas Neue', serif; font-size: 50px;
  line-height: 0.9; letter-spacing: 2px; transition: color 0.4s ease;
}
.gr-label {
  font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px;
  color: #786040; text-transform: uppercase; margin-top: 5px;
}
.gr-screen {
  font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 3px;
  text-transform: uppercase; margin-top: 3px;
}
.h-center { text-align: center; }
.logo-main {
  font-family: 'Bebas Neue', serif; font-size: 34px;
  letter-spacing: 6px; line-height: 1; color: var(--cream);
}
.logo-accent { color: var(--gold); }
.logo-sub {
  font-family: 'DM Mono', monospace; font-size: 8px; letter-spacing: 3px;
  text-transform: uppercase; color: #504438; margin-top: 6px;
}
.shift-lights {
  display: flex; gap: 5px; justify-content: center; margin-top: 10px; align-items: center;
}
.sl {
  width: 20px; height: 7px; border-radius: 2px;
  background: var(--chrome-dark); border: 1px solid rgba(255,255,255,0.04);
  transition: all 0.12s ease;
}
.sl.g { background: #22AA44; box-shadow: 0 0 8px #22AA44; }
.sl.y { background: #CCAA00; box-shadow: 0 0 8px #CCAA00; }
.sl.r { background: #CC2200; box-shadow: 0 0 12px #CC2200;
        animation: rpulse 0.25s ease-in-out infinite alternate; }
@keyframes rpulse { from{opacity:1} to{opacity:0.3} }
.h-right { display: flex; align-items: center; justify-content: flex-end; gap: 18px; }
.pilot-info { text-align: right; }
.pi-label {
  font-family: 'DM Mono', monospace; font-size: 8px; letter-spacing: 3px;
  color: #504438; text-transform: uppercase;
}
.pi-name {
  font-family: 'Bebas Neue', serif; font-size: 24px;
  letter-spacing: 3px; line-height: 1.1; color: var(--cream);
}
.pi-sub {
  font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 2px;
  color: #786040; margin-top: 3px;
}
.header-stripe {
  height: 3px;
  background: linear-gradient(90deg, transparent 0%, var(--leather-rim) 15%,
    var(--gold) 40%, var(--amber) 50%, var(--gold) 60%, var(--leather-rim) 85%, transparent 100%);
  animation: stripeIn 0.8s cubic-bezier(.16,1,.3,1) both;
  transform-origin: center;
}
@keyframes stripeIn { from{transform:scaleX(0)} to{transform:scaleX(1)} }

/* ── GEAR DOTS ── */
.gear-dots { display: flex; gap: 8px; align-items: center; }
.gdot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--chrome-dark); border: 1px solid rgba(200,149,42,0.1);
  transition: all 0.3s ease;
}
.gdot.done   { background: rgba(200,149,42,0.3); }
.gdot.active { background: var(--gold); border-color: var(--gold-light);
               box-shadow: 0 0 8px var(--gold); transform: scale(1.4); }

/* ── DISPLAY AREA ── */
.display-wrap {
  padding: 28px 44px 36px;
  animation: screenSlide 0.35s cubic-bezier(.16,1,.3,1) both;
}
@keyframes screenSlide {
  from { opacity:0; transform:translateY(12px) scale(0.99); }
  to   { opacity:1; transform:translateY(0) scale(1); }
}
.section-title {
  font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 4px;
  color: #786040; text-transform: uppercase;
  display: flex; align-items: center; gap: 14px; margin-bottom: 24px;
}
.section-title::before {
  content:''; display:block; width:28px; height:1px; background:var(--gold);
}
.section-title::after {
  content:''; flex:1; height:1px;
  background:linear-gradient(90deg, rgba(200,149,42,0.2), transparent);
}

/* ── WELCOME ── */
.welcome-wrap {
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; text-align:center; padding:28px 20px 20px;
}
.crest-outer {
  width:180px; height:180px; border-radius:50%;
  border:1px solid rgba(200,149,42,0.15);
  display:flex; align-items:center; justify-content:center;
  margin:0 auto 28px; animation:crestPulse 4s ease-in-out infinite;
}
@keyframes crestPulse {
  0%,100% { box-shadow:0 0 0 0 rgba(200,149,42,0); border-color:rgba(200,149,42,0.15); }
  50%     { box-shadow:0 0 40px 8px rgba(200,149,42,0.08); border-color:rgba(200,149,42,0.35); }
}
.crest-inner {
  width:148px; height:148px; border-radius:50%;
  border:1px solid rgba(200,149,42,0.07);
  display:flex; align-items:center; justify-content:center;
}
.crest-logo {
  font-family:'Bebas Neue',serif; font-size:42px;
  letter-spacing:4px; line-height:0.9; text-align:center;
}
.crest-model {
  font-family:'DM Mono',monospace; font-size:7px; letter-spacing:4px;
  color:var(--chrome-light); text-transform:uppercase; margin-top:6px; text-align:center;
}
.welcome-title {
  font-family:'Bebas Neue',serif; font-size:13px; letter-spacing:8px;
  color:var(--gold-dim); text-transform:uppercase; margin-bottom:12px;
}
.welcome-desc {
  font-family:'DM Sans',sans-serif; font-size:14px; font-weight:300;
  line-height:1.8; color:var(--chrome-bright); max-width:480px; margin:0 auto 28px;
}
.welcome-grid {
  display:grid; grid-template-columns:repeat(5,1fr);
  gap:8px; max-width:560px; margin:0 auto 28px;
}
.wg-item {
  background:var(--leather-mid); border:1px solid rgba(200,149,42,0.1);
  border-top:2px solid var(--gold-dim); border-radius:2px;
  padding:14px 10px; text-align:center; transition:all 0.25s ease;
}
.wg-item:hover { border-top-color:var(--gold); transform:translateY(-2px); }
.wg-num { font-family:'Bebas Neue',serif; font-size:26px; color:var(--gold); }
.wg-lbl { font-family:'DM Mono',monospace; font-size:8px; letter-spacing:2px;
           color:#786040; text-transform:uppercase; margin-top:5px; }
.start-hint {
  font-family:'DM Mono',monospace; font-size:10px; letter-spacing:4px;
  color:#504438; text-transform:uppercase;
  animation:hintBlink 2.5s ease-in-out infinite;
}
@keyframes hintBlink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── CARDS ── */
.gt3-card {
  background:var(--leather-mid); border:1px solid rgba(255,255,255,0.05);
  border-top:2px solid var(--gold-dim); border-radius:2px; padding:20px 22px;
  position:relative; overflow:hidden; transition:all 0.25s cubic-bezier(.16,1,.3,1);
}
.gt3-card:hover { border-top-color:var(--gold); transform:translateY(-2px);
                  box-shadow:0 8px 30px rgba(0,0,0,0.6); }
.card-tag {
  font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2.5px;
  color:#786040; text-transform:uppercase; margin-bottom:10px;
}
.card-big {
  font-family:'Bebas Neue',serif; font-size:48px;
  letter-spacing:-1px; line-height:0.9;
}
.card-sub {
  font-family:'DM Sans',sans-serif; font-size:12px; font-weight:400;
  color:#786040; margin-top:10px; line-height:1.5;
}
.card-bar { height:2px; background:var(--chrome-dark); border-radius:1px; margin-top:16px; overflow:hidden; }
.card-fill { height:100%; border-radius:1px; }

/* ── POSTERIOR HERO ── */
.post-hero {
  background:var(--leather-mid); border:1px solid rgba(200,149,42,0.2);
  border-radius:3px; padding:26px 22px; text-align:center;
  position:relative; overflow:hidden; height:100%;
}
.post-hero::before {
  content:''; position:absolute; inset:0;
  background:radial-gradient(ellipse 100% 70% at 50% 0%,
    rgba(200,149,42,0.07) 0%, transparent 60%);
}
.ph-eye {
  font-family:'DM Mono',monospace; font-size:7px; letter-spacing:4px;
  color:var(--gold); text-transform:uppercase; margin-bottom:10px; position:relative;
}
.ph-num {
  font-family:'Bebas Neue',serif; font-size:88px; line-height:0.88;
  letter-spacing:-3px; position:relative;
}
.ph-unit { font-size:0.3em; color:var(--chrome-light); }
.ph-delta {
  display:inline-flex; align-items:center; gap:6px;
  font-family:'Bebas Neue',serif; font-size:18px; letter-spacing:2px;
  padding:5px 14px; border-radius:2px; margin-top:8px; position:relative;
}
.ph-formula {
  font-family:'DM Mono',monospace; font-size:9px; color:var(--chrome-mid);
  line-height:1.8; margin-top:14px; padding-top:14px;
  border-top:1px solid rgba(255,255,255,0.04); position:relative;
}
.ph-evidence {
  font-family:'DM Mono',monospace; font-size:8px; color:var(--chrome-light);
  letter-spacing:1px; line-height:1.8; margin-top:12px; position:relative;
}

/* ── MODEL STRIP ── */
.model-strip {
  display:grid; grid-template-columns:repeat(4,1fr);
  gap:1px; background:rgba(200,149,42,0.07); border-radius:2px; overflow:hidden; margin-bottom:12px;
}
.ms-cell { background:var(--leather-mid); padding:18px 20px; position:relative; }
.ms-cell::before { content:''; position:absolute; top:0; left:0; width:100%; height:2px; }
.ms-y::before { background:var(--gold); }
.ms-g::before { background:var(--green); }
.ms-b::before { background:var(--blue); }
.ms-a::before { background:var(--amber); }
.ms-lbl { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px;
           color:#786040; text-transform:uppercase; margin-bottom:10px; }
.ms-val { font-family:'Bebas Neue',serif; font-size:38px; letter-spacing:-1px; line-height:1; }
.ms-sub { font-family:'DM Sans',sans-serif; font-size:11px; font-weight:400;
          color:#504438; margin-top:6px; }

/* ── CM STRIP ── */
.cm-strip {
  display:grid; grid-template-columns:1fr 1fr 1fr 1fr;
  gap:1px; background:rgba(200,149,42,0.07); border-radius:2px; overflow:hidden;
}
.cm-cell { background:var(--leather-mid); padding:18px 20px; }
.cm-lbl { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px;
           color:#786040; text-transform:uppercase; margin-bottom:8px; }
.cm-val { font-family:'Bebas Neue',serif; font-size:34px; letter-spacing:-1px; line-height:1; }

/* ── INFO PANEL ── */
.info-panel {
  background:var(--leather-mid); border:1px solid rgba(255,255,255,0.05);
  border-left:2px solid var(--gold); border-radius:2px; padding:16px 20px;
}
.ip-tag { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2.5px;
          color:var(--gold); text-transform:uppercase; margin-bottom:10px; }
.ip-text { font-family:'DM Sans',sans-serif; font-size:13px; font-weight:400;
           color:#786040; line-height:1.75; }

/* ── STATUS ── */
.status-row { display:flex; align-items:center; gap:12px;
              padding:11px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.sdot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.slbl { font-family:'DM Mono',monospace; font-size:9px; letter-spacing:2px; text-transform:uppercase; }

/* ── NATIVE OVERRIDES ── */
[data-testid="stFileUploader"] {
  background:var(--leather-mid) !important;
  border:1px dashed rgba(200,149,42,0.25) !important; border-radius:2px !important;
}
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div,
[data-testid="stTextInput"] > div > div {
  background:var(--leather-mid) !important;
  border:1px solid rgba(200,149,42,0.15) !important; border-radius:2px !important;
  color:var(--cream) !important; font-family:'DM Sans',sans-serif !important;
  font-size:14px !important; font-weight:500 !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] > div > div:focus-within,
[data-testid="stTextInput"] > div > div:focus-within {
  border-color:rgba(200,149,42,0.5) !important;
}
[data-testid="stTextInput"] > div > div {
  font-family:'DM Mono',monospace !important; font-size:13px !important;
}
label[data-testid="stWidgetLabel"] p {
  font-family:'DM Mono',monospace !important; font-size:9px !important;
  letter-spacing:2.5px !important; color:#786040 !important;
  text-transform:uppercase !important;
}
.stButton > button {
  font-family:'Bebas Neue',serif !important; font-size:13px !important;
  letter-spacing:4px !important; text-transform:uppercase !important;
  background:var(--chrome-dark) !important; color:var(--gold) !important;
  border:1px solid rgba(200,149,42,0.25) !important; border-radius:2px !important;
  padding:14px 36px 11px !important; transition:all 0.2s ease !important;
}
.stButton > button:hover {
  background:var(--gold) !important; color:var(--leather-deep) !important;
  box-shadow:0 0 16px rgba(200,149,42,0.3) !important; transform:translateY(-2px) !important;
}
[data-testid="stExpander"] {
  background:var(--leather-mid) !important;
  border:1px solid rgba(255,255,255,0.05) !important;
  border-left:2px solid var(--gold-dim) !important; border-radius:2px !important;
}
[data-testid="stExpander"] summary {
  font-family:'DM Mono',monospace !important; font-size:9px !important;
  letter-spacing:2.5px !important; text-transform:uppercase !important;
  color:#786040 !important; padding:14px 18px !important;
}
.stMarkdown p {
  font-family:'DM Sans',sans-serif !important; font-size:12px !important;
  font-weight:300 !important; color:var(--chrome-bright) !important;
}
[data-testid="column"]:nth-child(1) { animation:fadeUp .4s ease .05s both; }
[data-testid="column"]:nth-child(2) { animation:fadeUp .4s ease .12s both; }
[data-testid="column"]:nth-child(3) { animation:fadeUp .4s ease .19s both; }
[data-testid="column"]:nth-child(4) { animation:fadeUp .4s ease .26s both; }
@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }
hr { border:none !important; border-top:1px solid rgba(200,149,42,0.07) !important; margin:20px 0 !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# BUILD HEADER COMPONENTS — all as Python strings, no nested f-strings
# ═══════════════════════════════════════════════════════════════
scr  = st.session_state.screen
gear = scr + 1

GEAR_COLORS = {1:'#504438', 2:'#2A5A30', 3:'#2A4870',
               4:'#705A10', 5:'#802020', 6:'#C8952A'}
gear_col    = GEAR_COLORS.get(gear, '#504438')
gear_screen = SCREENS[scr]

# Shift lights
def build_shift_lights(gear, max_gear=6):
    total  = 12
    filled = int((gear - 1) / (max_gear - 1) * total) if max_gear > 1 else 0
    lights = []
    for i in range(total):
        if i < filled:
            cls = 'sl r' if filled >= 10 else ('sl y' if filled >= 7 else 'sl g')
        else:
            cls = 'sl'
        lights.append('<div class="%s"></div>' % cls)
    return ''.join(lights[:6]), ''.join(lights[6:])

sl_left, sl_right = build_shift_lights(gear)

# Live posterior gauge
res_live  = compute_bayes()
post_pct  = res_live['p_a_b'] * 100 if res_live else 0.0
post_col  = C_GREEN if (res_live and res_live['p_a_b'] > res_live['p_a']) else C_GOLD
mini_gauge_html = build_mini_gauge(post_pct, post_col, 'P(A|B)')

# RPM ring SVG
rpm_inner = build_rpm_svg(gear)

# Gear dots
dot_parts = []
for i, nm in enumerate(SCREENS):
    if i == scr:    dcls = 'gdot active'
    elif i < scr:   dcls = 'gdot done'
    else:           dcls = 'gdot'
    dot_parts.append('<div class="%s" title="%s"></div>' % (dcls, nm))
dots_html = ''.join(dot_parts)

# Screen label spans for dots bar — larger, higher contrast
label_parts = []
for i, nm in enumerate(SCREENS):
    if i == scr:
        style = 'color:#C8952A;font-weight:700;font-size:9px;letter-spacing:3px;'
    else:
        style = 'color:#504438;font-size:8px;letter-spacing:2px;'
    label_parts.append(
        '<span style="font-family:DM Mono,monospace;text-transform:uppercase;%s">%s</span>' % (style, nm)
    )
labels_html = '&nbsp;<span style="color:#2A2218;">&middot;</span>&nbsp;'.join(label_parts)

# ─── RENDER HEADER ──────────────────────────────────────────
header_html = """
<div class="cockpit-header">
  <div class="header-inner">
    <div class="h-left">
      <svg width="120" height="70" viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg">
        {rpm}
      </svg>
      <div class="gear-readout">
        <div class="gr-gear" style="color:{gc};text-shadow:0 0 20px {gc}44;">{g}</div>
        <div class="gr-label">Marcha activa</div>
        <div class="gr-screen" style="color:{gc}88;">{gs}</div>
      </div>
    </div>
    <div class="h-center">
      <div class="logo-main">BAYES<span class="logo-accent">&middot;</span>GT3</div>
      <div class="logo-sub">Motor de Inferencia Estad&iacute;stica &middot; Cockpit Analytics</div>
      <div class="shift-lights">
        {sl_l}
        <div style="width:1px;height:7px;background:rgba(200,149,42,0.15);margin:0 4px;"></div>
        {sl_r}
      </div>
    </div>
    <div class="h-right">
      <div class="pilot-info">
        <div class="pi-label">APP hecha por: </div>
        <div class="pi-name">DCH|CDEL|CAAV </div>
        <div class="pi-sub">Estadistica y Probabilidad &middot; UP Chiapas</div>
      </div>
      <div>{gauge}</div>
    </div>
  </div>
  <div class="header-stripe"></div>
</div>
""".format(
    rpm=rpm_inner, gc=gear_col, g=gear, gs=gear_screen,
    sl_l=sl_left, sl_r=sl_right, gauge=mini_gauge_html
)
st.markdown(header_html, unsafe_allow_html=True)

# ─── PADDLE NAVIGATION BAR ──────────────────────────────────
# Three real Streamlit columns — left paddle, center dots, right paddle
# CSS hides the default button look and replaces with GT3 paddle style
st.markdown("""
<style>
/* ── PADDLE BAR WRAPPER ── */
div[data-testid="stHorizontalBlock"].paddle-row {
  background: #100C08;
  border-bottom: 1px solid rgba(200,149,42,0.10);
  padding: 0 !important;
  margin: 0 !important;
  align-items: center !important;
}
/* Left paddle col */
div[data-testid="stHorizontalBlock"].paddle-row
  > div[data-testid="column"]:first-child {
  padding: 10px 0 10px 32px !important;
  display: flex; align-items: center;
}
/* Right paddle col */
div[data-testid="stHorizontalBlock"].paddle-row
  > div[data-testid="column"]:last-child {
  padding: 10px 32px 10px 0 !important;
  display: flex; align-items: center; justify-content: flex-end;
}
/* Style BOTH paddle buttons */
div[data-testid="stHorizontalBlock"].paddle-row .stButton > button {
  font-family: 'Bebas Neue', serif !important;
  font-size: 22px !important;
  letter-spacing: 4px !important;
  line-height: 1 !important;
  background: #1C1814 !important;
  color: #C8952A !important;
  border: 1px solid rgba(200,149,42,0.30) !important;
  border-radius: 3px !important;
  padding: 10px 28px 8px !important;
  min-width: 80px !important;
  transition: all 0.2s ease !important;
  box-shadow: inset 0 -2px 0 rgba(0,0,0,0.5) !important;
}
div[data-testid="stHorizontalBlock"].paddle-row .stButton > button:hover {
  background: #C8952A !important;
  color: #0A0806 !important;
  box-shadow: 0 0 20px rgba(200,149,42,0.4), inset 0 -2px 0 rgba(0,0,0,0.3) !important;
  transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# Inject class on the next horizontal block via a marker
st.markdown('<div class="paddle-row" style="display:contents;">', unsafe_allow_html=True)
pb_l, pb_c, pb_r = st.columns([1, 3, 1], gap="small")

with pb_l:
    if scr > 0:
        if st.button("\u2212", key="dn", help=SCREENS[scr - 1]):
            go(scr - 1)

with pb_c:
    center_nav = (
        '<div style="display:flex;flex-direction:column;align-items:center;'
        'gap:8px;padding:12px 0 10px;">'
        '<div style="display:flex;gap:10px;align-items:center;">'
        + dots_html +
        '</div>'
        '<div style="display:flex;gap:6px;align-items:center;">'
        + labels_html +
        '</div>'
        '</div>'
    )
    st.markdown(center_nav, unsafe_allow_html=True)

with pb_r:
    if scr < MAX_SCREEN:
        if st.button("\u002B", key="up", help=SCREENS[scr + 1]):
            go(scr + 1)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div style="height:1px;background:rgba(200,149,42,0.10);margin:0;"></div>',
            unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SCREENS
# ═══════════════════════════════════════════════════════════════
st.markdown('<div class="display-wrap">', unsafe_allow_html=True)

# ── SCREEN 0: WELCOME ───────────────────────────────────────
if scr == 0:
    st.markdown("""
<div class="welcome-wrap">
  <div class="crest-outer">
    <div class="crest-inner">
      <div>
        <div class="crest-logo">BAYES<br><span style="color:var(--gold);">GT3</span></div>
        <div class="crest-model">911 &middot; Cockpit Analytics</div>
      </div>
    </div>
  </div>
  <div class="welcome-title">Motor de Inferencia Estad&iacute;stica</div>
  <p class="welcome-desc">
    An&aacute;lisis bayesiano completo con interfaz de cockpit GT3.<br>
    Sube de marcha con la paleta derecha para iniciar la sesi&oacute;n de an&aacute;lisis.
  </p>
  <div class="welcome-grid">
    <div class="wg-item"><div class="wg-num">1</div><div class="wg-lbl">Datos</div></div>
    <div class="wg-item"><div class="wg-num">2</div><div class="wg-lbl">Bayes</div></div>
    <div class="wg-item"><div class="wg-num">3</div><div class="wg-lbl">Modelo</div></div>
    <div class="wg-item"><div class="wg-num">4</div><div class="wg-lbl">Charts</div></div>
    <div class="wg-item"><div class="wg-num">5</div><div class="wg-lbl">IA</div></div>
  </div>
  <div class="start-hint">&#9658;&nbsp; Paleta derecha &middot; DATOS</div>
</div>
""", unsafe_allow_html=True)

# ── SCREEN 1: DATA ───────────────────────────────────────────
elif scr == 1:
    st.markdown('<div class="section-title">Ingesta de Datos &middot; Configuraci&oacute;n de Variables</div>',
                unsafe_allow_html=True)
    col_up, col_mid, col_stat = st.columns([1, 1.6, 0.85], gap="large")

    with col_up:
        st.markdown('<div class="card-tag">Carga de Dataset</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")

        if uploaded:
            df_ = pd.read_csv(uploaded)
            for c in df_.columns:
                if df_[c].dtype == 'object':
                    try: df_[c] = pd.to_datetime(df_[c])
                    except (ValueError, TypeError): pass
            st.session_state.df            = df_
            st.session_state.datetime_cols = df_.select_dtypes(include=['datetime64']).columns.tolist()
            st.session_state.numeric_cols  = df_.select_dtypes(include=['int64','float64']).columns.tolist()
            _bin = [c for c in df_.columns if df_[c].nunique() == 2]
            _cat = df_.select_dtypes(include=['object','category']).columns.tolist()
            st.session_state._opciones = _bin if _bin else _cat
            n_r, n_c = df_.shape
            st.markdown(
                '<div class="info-panel" style="margin-top:14px;">'
                '<div class="ip-tag">Archivo recibido</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px;">'
                '<div><div class="card-tag">Filas</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:30px;color:' + C_GOLD + ';">' + str(n_r) + '</div></div>'
                '<div><div class="card-tag">Cols</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:30px;color:' + C_GOLD + ';">' + str(n_c) + '</div></div>'
                '<div><div class="card-tag">Num&eacute;ricas</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:30px;color:' + C_GREEN + ';">'
                + str(len(st.session_state.numeric_cols)) + '</div></div>'
                '<div><div class="card-tag">Binarias</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:30px;color:' + C_BLUE + ';">'
                + str(len(_bin)) + '</div></div>'
                '</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="height:200px;display:flex;align-items:center;justify-content:center;'
                'background:var(--leather-mid);border:1px dashed rgba(200,149,42,0.12);'
                'border-radius:2px;margin-top:14px;">'
                '<div style="text-align:center;">'
                '<div style="font-family:Bebas Neue,serif;font-size:40px;'
                'letter-spacing:4px;color:rgba(200,149,42,0.07);">NO DATA</div>'
                '<div style="font-family:DM Mono,monospace;font-size:8px;'
                'letter-spacing:3px;color:#2A2218;margin-top:6px;">Carga un CSV para continuar</div>'
                '</div></div>',
                unsafe_allow_html=True
            )

    with col_mid:
        df_ = st.session_state.df
        ops = getattr(st.session_state, '_opciones', [])
        if df_ is not None and ops:
            st.markdown('<div class="card-tag">Configuraci&oacute;n del Modelo</div>', unsafe_allow_html=True)
            r1, r2 = st.columns(2)
            with r1:
                tc = st.selectbox("VARIABLE OBJETIVO", ops)
                st.session_state.target_col = tc
            with r2:
                te_opts = df_[tc].dropna().unique()
                te = st.selectbox("EVENTO", te_opts)
                st.session_state.target_event = te
            r3, r4 = st.columns(2)
            with r3:
                ec = st.selectbox("EVIDENCIA", df_.columns.drop(tc))
                st.session_state.evidence_col = ec
            with r4:
                if df_[ec].dtype in ['int64', 'float64']:
                    mv = float(df_[ec].mean())
                    ub = st.number_input("UMBRAL", value=mv, format="%.4f")
                    st.session_state.evidence_condition = df_[ec] > ub
                    st.session_state.evidence_name = "%s > %.3f" % (ec, ub)
                else:
                    ev = st.selectbox("VALOR", df_[ec].dropna().unique())
                    st.session_state.evidence_condition = df_[ec] == ev
                    st.session_state.evidence_name = "%s = %s" % (ec, ev)
            with st.expander("VER DATASET"):
                st.dataframe(df_.head(8), use_container_width=True)
        else:
            st.markdown(
                '<div style="height:260px;display:flex;align-items:center;justify-content:center;'
                'background:var(--leather-mid);border:1px dashed rgba(200,149,42,0.1);border-radius:2px;">'
                '<div class="start-hint">Carga un archivo CSV</div>'
                '</div>',
                unsafe_allow_html=True
            )

    with col_stat:
        st.markdown('<div class="card-tag">Estado del Sistema</div>', unsafe_allow_html=True)
        checks = [
            ("Dataset",   st.session_state.df is not None),
            ("Objetivo",  st.session_state.target_col is not None),
            ("Evidencia", st.session_state.evidence_condition is not None),
        ]
        for lbl, ok in checks:
            dc_c = C_GOLD if ok else C_DIM
            txt_c = '#C0B898' if ok else '#2A2218'
            st.markdown(
                '<div class="status-row">'
                '<div class="sdot" style="background:' + dc_c + ';' +
                ('box-shadow:0 0 6px ' + dc_c + ';' if ok else '') + '"></div>'
                '<span class="slbl" style="color:' + txt_c + ';">' + lbl + '</span>'
                '<span style="margin-left:auto;font-family:DM Mono,monospace;font-size:7px;'
                'letter-spacing:2px;color:' + dc_c + ';">' + ('OK' if ok else '\u2014') + '</span>'
                '</div>',
                unsafe_allow_html=True
            )
        if st.session_state.target_col:
            st.markdown(
                '<div class="info-panel" style="margin-top:16px;">'
                '<div class="ip-tag">Configuraci&oacute;n activa</div>'
                '<div class="ip-text">'
                'Objetivo: <strong style="color:' + C_GOLD + ';">' + str(st.session_state.target_col) + '</strong><br>'
                'Evento: ' + str(st.session_state.target_event) + '<br>'
                'Evidencia: ' + (st.session_state.evidence_name or '&mdash;') +
                '</div></div>',
                unsafe_allow_html=True
            )


# ── SCREEN 2: BAYES ──────────────────────────────────────────
elif scr == 2:
    st.markdown('<div class="section-title">An&aacute;lisis Bayesiano &middot; Probabilidades en Vivo</div>',
                unsafe_allow_html=True)
    res = compute_bayes()
    if res is None:
        st.markdown(
            '<div class="info-panel"><div class="ip-tag">Sin configuraci&oacute;n</div>'
            '<div class="ip-text">Ve a DATOS (marcha 2) y configura el dataset primero.</div></div>',
            unsafe_allow_html=True)
    else:
        p_a   = res['p_a'];  p_b   = res['p_b']
        p_b_a = res['p_b_a']; p_a_b = res['p_a_b']
        df_   = st.session_state.df
        tc_   = st.session_state.target_col
        te_   = st.session_state.target_event
        econd_= st.session_state.evidence_condition

        # Conteos exactos — vienen del backend
        n_total = res['n_total']
        n_a  = res['n_a']
        n_b  = res['n_b']
        n_ab = res['n_ab']

        da   = p_a_b - p_a
        dr   = (da / p_a * 100) if p_a > 0 else 0
        dc_c = C_GREEN if da >= 0 else C_RED
        arr  = '&#9650;' if da >= 0 else '&#9660;'

        g_col, p_col = st.columns([2.3, 1], gap="large")

        with g_col:
            probs = [
                ('P(A) &middot; Probabilidad Previa',   'Frecuencia hist&oacute;rica sin evidencia', p_a,   C_AMBER),
                ('P(B) &middot; Evidencia Marginal',    'Cobertura activa de la evidencia',          p_b,   C_GOLD),
                ('P(B|A) &middot; Verosimilitud',       'Se&ntilde;al: evidencia dado evento',       p_b_a, C_BLUE),
                ('P(A|B) &middot; Posterior Bayesiana', 'Riesgo actualizado tras la evidencia',      p_a_b, C_GREEN),
            ]
            gg1, gg2 = st.columns(2, gap="small")
            gcols = [gg1, gg2, gg1, gg2]
            for i, (lbl, desc, val, col) in enumerate(probs):
                pct = val * 100
                with gcols[i]:
                    st.markdown(
                        '<div class="gt3-card" style="margin-bottom:12px;border-top-color:' + col + '55;">'
                        '<div class="card-tag">' + lbl + '</div>'
                        '<div class="card-big" style="color:' + col + ';">'
                        '%.2f<span style="font-size:.32em;color:#504438"> %%</span></div>' % pct +
                        '<div class="card-sub">' + desc + '</div>'
                        '<div class="card-bar"><div class="card-fill" style="width:%.1f%%;'
                        'background:linear-gradient(90deg,%s44,%s);"></div></div>'
                        '</div>' % (min(pct, 100), col, col),
                        unsafe_allow_html=True
                    )

            # ── Panel de verificación con conteos exactos ──────────────
            st.markdown(
                '<div style="margin-top:14px;background:#0D0A07;border:1px solid rgba(200,149,42,0.15);'
                'border-left:3px solid ' + C_GOLD + ';border-radius:2px;padding:16px 20px;">'
                '<div style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:3px;'
                'color:' + C_GOLD + ';text-transform:uppercase;margin-bottom:12px;">'
                '&#10003; Verificaci&oacute;n &middot; Conteos del Dataset</div>'
                '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px;">'

                '<div style="text-align:center;background:#181410;padding:10px 8px;border-radius:2px;">'
                '<div style="font-family:DM Mono,monospace;font-size:8px;color:#786040;letter-spacing:2px;">N total</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:28px;color:#E8E0D0;line-height:1.1;">%d</div></div>' % n_total +

                '<div style="text-align:center;background:#181410;padding:10px 8px;border-radius:2px;">'
                '<div style="font-family:DM Mono,monospace;font-size:8px;color:#786040;letter-spacing:2px;">|A| eventos</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:28px;color:' + C_AMBER + ';line-height:1.1;">%d</div></div>' % n_a +

                '<div style="text-align:center;background:#181410;padding:10px 8px;border-radius:2px;">'
                '<div style="font-family:DM Mono,monospace;font-size:8px;color:#786040;letter-spacing:2px;">|B| evidencia</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:28px;color:' + C_GOLD + ';line-height:1.1;">%d</div></div>' % n_b +

                '<div style="text-align:center;background:#181410;padding:10px 8px;border-radius:2px;">'
                '<div style="font-family:DM Mono,monospace;font-size:8px;color:#786040;letter-spacing:2px;">|A&cap;B|</div>'
                '<div style="font-family:Bebas Neue,serif;font-size:28px;color:' + C_GREEN + ';line-height:1.1;">%d</div></div>' % n_ab +
                '</div>'

                '<div style="font-family:DM Mono,monospace;font-size:10px;color:#786040;line-height:2.0;">'
                'P(A)&nbsp;&nbsp;&nbsp;= %d / %d &nbsp;=&nbsp; <span style="color:#E8A020;">%.4f &nbsp;(%.2f%%)</span><br>' % (n_a, n_total, p_a, p_a*100) +
                'P(B)&nbsp;&nbsp;&nbsp;= %d / %d &nbsp;=&nbsp; <span style="color:#C8952A;">%.4f &nbsp;(%.2f%%)</span><br>' % (n_b, n_total, p_b, p_b*100) +
                'P(B|A) = %d / %d &nbsp;=&nbsp; <span style="color:#4488CC;">%.4f &nbsp;(%.2f%%)</span><br>' % (n_ab, n_a, p_b_a, p_b_a*100) +
                'P(A|B) = %.4f &times; %.4f / %.4f &nbsp;=&nbsp; <span style="color:#3DAA55;font-size:12px;">%.4f &nbsp;(%.2f%%)</span>' % (p_b_a, p_a, p_b, p_a_b, p_a_b*100) +
                '</div></div>',
                unsafe_allow_html=True
            )

        with p_col:
            bg_d = 'rgba(61,170,85,0.1)' if da >= 0 else 'rgba(204,34,0,0.1)'
            st.markdown(
                '<div class="post-hero">'
                '<div class="ph-eye">Resultado Bayesiano</div>'
                '<div class="ph-num" style="font-size:72px;">%.2f<span class="ph-unit"> %%</span></div>' % (p_a_b * 100) +
                '<div class="ph-delta" style="color:' + dc_c + ';background:' + bg_d + ';">'
                + arr + ' %.1f%% vs prior</div>' % abs(dr) +
                '<div class="ph-formula" style="text-align:left;font-size:10px;line-height:2.0;">'
                '<span style="color:#786040;">P(A|B) =</span> P(B|A) &middot; P(A) / P(B)<br>'
                '<span style="color:#786040;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=</span> '
                + ('%.4f &times; %.4f / %.4f' % (p_b_a, p_a, p_b)) +
                '<br>'
                '<span style="color:#786040;">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=</span> '
                + ('<span style="color:' + C_GREEN + ';font-size:13px;">%.4f</span>' % p_a_b) +
                '</div>'
                '<div class="ph-evidence" style="font-size:9px;line-height:1.9;">'
                '<span style="color:' + C_GOLD + ';">Objetivo&nbsp;&nbsp;</span> ' + str(st.session_state.target_col or '&mdash;') + '<br>'
                '<span style="color:' + C_GOLD + ';">Evento&nbsp;&nbsp;&nbsp;&nbsp;</span> ' + str(st.session_state.target_event or '&mdash;') + '<br>'
                '<span style="color:' + C_GOLD + ';">Evidencia</span> ' + (st.session_state.evidence_name or '&mdash;') +
                '</div></div>',
                unsafe_allow_html=True
            )

# ── SCREEN 3: MODEL ──────────────────────────────────────────
elif scr == 3:
    st.markdown('<div class="section-title">Modelo Predictivo &middot; Gaussian Naive Bayes</div>',
                unsafe_allow_html=True)
    res = compute_bayes()
    if res is None or not res['mv']:
        st.markdown(
            '<div class="info-panel"><div class="ip-tag">Modelo no disponible</div>'
            '<div class="ip-text">Se requieren columnas num&eacute;ricas continuas y al menos 2 clases. '
            'Las columnas binarias (0/1) se excluyen autom&aacute;ticamente para evitar data leakage.</div></div>',
            unsafe_allow_html=True)
    else:
        acc, sens, spec, prec = res['acc'], res['sens'], res['spec'], res['prec']
        tn, fp, fn, tp = res['tn'], res['fp'], res['fn'], res['tp']

        st.markdown(
            '<div class="model-strip">'
            '<div class="ms-cell ms-y"><div class="ms-lbl">Accuracy</div>'
            '<div class="ms-val" style="color:' + C_GOLD + ';">%.1f<span style="font-size:.4em;opacity:.6"> %%</span></div>' % (acc*100) +
            '<div class="ms-sub">Exactitud global</div></div>'
            '<div class="ms-cell ms-g"><div class="ms-lbl">Sensibilidad</div>'
            '<div class="ms-val" style="color:' + C_GREEN + ';">%.1f<span style="font-size:.4em;opacity:.6"> %%</span></div>' % (sens*100) +
            '<div class="ms-sub">Recall &middot; TPR</div></div>'
            '<div class="ms-cell ms-b"><div class="ms-lbl">Especificidad</div>'
            '<div class="ms-val" style="color:' + C_BLUE + ';">%.1f<span style="font-size:.4em;opacity:.6"> %%</span></div>' % (spec*100) +
            '<div class="ms-sub">True Negative Rate</div></div>'
            '<div class="ms-cell ms-a"><div class="ms-lbl">Precisi&oacute;n</div>'
            '<div class="ms-val" style="color:' + C_AMBER + ';">%.1f<span style="font-size:.4em;opacity:.6"> %%</span></div>' % ((prec or 0)*100) +
            '<div class="ms-sub">Positive Pred. Value</div></div>'
            '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="cm-strip">'
            '<div class="cm-cell"><div class="cm-lbl">Verdaderos Neg.</div>'
            '<div class="cm-val" style="color:' + C_GREEN + ';">%s</div></div>' % str(tn) +
            '<div class="cm-cell"><div class="cm-lbl">Falsos Positivos</div>'
            '<div class="cm-val" style="color:' + C_RED + ';">%s</div></div>' % str(fp) +
            '<div class="cm-cell"><div class="cm-lbl">Falsos Negativos</div>'
            '<div class="cm-val" style="color:' + C_AMBER + ';">%s</div></div>' % str(fn) +
            '<div class="cm-cell"><div class="cm-lbl">Verdaderos Pos.</div>'
            '<div class="cm-val" style="color:' + C_BLUE + ';">%s</div></div>' % str(tp) +
            '</div>',
            unsafe_allow_html=True
        )
        spec_note = ('Alta especificidad &mdash; buen control de falsas alarmas.'
                     if spec > 0.9 else 'Especificidad moderada &mdash; ajusta el umbral.')
        used = res.get('used_cols', [])
        st.markdown(
            '<div class="info-panel" style="margin-top:16px;">'
            '<div class="ip-tag">Lectura del Modelo</div>'
            '<div class="ip-text">El clasificador detecta '
            '<strong style="color:#E8E0D0;">%.2f%%</strong> de los eventos positivos ' % (sens*100) +
            'con exactitud global de <strong style="color:#E8E0D0;">%.2f%%</strong>. ' % (acc*100) +
            spec_note + '</div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div style="margin-top:10px;background:#0D0A07;border:1px solid rgba(200,149,42,0.12);'
            'border-left:3px solid #786040;border-radius:2px;padding:14px 18px;">'
            '<div style="font-family:DM Mono,monospace;font-size:9px;letter-spacing:2px;'
            'color:#786040;text-transform:uppercase;margin-bottom:8px;">Metodolog&iacute;a &middot; Resustitución (Listing 9, pág. 17)</div>'
            '<div style="font-family:DM Mono,monospace;font-size:10px;color:#504438;line-height:1.9;">'
            'nb = GaussianNB()<br>'
            'nb.fit(features, y) &nbsp;&nbsp;<span style="color:#3A3228;"># Estimación µₖ y σₖ</span><br>'
            'y_pred = nb.predict(features) &nbsp;&nbsp;<span style="color:#3A3228;"># Resustitución</span><br>'
            '<br>'
            '<span style="color:#786040;">Features usadas (%d):</span> ' % len(used) +
            '<span style="color:#504438;word-break:break-all;">' + (', '.join(used) if used else '&mdash;') + '</span>'
            '<br><span style="color:#3A3228;font-size:9px;">Columnas binarias (0/1) excluidas para evitar data leakage</span>'
            '</div></div>',
            unsafe_allow_html=True
        )


# ── SCREEN 4: CHARTS ─────────────────────────────────────────
elif scr == 4:
    st.markdown('<div class="section-title">Telemetr&iacute;a Visual &middot; Estad&iacute;stica Exploratoria</div>',
                unsafe_allow_html=True)
    res = compute_bayes()
    df_ = st.session_state.df
    ec  = st.session_state.evidence_col
    tc  = st.session_state.target_col
    te  = st.session_state.target_event
    dc_ = st.session_state.datetime_cols
    nc  = st.session_state.numeric_cols

    if df_ is None or ec is None:
        st.markdown(
            '<div class="info-panel"><div class="ip-tag">Sin datos</div>'
            '<div class="ip-text">Configura en DATOS (marcha 2) primero.</div></div>',
            unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2, gap="medium")
        c3, c4 = st.columns(2, gap="medium")

        with c1:
            fig, ax = plt.subplots(figsize=(7, 3.8), facecolor='#060505')
            ax_s(ax)
            de = df_[ec].dropna()
            if de.dtype in ['int64', 'float64']:
                nb_ = min(30, max(10, len(de)//8))
                _, _, patches = ax.hist(de, bins=nb_, edgecolor='none', zorder=2)
                for i, p_ in enumerate(patches):
                    t = i / max(len(patches)-1, 1)
                    p_.set_facecolor((0.78, 0.56 - 0.38*t, 0.16, 0.9))
                mv_ = de.mean()
                ax.axvline(mv_, color='#E8E0D0', lw=1, ls='--', alpha=.3, zorder=3)
            else:
                vc = de.value_counts()
                ax.barh(range(len(vc)), vc.values,
                        color=[C_GOLD if i==0 else C_DIM for i in range(len(vc))],
                        height=.55, edgecolor='none', zorder=2)
                ax.set_yticks(range(len(vc)))
                ax.set_yticklabels([str(v)[:14].upper() for v in vc.index], fontsize=7)
            ax.set_title('DISTRIBUCION  ' + ec.upper()[:26],
                         fontsize=8, fontweight='bold', color=C_GOLD, pad=10, loc='left')
            plt.tight_layout(pad=1.1)
            st.pyplot(fig); plt.close(fig)

        with c2:
            if res:
                fig2, ax2 = plt.subplots(figsize=(7, 3.8), facecolor='#060505')
                ax_s(ax2)
                labs = ['P(A)\nBASE', 'P(B)\nEVID.', 'P(B|A)\nVEROS.', 'P(A|B)\nPOST.']
                vals = [res['p_a'], res['p_b'], res['p_b_a'], res['p_a_b']]
                cs   = [C_DIM, '#1E1A10', '#101820', C_GOLD]
                bars = ax2.bar(labs, vals, width=.52, color=cs, edgecolor='none', zorder=3)
                bars[3].set_color(C_GOLD)
                for b_, v_ in zip(bars, vals):
                    ax2.text(b_.get_x()+b_.get_width()/2, v_+.007, '%.1f%%' % (v_*100),
                             ha='center', va='bottom', fontsize=9, fontweight='bold',
                             color=C_CREAM, fontfamily='monospace')
                da2 = res['p_a_b'] - res['p_a']
                if abs(da2) > .002:
                    ca = C_GREEN if da2 >= 0 else C_RED
                    ax2.annotate('', xy=(3, res['p_a_b']), xytext=(0, res['p_a']),
                                 arrowprops=dict(arrowstyle='->', color=ca, lw=1.4,
                                                connectionstyle='arc3,rad=-0.28'))
                    ax2.text(1.5, (res['p_a']+res['p_a_b'])/2+.022,
                             'D %+.1f%%' % (da2*100),
                             ha='center', color=ca, fontsize=8, fontfamily='monospace', fontweight='bold')
                ax2.set_ylim(0, min(1.2, max(vals)+.18))
                ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '%.0f%%' % (x*100)))
                ax2.set_title('WATERFALL BAYESIANO',
                              fontsize=8, fontweight='bold', color=C_GOLD, pad=10, loc='left')
                plt.tight_layout(pad=1.1)
                st.pyplot(fig2); plt.close(fig2)

        with c3:
            fig3, ax3 = plt.subplots(figsize=(7, 3.8), facecolor='#060505')
            ax_s(ax3)
            if dc_ and df_[ec].dtype in ['int64', 'float64']:
                ts = df_[[dc_[0], ec]].dropna().sort_values(dc_[0]).reset_index(drop=True)
                xi = np.arange(len(ts)); yv = ts[ec].values
                ax3.fill_between(xi, yv, alpha=.07, color=C_GOLD, zorder=1)
                ax3.plot(xi, yv, color=C_GOLD, lw=1.5, alpha=.9, zorder=2)
                ax3.scatter([xi[-1]], [yv[-1]], color=C_AMBER, s=45, zorder=5, edgecolors='none')
                ax3.set_title('SERIE TEMPORAL  ' + ec.upper()[:22],
                              fontsize=8, fontweight='bold', color=C_GOLD, pad=10, loc='left')
            else:
                vc2 = df_[tc].value_counts(normalize=True)
                clrs = [C_GOLD if c == te else C_DIM for c in vc2.index]
                bs3 = ax3.bar(range(len(vc2)), vc2.values, width=.52, color=clrs, edgecolor='none', zorder=2)
                for b3, v3 in zip(bs3, vc2.values):
                    ax3.text(b3.get_x()+b3.get_width()/2, v3+.007, '%.1f%%' % (v3*100),
                             ha='center', va='bottom', fontsize=9, fontweight='bold',
                             color=C_CREAM, fontfamily='monospace')
                ax3.set_xticks(range(len(vc2)))
                ax3.set_xticklabels([str(c).upper()[:14] for c in vc2.index], fontsize=7)
                ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: '%.0f%%' % (x*100)))
                ax3.set_title('PROPORCION CLASES  ' + tc.upper()[:20],
                              fontsize=8, fontweight='bold', color=C_GOLD, pad=10, loc='left')
            plt.tight_layout(pad=1.1)
            st.pyplot(fig3); plt.close(fig3)

        with c4:
            fig4, ax4 = plt.subplots(figsize=(7, 3.8), facecolor='#060505')
            ax4.set_facecolor('#080707')
            for sp in ax4.spines.values(): sp.set_color('#2A2218')
            ax4.tick_params(colors='#3A3228')
            if res and res['mv'] and res['tn'] is not None:
                cm_d = np.array([[res['tn'], res['fp']], [res['fn'], res['tp']]], dtype=float)
                cmap_ = LinearSegmentedColormap.from_list('gt3lth', ['#080707','#140E04','#785020',C_GOLD])
                ax4.imshow(cm_d, cmap=cmap_, aspect='auto')
                for i in range(2):
                    for j in range(2):
                        v_ = int(cm_d[i, j])
                        bright = v_ >= cm_d.max() * .3
                        ax4.text(j, i, str(v_), ha='center', va='center',
                                 fontsize=22, fontweight='bold', fontfamily='monospace',
                                 color=C_CREAM if bright else '#2A2218')
                ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
                ax4.set_xticklabels(['PRED NEG', 'PRED POS'], fontsize=7, fontweight='bold')
                ax4.set_yticklabels(['REAL NEG', 'REAL POS'], fontsize=7, fontweight='bold')
            else:
                ax4.text(.5, .5, 'DATOS\nINSUFICIENTES\nPARA EL MODELO',
                         ha='center', va='center', fontsize=11, color='#2A2218',
                         fontfamily='monospace', transform=ax4.transAxes, linespacing=1.8)
            ax4.set_title('MATRIZ DE CONFUSION',
                          fontsize=8, fontweight='bold', color=C_GOLD, pad=10, loc='left')
            plt.tight_layout(pad=1.1)
            st.pyplot(fig4); plt.close(fig4)

# ── SCREEN 5: AI ─────────────────────────────────────────────
elif scr == 5:
    st.markdown('<div class="section-title">Motor Generativo &middot; Gemini 2.5 Flash &middot; S&iacute;ntesis Ejecutiva</div>',
                unsafe_allow_html=True)
    res = compute_bayes()
    ai_l, ai_r = st.columns([1.3, 1], gap="large")

    with ai_l:
        st.markdown('<div class="card-tag" style="margin-bottom:10px;">Autenticaci&oacute;n Gemini 2.5</div>',
                    unsafe_allow_html=True)
        api_key = st.text_input("GEMINI API KEY", type="password", placeholder="AIzaSy...")
        run_btn = st.button("GENERAR SINTESIS EJECUTIVA")

        if run_btn:
            if not api_key:
                st.error("Ingresa una API key valida de Google Gemini.")
            elif res is None:
                st.error("Configura el analisis en DATOS (marcha 2) primero.")
            else:
                with st.spinner("Generando sintesis ejecutiva..."):
                    try:
                        genai.configure(api_key=api_key)
                        gm = genai.GenerativeModel('gemini-2.5-flash')
                        acc_s = ('%.2f%%' % (res['acc']*100)) if res['mv'] else 'N/A'
                        dr_ = ((res['p_a_b']-res['p_a'])/res['p_a']*100) if res['p_a'] > 0 else 0
                        prompt = (
                            "Eres un cientifico de datos senior experto en estadistica bayesiana.\n"
                            "Escribe un resumen ejecutivo preciso, tecnico y accionable en maximo 160 palabras.\n\n"
                            "Analisis bayesiano:\n"
                            "- Variable objetivo: %s | Evento: %s\n"
                            "- Evidencia evaluada: %s\n"
                            "- P(A) previa: %.3f (%.1f%%)\n"
                            "- P(A|B) posterior: %.3f (%.1f%%)\n"
                            "- Cambio relativo: %+.1f%%\n"
                            "- Modelo GNB: %s\n\n"
                            "Que implica este resultado? La evidencia es util? Que accion recomiendas?\n"
                            "Se directo, tecnico y conciso."
                        ) % (
                            st.session_state.target_col, st.session_state.target_event,
                            st.session_state.evidence_name,
                            res['p_a'], res['p_a']*100,
                            res['p_a_b'], res['p_a_b']*100,
                            dr_,
                            ('Accuracy=' + acc_s + ', Sensibilidad=%.1f%%' % (res['sens']*100)) if res['mv'] else 'no entrenado'
                        )
                        resp = gm.generate_content(prompt)
                        st.balloons()
                        st.markdown(
                            '<div class="info-panel" style="margin-top:16px;border-left-color:' + C_GREEN + ';">'
                            '<div class="ip-tag" style="color:' + C_GREEN + ';">Sintesis Gemini &middot; Completada</div>'
                            '<div class="ip-text" style="font-size:13px;line-height:1.8;color:#A0B8A0;">'
                            + resp.text.replace('\n', '<br>') +
                            '</div></div>',
                            unsafe_allow_html=True
                        )
                    except Exception as e:
                        st.error("Error Gemini: " + str(e))

    with ai_r:
        st.markdown('<div class="card-tag" style="margin-bottom:10px;">Resumen del An&aacute;lisis</div>',
                    unsafe_allow_html=True)
        if res:
            dr_val = ((res['p_a_b']-res['p_a'])/res['p_a']*100) if res['p_a'] > 0 else 0
            rows = [
                ("Objetivo",    str(st.session_state.target_col or '&mdash;'),            C_GOLD),
                ("Evento",      str(st.session_state.target_event or '&mdash;'),          C_AMBER),
                ("Evidencia",   str(st.session_state.evidence_name or '&mdash;'),         C_BLUE),
                ("P(A) Prior",  "%.4f (%.1f%%)" % (res['p_a'], res['p_a']*100),          C_GREEN),
                ("P(A|B) Post", "%.4f (%.1f%%)" % (res['p_a_b'], res['p_a_b']*100),      C_GREEN),
                ("Delta",       "%+.1f%%" % dr_val,                                       C_GREEN if dr_val >= 0 else C_RED),
                ("Accuracy",    ('%.2f%%' % (res['acc']*100)) if res['mv'] else 'N/A',    C_GOLD if res['mv'] else C_DIM),
            ]
            for lbl, val, col in rows:
                st.markdown(
                    '<div style="display:flex;justify-content:space-between;align-items:flex-start;'
                    'padding:10px 0;border-bottom:1px solid rgba(255,255,255,0.03);">'
                    '<span style="font-family:DM Mono,monospace;font-size:7px;'
                    'letter-spacing:2px;color:#3A3228;text-transform:uppercase;'
                    'flex-shrink:0;padding-right:12px;">' + lbl + '</span>'
                    '<span style="font-family:DM Sans,sans-serif;font-size:12px;'
                    'font-weight:600;color:' + col + ';text-align:right;'
                    'word-break:break-all;">' + val + '</span>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div class="info-panel"><div class="ip-tag">Sin datos</div>'
                '<div class="ip-text">Configura el analisis en DATOS primero.</div></div>',
                unsafe_allow_html=True
            )

st.markdown('</div>', unsafe_allow_html=True)
