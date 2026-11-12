import math
import random
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="NeuroMotion", page_icon="🧠", layout="wide",
                   initial_sidebar_state="collapsed")
ROOT = Path(__file__).parent
MODEL = ROOT / "best_emg_gesture_model.joblib"
DEMO = ROOT / "demo_features.csv"

STATS = ("MAV", "RMS", "WL", "ZC")
FEATURES = [f"ch{i}_{f}" for i in range(1, 9) for f in STATS]
GESTURES = {
    1: "Hand at rest", 2: "Hand clenched in a fist", 3: "Wrist flexion",
    4: "Wrist extension", 5: "Radial deviation", 6: "Ulnar deviation",
}
EMOJI = {1: "🖐️", 2: "✊", 3: "🫳", 4: "🫴", 5: "👈", 6: "👉"}
COLORS = {
    "Hand at rest": "#8b97b5", "Hand clenched in a fist": "#7c6cff",
    "Wrist flexion": "#14b8d4", "Wrist extension": "#3b9cff",
    "Radial deviation": "#f5a524", "Ulnar deviation": "#f0587a",
}
STAT_INFO = {
    "MAV": ("Mean absolute value", "How strong the muscle signal is on average."),
    "RMS": ("Root mean square", "Signal power, weighted toward big spikes."),
    "WL": ("Waveform length", "How much the signal wiggles inside a window."),
    "ZC": ("Zero crossings", "How often the signal flips above and below zero."),
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&display=swap');
:root{--base:#e3e8f1;--hi:#ffffff;--lo:#b9c3d6;--ink:#1d2640;--muted:#65729a;
--indigo:#5b6cff;--mint:#22c9a0;--amber:#f5a524;--rose:#f0587a}
.stApp,.stApp p,.stApp li,.stApp label,.stApp button,.stApp h1,.stApp h2,.stApp h3,.stApp input{font-family:Sora,sans-serif}
:where(.stApp) :where(p,li,label,h1,h2,h3,span,div,small){color:var(--ink)}
.stApp{background:var(--base)}
.stApp::before{content:"";position:fixed;inset:-20%;z-index:0;pointer-events:none;filter:blur(36px);
background:radial-gradient(circle at 18% 14%,rgba(91,108,255,.30),transparent 32%),
radial-gradient(circle at 82% 8%,rgba(34,201,160,.24),transparent 30%),
radial-gradient(circle at 72% 88%,rgba(240,88,170,.16),transparent 30%);
animation:drift 30s ease-in-out infinite alternate}
@keyframes drift{to{transform:translate3d(4%,3%,0) rotate(6deg) scale(1.08)}}
@media (prefers-reduced-motion:reduce){.stApp::before{animation:none}*{transition:none!important}}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{position:relative;z-index:1;padding-top:2rem;max-width:1380px}

/* surfaces */
.neo{background:var(--base);border-radius:26px;padding:1.2rem 1.4rem;
box-shadow:12px 12px 26px var(--lo),-12px -12px 26px var(--hi)}
.glass{background:rgba(255,255,255,.45);backdrop-filter:blur(22px) saturate(1.4);
-webkit-backdrop-filter:blur(22px) saturate(1.4);border:1px solid rgba(255,255,255,.8);border-radius:34px;
box-shadow:0 30px 60px -24px rgba(60,80,140,.38),inset 0 1px 0 rgba(255,255,255,.95)}
.tile{transition:transform .25s,box-shadow .25s}
.tile:hover{transform:translateY(-5px);box-shadow:16px 22px 34px var(--lo),-12px -12px 26px var(--hi)}
.muted{color:var(--muted)!important}

/* hero */
.hero{display:flex;align-items:center;justify-content:space-between;gap:1.5rem;padding:1.8rem 2.2rem;margin-bottom:1.2rem}
.hero h1{font-size:2.8rem;font-weight:800;letter-spacing:-.03em;margin:0;padding:0}
.hero p{margin:.4rem 0 0;color:var(--muted);font-size:1.05rem}
.chip{display:inline-flex;align-items:center;gap:.5rem;margin-top:1rem;padding:.4rem .9rem;border-radius:999px;
background:var(--base);font-size:.8rem;font-weight:600;
box-shadow:inset 3px 3px 6px var(--lo),inset -3px -3px 6px var(--hi)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mint);box-shadow:0 0 10px var(--mint)}
.orb{flex:none;width:120px;height:120px;border-radius:50%;display:grid;place-items:center;font-size:3.4rem;
background:var(--base);box-shadow:10px 10px 22px var(--lo),-10px -10px 22px var(--hi)}

/* metrics */
.metric{display:flex;gap:1rem;align-items:center}
.ico{flex:none;width:52px;height:52px;border-radius:50%;display:grid;place-items:center;font-size:1.4rem;
box-shadow:inset 4px 4px 8px var(--lo),inset -4px -4px 8px var(--hi)}
.metric small{color:var(--muted)!important;font-weight:600;font-size:.78rem}
.metric h2{margin:.1rem 0;font-size:1.5rem;font-weight:800;padding:0}
.metric .muted{font-size:.78rem}

/* prediction */
.pred{display:flex;align-items:center;gap:1.3rem}
.bubble{flex:none;width:96px;height:96px;border-radius:50%;display:grid;place-items:center;font-size:3rem;
box-shadow:inset 6px 6px 12px var(--lo),inset -6px -6px 12px var(--hi)}
.pred-txt{flex:1;min-width:0}
.pred-txt small{color:var(--muted)!important;font-weight:600}
.pred-txt h2{margin:.1rem 0 .5rem;font-size:1.55rem;font-weight:800;line-height:1.15;padding:0}
.ring{flex:none;width:104px;height:104px;border-radius:50%;display:grid;place-items:center;
background:conic-gradient(var(--indigo) calc(var(--p)*1%),#cdd6e8 0);
box-shadow:6px 6px 12px var(--lo),-6px -6px 12px var(--hi)}
.ring-in{width:76px;height:76px;border-radius:50%;background:var(--base);display:flex;flex-direction:column;
align-items:center;justify-content:center;box-shadow:inset 4px 4px 8px var(--lo),inset -4px -4px 8px var(--hi)}
.ring-in b{font-size:1.15rem}.ring-in span{font-size:.6rem;color:var(--muted)!important}
.badge{display:inline-block;padding:.3rem .8rem;border-radius:999px;font-size:.78rem;font-weight:600;
background:var(--base);box-shadow:inset 3px 3px 6px var(--lo),inset -3px -3px 6px var(--hi)}
.badge.ok{color:#0e8f72!important}.badge.no{color:#c47a00!important}
.notice{border-radius:20px;padding:1rem 1.2rem;font-size:.9rem;background:var(--base);
box-shadow:inset 6px 6px 12px var(--lo),inset -6px -6px 12px var(--hi)}
.notice b{color:var(--indigo)!important}
.feat h3{margin:.2rem 0;font-size:1.3rem;font-weight:800;padding:0;color:var(--indigo)!important}
.feat b{font-size:.85rem}.feat p{margin:.3rem 0 0;font-size:.82rem;color:var(--muted)!important}
.neo ul{margin:.4rem 0 0;padding-left:1.1rem}.neo li{margin:.35rem 0;font-size:.9rem}

/* streamlit widgets */
div.stButton>button,div.stDownloadButton>button{border:0!important;border-radius:16px;background:var(--base)!important;
color:var(--ink)!important;font-weight:600;padding:.6rem 1.2rem;
box-shadow:6px 6px 12px var(--lo),-6px -6px 12px var(--hi);transition:all .2s}
div.stButton>button:hover,div.stDownloadButton>button:hover{transform:translateY(-2px);color:var(--indigo)!important}
div.stButton>button:active{transform:none;box-shadow:inset 4px 4px 8px var(--lo),inset -4px -4px 8px var(--hi)}
div.stButton>button[kind="primary"]{background:linear-gradient(135deg,#7482ff,#4a5af0)!important;color:#fff!important;
box-shadow:6px 8px 16px rgba(74,90,240,.35),-4px -4px 10px var(--hi)}
div[role="radiogroup"]{gap:.7rem!important;flex-wrap:wrap}
div[role="radiogroup"]>label{background:var(--base);border-radius:999px;padding:.5rem 1.15rem!important;margin:0!important;
cursor:pointer;box-shadow:5px 5px 10px var(--lo),-5px -5px 10px var(--hi);transition:all .2s}
div[role="radiogroup"]>label>div:first-child{display:none}
div[role="radiogroup"]>label:hover{transform:translateY(-2px)}
div[role="radiogroup"]>label:has(input:checked){box-shadow:inset 4px 4px 8px var(--lo),inset -4px -4px 8px var(--hi)}
div[role="radiogroup"]>label:has(input:checked) p{color:var(--indigo)!important;font-weight:700}
[data-testid="stMetric"]{background:var(--base);border-radius:22px;padding:1rem 1.3rem;
box-shadow:8px 8px 16px var(--lo),-8px -8px 16px var(--hi)}
[data-testid="stFileUploaderDropzone"]{background:var(--base);border:0;border-radius:22px;
box-shadow:inset 6px 6px 12px var(--lo),inset -6px -6px 12px var(--hi)}
[data-testid="stAlert"]{border-radius:18px;border:0;background:var(--base)!important;
box-shadow:6px 6px 12px var(--lo),-6px -6px 12px var(--hi)}
[data-testid="stAlert"] *{color:var(--ink)!important}
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]){border-left:6px solid var(--rose)}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){border-left:6px solid var(--amber)}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){border-left:6px solid var(--mint)}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){border-left:6px solid var(--indigo)}
[data-testid="stDataFrame"]{border-radius:18px;overflow:hidden;box-shadow:6px 6px 12px var(--lo),-6px -6px 12px var(--hi)}
hr{border-color:transparent!important;height:2px;background:linear-gradient(90deg,transparent,var(--lo),transparent)!important}
/* readable text on any Streamlit theme (light or dark) */
html,.stApp{color-scheme:light}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--ink)!important}
.stApp [data-testid="stMarkdownContainer"] :is(p,li,span,small,b,div,h1,h2,h3,h4){color:var(--ink)!important}
.stApp [data-testid="stWidgetLabel"] *,.stApp [data-baseweb="tab"] *,.stApp [data-testid="stMetric"] *,
.stApp [data-testid="stFileUploaderDropzone"] *{color:var(--ink)!important}
.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] *{color:var(--muted)!important}
.stApp [data-testid="stFileUploaderDropzone"] button{color:var(--ink)!important;background:var(--base)!important}
.stApp [data-testid="stMarkdownContainer"] :is(.muted,.hero p,.feat p,.metric small,.pred-txt small,.ring-in span){color:var(--muted)!important}
.stApp [data-testid="stMarkdownContainer"] :is(.feat h3,.notice b){color:var(--indigo)!important}
.stApp [data-testid="stMarkdownContainer"] .badge.ok{color:#0b7a62!important}
.stApp [data-testid="stMarkdownContainer"] .badge.no{color:#a85f00!important}
.stApp [data-testid="stAlert"]{background:var(--base)!important}
.stApp [data-testid="stAlert"] *{background:transparent!important;color:var(--ink)!important}
.stApp [data-testid="stFileUploader"] section,.stApp [data-testid="stFileUploaderDropzone"]{background:var(--base)!important}
.stApp [data-testid="stFileUploader"] :is(button,[data-testid^="stBaseButton"]){background:var(--base)!important;
background-color:var(--base)!important;border:0!important;border-radius:14px!important;color:var(--ink)!important;
box-shadow:5px 5px 10px var(--lo),-5px -5px 10px var(--hi)!important}
.stApp [data-testid="stFileUploader"] :is(small,span,p,svg){color:var(--muted)!important;fill:var(--muted)!important}
.stApp [data-testid="stFileUploader"] button *{color:var(--ink)!important;fill:var(--ink)!important}
.stApp [data-testid="stSidebar"]{display:none}
@media (max-width:760px){.hero{flex-direction:column;align-items:flex-start}.orb{display:none}.pred{flex-wrap:wrap}}
</style>
""", unsafe_allow_html=True)


# ---------- helpers ----------
def metric(icon, label, value, detail):
    st.markdown(
        f'<div class="neo metric tile"><div class="ico">{icon}</div><div><small>{label}</small>'
        f'<h2>{value}</h2><span class="muted">{detail}</span></div></div>',
        unsafe_allow_html=True)


@st.cache_resource
def get_model(path, mtime):
    return joblib.load(path)


@st.cache_data(show_spinner="Reading every recorded window...")
def all_preds(_model, mtime, df):
    return np.asarray(_model.predict(df[FEATURES])).astype(int)


def valid(df):
    return [c for c in FEATURES if c not in df.columns]


def predict_row(model, row):
    x = pd.DataFrame([row[FEATURES].to_dict()], columns=FEATURES)
    pred = int(model.predict(x)[0])
    probs = None
    if hasattr(model, "predict_proba"):
        probs = (model.classes_, model.predict_proba(x)[0])
    return pred, probs


def pred_card(pred, probs, extra=""):
    name = GESTURES.get(pred, str(pred))
    ring = ""
    if probs is not None:
        conf = float(np.max(probs[1]))
        ring = (f'<div class="ring" style="--p:{conf * 100:.0f}"><div class="ring-in">'
                f'<b>{conf:.0%}</b><span>confidence</span></div></div>')
    st.markdown(
        f'<div class="neo pred"><div class="bubble">{EMOJI.get(pred, "🧠")}</div>'
        f'<div class="pred-txt"><small>The model sees</small><h2>{name}</h2>{extra}</div>{ring}</div>',
        unsafe_allow_html=True)


def clear_plot(fig, height):
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#1d2640", family="Sora"), margin=dict(l=10, r=10, t=20, b=10),
                      modebar=dict(bgcolor="rgba(0,0,0,0)", color="#65729a", activecolor="#5b6cff"))
    return fig


def odds_chart(probs):
    classes, p = probs
    d = pd.DataFrame({"Gesture": [GESTURES.get(int(c), str(c)) for c in classes], "Probability": p})
    d = d.sort_values("Probability")
    fig = px.bar(d, x="Probability", y="Gesture", orientation="h", color="Gesture",
                 color_discrete_map=COLORS, text=d["Probability"].map(lambda v: f"{v:.1%}"))
    clear_plot(fig, 300)
    fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.2], tickformat=".0%", gridcolor="#cdd6e8"),
                      yaxis=dict(title="", gridcolor="rgba(0,0,0,0)"))
    fig.update_traces(textposition="outside", marker_cornerradius=8)
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("These are the classifier's probability estimates, not guaranteed certainty.")


def profile_chart(row):
    prof = [{"Channel": f"CH {i}", **{s: row[f"ch{i}_{s}"] for s in STATS}} for i in range(1, 9)]
    long = pd.DataFrame(prof).melt(id_vars="Channel", var_name="Feature", value_name="Value")
    fig = px.line(long, x="Channel", y="Value", color="Feature", markers=True,
                  color_discrete_sequence=["#5b6cff", "#22c9a0", "#f5a524", "#f0587a"])
    clear_plot(fig, 330)
    fig.update_layout(xaxis=dict(gridcolor="#cdd6e8"), yaxis=dict(gridcolor="#cdd6e8"))
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption("Extracted feature values, not raw EMG voltage waveforms.")


def _mix(a, b, t):
    return tuple(round(a[k] + (b[k] - a[k]) * t) for k in range(3))


def armband(row, feature, ref_df):
    """Interactive 3D-tilting armband: each sensor segment glows with its feature value."""
    cols = [f"ch{i}_{feature}" for i in range(1, 9)]
    vals = row[cols].astype(float).values
    if ref_df is not None and not valid(ref_df):
        ref = ref_df[cols].quantile(0.95).values.astype(float)
    else:
        ref = np.full(8, max(vals.max(), 1e-9))
    ref = np.where(ref == 0, 1, ref)
    norm = np.clip(vals / ref, 0, 1)
    cool, indigo, mint = (207, 216, 234), (91, 108, 255), (34, 201, 160)

    def pt(r, a):
        return f"{150 + r * math.cos(a):.1f},{150 + r * math.sin(a):.1f}"

    segs = ""
    for i in range(8):
        a0, a1 = math.radians(i * 45 - 90 + 2.5), math.radians((i + 1) * 45 - 90 - 2.5)
        am = (a0 + a1) / 2
        v = float(norm[i])
        rgb = _mix(cool, indigo, v * 2) if v < 0.5 else _mix(indigo, mint, (v - 0.5) * 2)
        color = f"rgb{rgb}"
        glow = f"drop-shadow(0 0 {v * 12:.0f}px {color})" if v > 0.35 else ""
        d = (f"M{pt(122, a0)} A122,122 0 0 1 {pt(122, a1)} L{pt(74, a1)} A74,74 0 0 0 {pt(74, a0)} Z")
        tx, ty = pt(98, am).split(",")
        txt = "#fff" if v > 0.55 else "#1d2640"
        segs += (f'<g class="seg" data-ch="{i + 1}" data-v="{vals[i]:.3g}">'
                 f'<path d="{d}" fill="{color}" style="filter:drop-shadow(4px 4px 5px #b9c3d6) '
                 f'drop-shadow(-4px -4px 5px #fff) {glow}"/>'
                 f'<text x="{tx}" y="{ty}" fill="{txt}" text-anchor="middle" dominant-baseline="middle" '
                 f'font-size="15" font-weight="700">{i + 1}</text></g>')

    html = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&display=swap');
html,body{margin:0;background:transparent;font-family:Sora,sans-serif;color:#1d2640}
.panel{margin:14px;height:352px;border-radius:30px;background:#e3e8f1;position:relative;display:grid;
place-items:center;perspective:900px;box-shadow:12px 12px 26px #b9c3d6,-12px -12px 26px #fff}
.stage{transition:transform .15s ease-out;transform-style:preserve-3d}
svg{width:290px;height:290px;overflow:visible;display:block}
.seg{transition:transform .2s;cursor:pointer;transform-origin:150px 150px}
.seg:hover{transform:scale(1.07)}
.hub{filter:drop-shadow(-3px -3px 4px #fff) drop-shadow(3px 3px 4px #b9c3d6)}
.cap{position:absolute;bottom:12px;left:0;right:0;text-align:center;font-size:11.5px;color:#65729a}
</style>
<div class="panel" id="p"><div class="stage" id="s">
<svg viewBox="0 0 300 300">__SEGS__
<circle class="hub" cx="150" cy="150" r="58" fill="#e3e8f1"/>
<text id="t1" x="150" y="144" text-anchor="middle" font-size="19" font-weight="800">__FEAT__</text>
<text id="t2" x="150" y="166" text-anchor="middle" font-size="11" fill="#65729a">across 8 sensors</text>
</svg></div>
<div class="cap">Hover a sensor to read it. Move your mouse to tilt the band.</div></div>
<script>
const p=document.getElementById('p'),s=document.getElementById('s'),t1=document.getElementById('t1'),t2=document.getElementById('t2');
const still=window.matchMedia('(prefers-reduced-motion: reduce)').matches;
if(!still){p.addEventListener('mousemove',e=>{const r=p.getBoundingClientRect();
const x=(e.clientX-r.left)/r.width-.5,y=(e.clientY-r.top)/r.height-.5;
s.style.transform='rotateX('+(-y*30)+'deg) rotateY('+(x*30)+'deg)'});
p.addEventListener('mouseleave',()=>{s.style.transform=''})}
document.querySelectorAll('.seg').forEach(g=>{
g.addEventListener('mouseenter',()=>{t1.textContent=g.dataset.v;t2.textContent='sensor '+g.dataset.ch});
g.addEventListener('mouseleave',()=>{t1.textContent='__FEAT__';t2.textContent='across 8 sensors'})});
</script>"""
    components.html(html.replace("__SEGS__", segs).replace("__FEAT__", feature), height=385)


# ---------- header ----------
st.markdown(
    '<div class="glass hero"><div><h1>NeuroMotion</h1>'
    '<p>Eight forearm sensors go in, one hand gesture comes out.</p>'
    '<span class="chip"><span class="dot"></span>Extra Trees model loaded, 32 inputs, 6 gestures</span></div>'
    '<div class="orb">🧠</div></div>', unsafe_allow_html=True)

if not MODEL.exists():
    st.error("Model missing. Add best_emg_gesture_model.joblib beside app.py in your GitHub repository.")
    st.stop()
try:
    model = get_model(str(MODEL), MODEL.stat().st_mtime)
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.info("Check that scikit-learn and joblib versions match the training environment.")
    st.stop()
mtime = MODEL.stat().st_mtime

demo = pd.read_csv(DEMO) if DEMO.exists() else None
demo_ok = demo is not None and not valid(demo) and not demo.empty

page = st.radio("Navigate", ["Overview", "Gesture replay", "Upload & predict", "Model insights"],
                horizontal=True, label_visibility="collapsed", key="page")
st.write("")

# ---------- pages ----------
if page == "Overview":
    c = st.columns(4)
    with c[0]: metric("🌲", "Model", "Extra Trees", "Trained pipeline")
    with c[1]: metric("🎚️", "Input features", "32", "4 features from 8 channels")
    with c[2]: metric("🖐️", "Gesture classes", "6", "Recognized movements")
    with c[3]: metric("🎯", "Test accuracy", "84.63%", "On held-out subjects")
    st.write("")

    l, r = st.columns([1.15, 1], gap="large")
    with l:
        st.subheader("From muscle activity to gesture")
        st.write("Each recorded window holds four numbers per sensor. The model reads all 32 "
                 "and picks the most likely of six gestures.")
        f = st.columns(2)
        for n, s in enumerate(STATS):
            with f[n % 2]:
                st.markdown(f'<div class="neo feat tile"><h3>{s}</h3><b>{STAT_INFO[s][0]}</b>'
                            f'<p>{STAT_INFO[s][1]}</p></div>', unsafe_allow_html=True)
                st.write("")
        st.markdown('<div class="notice"><b>Demo mode.</b> This hosted prototype replays recorded '
                    'feature windows. It does not connect to a MYO armband or read live EMG signals.</div>',
                    unsafe_allow_html=True)
    with r:
        st.subheader("Try a gesture")
        if not demo_ok or "actual_gesture" not in demo.columns:
            st.info("Add demo_features.csv with an actual_gesture column to enable this.")
        else:
            g = st.radio("Pick a gesture", list(GESTURES), format_func=lambda k: f"{EMOJI[k]} {GESTURES[k]}",
                         key="try_g", label_visibility="collapsed")
            again = st.button("Try another recorded window")
            actual_all = demo["actual_gesture"].astype(int).values
            pool = np.flatnonzero(actual_all == g)
            if len(pool) == 0:
                st.warning("No recorded windows for this gesture in the demo file.")
            else:
                if again or st.session_state.get("try_for") != g:
                    st.session_state.try_i = int(random.choice(pool))
                    st.session_state.try_for = g
                row = demo.iloc[st.session_state.try_i]
                pred, probs = predict_row(model, row)
                ok = pred == g
                badge = ('<span class="badge ok">Got it right</span>' if ok else
                         f'<span class="badge no">Mixed it up with {GESTURES.get(pred, pred)}</span>')
                pred_card(pred, probs, badge)
                st.write("")
                armband(row, "RMS", demo)
    st.divider()
    if demo is None:
        st.info("Add demo_features.csv to the repository root to enable replay.")
    elif valid(demo):
        st.error("Demo CSV is missing columns: " + ", ".join(valid(demo)))
    else:
        st.success(f"Held-out test dataset loaded: {len(demo):,} windows.")

elif page == "Gesture replay":
    st.subheader("Recorded gesture replay")
    st.caption("Step through held-out feature windows and watch the sensors light up.")
    if demo is None:
        st.warning("demo_features.csv is missing. Add it to the repository or use Upload & predict.")
    elif valid(demo):
        st.error("Missing feature columns: " + ", ".join(valid(demo)))
    elif demo.empty:
        st.warning("Demo dataset is empty.")
    else:
        max_i = len(demo) - 1
        has_label = "actual_gesture" in demo.columns
        if "idx" not in st.session_state:
            st.session_state.idx = 0
        st.session_state.idx = min(st.session_state.idx, max_i)

        def step(d):
            st.session_state.idx = int(np.clip(st.session_state.idx + d, 0, max_i))

        def surprise():
            st.session_state.idx = random.randint(0, max_i)

        def next_miss():
            miss = np.flatnonzero(all_preds(model, mtime, demo) != demo["actual_gesture"].astype(int).values)
            if len(miss) == 0:
                return
            later = miss[miss > st.session_state.idx]
            st.session_state.idx = int(later[0] if len(later) else miss[0])

        b = st.columns(4)
        b[0].button("← Previous", on_click=step, args=(-1,), use_container_width=True)
        b[1].button("Next window →", on_click=step, args=(1,), use_container_width=True)
        b[2].button("🎲 Surprise me", on_click=surprise, use_container_width=True)
        if has_label:
            b[3].button("🔍 Find a mistake", on_click=next_miss, use_container_width=True)
        st.slider("Window index", 0, max_i, key="idx")

        row = demo.iloc[st.session_state.idx]
        pred, probs = predict_row(model, row)
        extra = ""
        if has_label:
            actual = int(row["actual_gesture"])
            an = GESTURES.get(actual, str(actual))
            extra = (f'<span class="badge ok">Matches the recorded label: {an}</span>' if actual == pred
                     else f'<span class="badge no">Recorded label was {an}</span>')
        st.caption(f"Window {st.session_state.idx + 1:,} of {len(demo):,}")

        left, right = st.columns([1, 1.1], gap="large")
        with left:
            pred_card(pred, probs, extra)
            if "subject" in row.index:
                st.caption(f"Test subject {row['subject']}, recording {row.get('recording', 'not specified')}")
            tab1, tab2 = st.tabs(["Gesture odds", "Channel profile"])
            with tab1:
                if probs is not None:
                    odds_chart(probs)
                else:
                    st.info("This model does not provide probabilities.")
            with tab2:
                profile_chart(row)
        with right:
            feat = st.radio("Feature shown on the armband", list(STATS), horizontal=True, key="feat")
            st.caption(f"{STAT_INFO[feat][0]}: {STAT_INFO[feat][1]} Brighter means stronger, "
                       "compared with a typical peak in the demo data.")
            armband(row, feat, demo)

elif page == "Upload & predict":
    st.subheader("Upload EMG feature windows")
    st.write("Your CSV needs all 32 feature columns. Add an `actual_gesture` column to also get an accuracy score.")
    uploaded = st.file_uploader("Choose a feature CSV", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
            missing = valid(data)
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            elif data.empty:
                st.warning("The CSV has no rows.")
            else:
                st.success(f"Loaded {len(data):,} feature windows.")
                if st.button("Run gesture recognition", type="primary"):
                    with st.spinner("Predicting gestures..."):
                        data["predicted_gesture"] = model.predict(data[FEATURES])
                        data["predicted_gesture_name"] = data["predicted_gesture"].map(GESTURES)
                    st.session_state.results = data
                if "results" in st.session_state:
                    result = st.session_state.results
                    st.dataframe(result[["predicted_gesture", "predicted_gesture_name"]].head(100),
                                 use_container_width=True)
                    st.download_button("Download prediction results", result.to_csv(index=False).encode(),
                                       "neuromotion_predictions.csv", "text/csv")
                    counts = (result["predicted_gesture_name"].value_counts()
                              .rename_axis("Gesture").reset_index(name="Windows"))
                    fig = px.bar(counts, x="Gesture", y="Windows", color="Gesture", color_discrete_map=COLORS)
                    clear_plot(fig, 320)
                    fig.update_layout(showlegend=False, xaxis=dict(gridcolor="rgba(0,0,0,0)"),
                                      yaxis=dict(gridcolor="#cdd6e8"))
                    fig.update_traces(marker_cornerradius=8)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                    if "actual_gesture" in result.columns:
                        score = accuracy_score(result["actual_gesture"].astype(int),
                                               result["predicted_gesture"].astype(int))
                        st.metric("Accuracy on uploaded labelled data", f"{score:.2%}")
        except Exception as e:
            st.error(f"Could not process CSV: {e}")

elif page == "Model insights":
    st.subheader("Model performance and methodology")
    p, q, r = st.columns(3)
    with p: metric("🎯", "Held-out test accuracy", "84.63%", "Unseen subjects")
    with q: metric("⚖️", "Held-out macro F1", "0.8464", "Balanced across gestures")
    with r: metric("🔁", "Grouped CV macro F1", "0.8217", "Used for tuning")
    st.write("")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            '<div class="neo"><b>How it was evaluated</b><ul>'
            '<li>Eight-channel EMG recordings from the UCI EMG Data for Gestures dataset.</li>'
            '<li>200 ms windows with a 50 ms stride.</li>'
            '<li>Four time-domain features per channel: MAV, RMS, WL and ZC.</li>'
            '<li>Subject-level 75:25 GroupShuffleSplit, so test participants never appear in training.</li>'
            '<li>GroupKFold cross-validation for model tuning.</li>'
            '<li>Extra Trees was the best-performing model in the reported experiment.</li></ul></div>',
            unsafe_allow_html=True)
    with b:
        st.markdown(
            '<div class="neo"><b>What to keep in mind</b><ul>'
            '<li>Displayed probabilities are classifier estimates, not necessarily calibrated certainty.</li>'
            '<li>This is an offline academic prototype.</li>'
            '<li>It has not been validated for clinical use, prosthetic control or real-time hardware.</li></ul></div>',
            unsafe_allow_html=True)

st.divider()
st.caption("NeuroMotion is an academic machine-learning demonstration that runs on recorded features only.")
