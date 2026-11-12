import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="NeuroMotion", page_icon="🧠", layout="wide")
ROOT = Path(__file__).parent
MODEL = ROOT / "best_emg_gesture_model.joblib"
DEMO = ROOT / "demo_features.csv"

FEATURES = [f"ch{i}_{f}" for i in range(1, 9) for f in ("MAV", "RMS", "WL", "ZC")]
GESTURES = {
    1: "Hand at rest", 2: "Hand clenched in a fist", 3: "Wrist flexion",
    4: "Wrist extension", 5: "Radial deviation", 6: "Ulnar deviation"
}
COLORS = {
    "Hand at rest": "#94a3b8", "Hand clenched in a fist": "#a78bfa",
    "Wrist flexion": "#22d3ee", "Wrist extension": "#38bdf8",
    "Radial deviation": "#fbbf24", "Ulnar deviation": "#fb7185"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{
 --bg:#f0eef8;--panel:#ffffff;--panel2:#f7f6fc;--ink:#171b3b;--muted:#777b99;
 --purple:#7658ed;--violet:#b58af6;--cyan:#32c9df;--pink:#ed83c8;
 --navy:#071538;--line:#e9e7f2;--shadow:0 14px 38px rgba(49,43,103,.09);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--ink)}
.stApp{background:radial-gradient(ellipse at 8% 8%,rgba(204,190,255,.35),transparent 32%),radial-gradient(ellipse at 92% 82%,rgba(248,196,231,.28),transparent 30%),var(--bg)}
.block-container{max-width:1500px;padding:1.5rem 2rem 3rem}
[data-testid="stHeader"]{background:rgba(240,238,248,.88)}
[data-testid="stSidebar"]{background:#fbfaff;border-right:1px solid #e9e6f3;box-shadow:8px 0 30px rgba(46,42,91,.045)}
[data-testid="stSidebar"]>div:first-child{padding:1.6rem 1.05rem}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2{font-family:'Space Grotesk';font-weight:700;letter-spacing:-.04em;color:#171b3b}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"]{color:#8588a4}
[data-testid="stRadio"]>div{gap:.48rem}
[data-testid="stRadio"] label{border-radius:13px;padding:.58rem .72rem;background:transparent;border:1px solid transparent;transition:.2s}
[data-testid="stRadio"] label:has(input:checked){background:linear-gradient(110deg,#eeeaff,#f8f4ff);border-color:#e2dafa;box-shadow:0 5px 14px rgba(118,88,237,.10);color:#6848df}
[data-testid="stRadio"] label p{font-weight:650}
[data-testid="stMetric"]{background:#fff;border:1px solid #eceaf5;border-radius:18px;padding:16px 18px;box-shadow:var(--shadow)}
[data-testid="stMetricLabel"]{color:#8587a0;font-size:.76rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em}
[data-testid="stMetricValue"]{color:#171b3b;font-family:'Space Grotesk';font-weight:700}
[data-testid="stMetricDelta"]{font-size:.78rem}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;color:#171b3b}
h1{letter-spacing:-.045em}
h2{letter-spacing:-.025em}
p,li{color:#565b78}
[data-testid="stMarkdownContainer"] strong{color:#27294d}
.hero{position:relative;overflow:hidden;padding:1.55rem 1.7rem;border-radius:25px;background:linear-gradient(112deg,#071538 0%,#101e50 55%,#34216d 100%);border:1px solid rgba(172,158,255,.22);box-shadow:0 20px 48px rgba(24,24,73,.18);margin-bottom:1.35rem;min-height:175px}
.hero:after{content:"";position:absolute;width:360px;height:360px;right:3%;top:-145px;border-radius:50%;background:radial-gradient(circle,rgba(76,219,255,.25),rgba(142,93,255,.14) 42%,transparent 70%);pointer-events:none}
.hero .eyebrow{color:#70e6f5;letter-spacing:.2em;font-size:.68rem;font-weight:800}
.hero h1{font-size:2.5rem;margin:.35rem 0 .2rem;color:#fff!important}
.hero p{color:#c4cde8!important;max-width:680px;margin:0}
.hero .hero-tag{display:inline-block;margin-top:1rem;padding:.38rem .7rem;border-radius:30px;background:rgba(112,230,245,.12);border:1px solid rgba(112,230,245,.28);color:#a9f2fa;font-size:.72rem;font-weight:700;letter-spacing:.06em}
.hero .orb{position:absolute;right:9%;top:23px;width:132px;height:132px;border-radius:50%;background:radial-gradient(circle at 30% 25%,#fff 0%,#b4e9ff 12%,#8b80ff 42%,#563bd0 66%,#171b5c 100%);box-shadow:inset -15px -18px 28px rgba(8,13,54,.45),0 0 50px rgba(94,125,255,.28)}
.hero .orb:before,.hero .orb:after{content:"";position:absolute;border:1px solid rgba(160,214,255,.42);border-radius:50%;inset:-14px;transform:rotate(-25deg) scaleY(.35)}
.hero .orb:after{inset:-26px;transform:rotate(38deg) scaleY(.28)}
.metric{background:#fff;border:1px solid #eceaf5;border-radius:18px;padding:1rem 1.1rem;min-height:112px;box-shadow:var(--shadow)}
.metric small{color:#8587a0;text-transform:uppercase;letter-spacing:.09em;font-size:.7rem;font-weight:800}
.metric h2{color:#191c40;margin:.35rem 0;font-size:1.65rem;font-family:'Space Grotesk'}
.metric .muted{color:#9698af;font-size:.78rem}
.pred{border:1px solid #d9d1ff;background:linear-gradient(120deg,#f2efff,#fff7fd);border-radius:20px;padding:1.2rem 1.35rem;box-shadow:0 12px 28px rgba(113,86,220,.09)}
.pred small{color:#8063df;letter-spacing:.13em;text-transform:uppercase;font-weight:800;font-size:.68rem}
.pred h2{color:#292052;margin:.35rem 0;font-size:1.5rem}
.notice{border:1px solid #d9d3f5;background:linear-gradient(110deg,#f2efff,#fff);border-radius:15px;padding:1rem 1.1rem;color:#625b88}
.stButton>button,.stDownloadButton>button{border-radius:12px;border:1px solid #d9d1ff;background:linear-gradient(110deg,#7658ed,#9b78f5);color:#fff;font-weight:750;padding:.55rem 1rem;box-shadow:0 7px 18px rgba(118,88,237,.2);transition:transform .18s,box-shadow .18s}
.stButton>button:hover,.stDownloadButton>button:hover{border-color:#7658ed;color:#fff;transform:translateY(-2px);box-shadow:0 11px 22px rgba(118,88,237,.26)}
.stButton>button:focus,.stDownloadButton>button:focus{color:#fff}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"],.stFileUploader section{border-radius:13px!important}
[data-testid="stFileUploader"]{background:#faf9ff;border:1.5px dashed #c9bdf6;border-radius:18px;padding:1rem}
[data-testid="stDataFrame"]{border:1px solid #eceaf5;border-radius:15px;overflow:hidden}
hr{border-color:#e5e2f0!important}
[data-testid="stAlert"]{border-radius:14px}
[data-testid="stSidebar"] [data-testid="stAlert"]{border-radius:13px}
footer{visibility:hidden}
@media(max-width:900px){.block-container{padding:1rem 1rem 2rem}.hero{padding:1.25rem;min-height:150px}.hero h1{font-size:2rem}.hero .orb{width:85px;height:85px;right:7%;top:32px;opacity:.8}}
@media(max-width:600px){.hero .orb{display:none}.hero h1{font-size:1.8rem}.block-container{padding-left:.75rem;padding-right:.75rem}}
</style>
""", unsafe_allow_html=True)

def metric(label, value, detail):
    st.markdown(f'<div class="metric"><small>{label}</small><h2>{value}</h2><span class="muted">{detail}</span></div>', unsafe_allow_html=True)

@st.cache_resource
def get_model(path, mtime):
    return joblib.load(path)

def valid(df):
    return [c for c in FEATURES if c not in df.columns]

def prediction_view(model, row):
    x = pd.DataFrame([row[FEATURES].to_dict()], columns=FEATURES)
    pred = model.predict(x)[0]
    name = GESTURES.get(int(pred), str(pred))
    st.markdown(f'<div class="pred"><small>Predicted gesture</small><h2>✦ {name}</h2></div>', unsafe_allow_html=True)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)[0]
        classes = model.classes_
        d = pd.DataFrame({"Gesture":[GESTURES.get(int(c),str(c)) for c in classes],"Probability":probs})
        d = d.sort_values("Probability")
        fig = px.bar(d, x="Probability", y="Gesture", orientation="h", color="Gesture",
                     color_discrete_map=COLORS, text=d["Probability"].map(lambda v:f"{v:.1%}"))
        fig.update_layout(height=300,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",font_color="#343858",
                          xaxis=dict(range=[0,1],tickformat=".0%",gridcolor="#eceaf5"),
                          yaxis=dict(title="",gridcolor="rgba(0,0,0,0)"))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig,use_container_width=True)
    return pred, name

st.markdown("""
<div class="hero">
  <div class="eyebrow">NEUROTECHNOLOGY · BIOSIGNAL INTELLIGENCE</div>
  <h1>NEUROMOTION<span style="color:#70e6f5">.</span></h1>
  <p>Decoding human hand intent through intelligent analysis of eight-channel EMG patterns.</p>
  <span class="hero-tag">✦ EMG GESTURE RECOGNITION &nbsp; / &nbsp; EXTRA TREES AI</span>
  <div class="orb"></div>
</div>
""",unsafe_allow_html=True)

if not MODEL.exists():
    st.error("Model missing. Add best_emg_gesture_model.joblib beside app.py in your GitHub repository.")
    st.stop()
try:
    model = get_model(str(MODEL), MODEL.stat().st_mtime)
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.info("Check that scikit-learn and joblib versions match the training environment.")
    st.stop()

demo = pd.read_csv(DEMO) if DEMO.exists() else None

with st.sidebar:
    st.markdown("## ◈ NEUROMOTION")
    st.caption("EMG GESTURE INTELLIGENCE")
    page = st.radio("NAVIGATION",["Overview","Gesture replay","Upload & predict","Model insights"])
    st.markdown("---")
    st.success("AI MODEL ONLINE")
    st.caption("32 INPUTS  ·  6 GESTURE CLASSES")

a,b,c,d = st.columns(4)
with a: metric("Model","Extra Trees","Trained pipeline")
with b: metric("Input features","32","4 features × 8 channels")
with c: metric("Gesture classes","06","Recognized movements")
with d: metric("Test accuracy","84.63%","Held-out subject test result")

if page == "Overview":
    l,r = st.columns([1.3,1])
    with l:
        st.subheader("From muscle activity to gesture prediction")
        st.write("NeuroMotion classifies feature windows extracted from eight EMG channels. Each window contains four time-domain features per channel:")
        st.markdown("- **MAV** — Mean Absolute Value\n- **RMS** — Root Mean Square\n- **WL** — Waveform Length\n- **ZC** — Zero Crossings")
        st.markdown('<div class="notice">DEMO MODE · This hosted prototype processes recorded feature windows. It does not connect to a physical MYO armband or acquire live EMG signals.</div>',unsafe_allow_html=True)
    with r:
        st.subheader("Six recognized gestures")
        for k,v in GESTURES.items(): st.write(f"**{k}.** {v}")
    st.divider()
    if demo is not None:
        missing=valid(demo)
        if missing: st.error("Demo CSV is missing columns: "+", ".join(missing))
        else: st.success(f"Bundled held-out test dataset loaded: {len(demo):,} windows.")
    else: st.info("Add demo_features.csv to the repository root to enable replay.")

elif page == "Gesture replay":
    st.subheader("Recorded gesture replay")
    st.caption("Step through held-out feature windows and inspect each prediction.")
    if demo is None: st.warning("demo_features.csv is missing. Add it to the repository or use Upload & predict.")
    elif valid(demo): st.error("Missing feature columns: "+", ".join(valid(demo)))
    elif demo.empty: st.warning("Demo dataset is empty.")
    else:
        max_i=len(demo)-1
        if "idx" not in st.session_state: st.session_state.idx=0
        st.session_state.idx=min(st.session_state.idx,max_i)
        x,y,z=st.columns([1,1,2])
        with x:
            if st.button("← Previous"): st.session_state.idx=max(0,st.session_state.idx-1)
        with y:
            if st.button("Next window →"): st.session_state.idx=min(max_i,st.session_state.idx+1)
        with z:
            st.session_state.idx=st.slider("Window index",0,max_i,st.session_state.idx)
        row=demo.iloc[st.session_state.idx]
        st.caption(f"Window {st.session_state.idx+1:,} of {len(demo):,}")
        pred,name=prediction_view(model,row)
        if "actual_gesture" in row.index:
            actual=int(row["actual_gesture"])
            actual_name=GESTURES.get(actual,str(actual))
            if actual==int(pred): st.success(f"Known test label: {actual_name} — prediction matches.")
            else: st.warning(f"Known test label: {actual_name} — prediction differs.")
        if "subject" in row.index:
            st.caption(f"Test subject: {row['subject']} · Recording: {row.get('recording','Not specified')}")
        st.subheader("Eight-channel feature profile")
        prof=[{"Channel":f"CH {i}","MAV":row[f"ch{i}_MAV"],"RMS":row[f"ch{i}_RMS"],
               "WL":row[f"ch{i}_WL"],"ZC":row[f"ch{i}_ZC"]} for i in range(1,9)]
        long=pd.DataFrame(prof).melt(id_vars="Channel",var_name="Feature",value_name="Value")
        fig=px.line(long,x="Channel",y="Value",color="Feature",markers=True)
        fig.update_layout(height=330,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#343858",xaxis=dict(gridcolor="#eceaf5"),yaxis=dict(gridcolor="#eceaf5"))
        st.plotly_chart(fig,use_container_width=True)
        st.caption("This shows extracted feature values, not raw EMG voltage waveforms.")

elif page == "Upload & predict":
    st.subheader("Upload EMG feature windows")
    st.write("CSV must contain all 32 feature columns. Optional `actual_gesture` labels enable evaluation.")
    uploaded=st.file_uploader("Choose a feature CSV",type=["csv"])
    if uploaded:
        try:
            data=pd.read_csv(uploaded)
            missing=valid(data)
            if missing: st.error("Missing required columns: "+", ".join(missing))
            elif data.empty: st.warning("CSV has no rows.")
            else:
                st.success(f"Loaded {len(data):,} feature windows.")
                if st.button("Run gesture recognition",type="primary"):
                    with st.spinner("Predicting gestures..."):
                        data["predicted_gesture"]=model.predict(data[FEATURES])
                        data["predicted_gesture_name"]=data["predicted_gesture"].map(GESTURES)
                    st.session_state.results=data
                if "results" in st.session_state:
                    result=st.session_state.results
                    st.dataframe(result[["predicted_gesture","predicted_gesture_name"]].head(100),use_container_width=True)
                    st.download_button("Download prediction results",result.to_csv(index=False).encode(),
                                       "neuromotion_predictions.csv","text/csv")
                    counts=result["predicted_gesture_name"].value_counts().rename_axis("Gesture").reset_index(name="Windows")
                    fig=px.bar(counts,x="Gesture",y="Windows",color="Gesture",color_discrete_map=COLORS)
                    fig.update_layout(height=320,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",
                                      plot_bgcolor="rgba(0,0,0,0)",font_color="#343858")
                    st.plotly_chart(fig,use_container_width=True)
                    if "actual_gesture" in result.columns:
                        score=accuracy_score(result["actual_gesture"].astype(int),result["predicted_gesture"].astype(int))
                        st.metric("Accuracy on uploaded labelled data",f"{score:.2%}")
        except Exception as e: st.error(f"Could not process CSV: {e}")

elif page == "Model insights":
    st.subheader("Model performance & methodology")
    p,q,r=st.columns(3)
    p.metric("Held-out test accuracy","84.63%")
    q.metric("Held-out Macro F1","0.8464")
    r.metric("Grouped CV Macro F1","0.8217")
    st.markdown("""
**Evaluation design**
- Eight-channel EMG recordings from the UCI EMG Data for Gestures dataset.
- 200 ms windows with a 50 ms stride.
- Four time-domain features per channel: MAV, RMS, WL and ZC.
- Subject-level 75:25 GroupShuffleSplit, keeping participants separate between training and held-out testing.
- GroupKFold cross-validation for model tuning.
- Extra Trees selected as the best-performing model in the reported experiment.

**Interpretation and limitations**

The displayed model probabilities are classifier probability estimates, not necessarily calibrated certainty. This is an offline academic prototype; it has not been validated for clinical use, prosthetic actuation or real-time hardware control.
""")

st.divider()
st.caption("NEUROMOTION · Academic machine-learning demonstration · Recorded feature inference only")
