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
    "Hand at rest": "#8295a8",
    "Hand clenched in a fist": "#9387c8",
    "Wrist flexion": "#55a9a1",
    "Wrist extension": "#729dc9",
    "Radial deviation": "#d7a65e",
    "Ulnar deviation": "#d88980"
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --nm-bg: #e9edf2;
  --nm-surface: #e9edf2;
  --nm-ink: #263449;
  --nm-muted: #718096;
  --nm-teal: #368f88;
  --nm-teal-dark: #28766f;
  --nm-lavender: #8b83bd;
  --nm-coral: #d98e83;
  --nm-shadow-dark: #c7cdd5;
  --nm-shadow-light: #ffffff;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', 'Manrope', sans-serif;
  color: var(--nm-ink);
}
.stApp {
  background: var(--nm-bg);
  color: var(--nm-ink);
}
.block-container {
  max-width: 1460px;
  padding-top: 2.1rem;
  padding-bottom: 3rem;
}
[data-testid="stHeader"] {
  background: rgba(233,237,242,.86);
}
[data-testid="stSidebar"] {
  background: var(--nm-bg);
  border-right: 0;
  box-shadow: 7px 0 18px rgba(190,198,208,.28);
}
[data-testid="stSidebar"] > div:first-child {
  padding-top: 1.5rem;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
  color: var(--nm-ink);
  letter-spacing: .04em;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--nm-muted);
}
[data-testid="stRadio"] > div {
  gap: .45rem;
}
[data-testid="stRadio"] label {
  background: var(--nm-surface);
  border-radius: 14px;
  padding: .55rem .75rem;
  box-shadow: 4px 4px 9px var(--nm-shadow-dark), -4px -4px 9px var(--nm-shadow-light);
  border: 1px solid rgba(255,255,255,.45);
}
[data-testid="stRadio"] label:has(input:checked) {
  box-shadow: inset 3px 3px 7px var(--nm-shadow-dark), inset -3px -3px 7px var(--nm-shadow-light);
  color: var(--nm-teal-dark);
  font-weight: 700;
}
.hero {
  padding: 1.8rem 2rem;
  border-radius: 28px;
  background: linear-gradient(135deg, #edf1f5, #e4e9ef);
  box-shadow: 10px 10px 24px var(--nm-shadow-dark), -10px -10px 24px var(--nm-shadow-light);
  border: 1px solid rgba(255,255,255,.7);
  margin-bottom: 1.5rem;
}
.eyebrow {
  color: var(--nm-teal-dark);
  letter-spacing: .19em;
  font-size: .72rem;
  font-weight: 800;
}
.hero h1 {
  font-size: clamp(2rem, 4vw, 2.8rem);
  margin: .35rem 0;
  color: #253449;
  font-weight: 800;
  letter-spacing: -.045em;
}
.hero p, .muted {
  color: var(--nm-muted);
}
.metric {
  background: var(--nm-surface);
  border: 1px solid rgba(255,255,255,.72);
  border-radius: 22px;
  padding: 1.15rem 1.2rem;
  min-height: 116px;
  box-shadow: 7px 7px 16px var(--nm-shadow-dark), -7px -7px 16px var(--nm-shadow-light);
  transition: transform .18s ease, box-shadow .18s ease;
}
.metric:hover {
  transform: translateY(-2px);
  box-shadow: 9px 9px 19px var(--nm-shadow-dark), -9px -9px 19px var(--nm-shadow-light);
}
.metric small {
  color: var(--nm-muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  font-size: .68rem;
  font-weight: 800;
}
.metric h2 {
  color: var(--nm-teal-dark);
  margin: .35rem 0 .2rem;
  font-size: 1.75rem;
  font-weight: 800;
}
.pred {
  border: 1px solid rgba(255,255,255,.75);
  background: linear-gradient(135deg, #e3f2ef, #e9edf2);
  border-radius: 24px;
  padding: 1.35rem 1.5rem;
  box-shadow: inset 4px 4px 9px #cbd8d5, inset -4px -4px 9px #ffffff;
}
.pred small {
  color: var(--nm-teal-dark);
  letter-spacing: .13em;
  text-transform: uppercase;
  font-size: .7rem;
  font-weight: 800;
}
.pred h2 {
  color: #286d67;
  margin: .35rem 0 0;
  font-size: 1.55rem;
}
.notice {
  border: 1px solid rgba(255,255,255,.8);
  background: #f4eee5;
  border-radius: 18px;
  padding: 1rem 1.1rem;
  color: #8a6844;
  box-shadow: inset 3px 3px 7px #ded8d0, inset -3px -3px 7px #ffffff;
}
.stButton > button, .stDownloadButton > button {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,.8);
  background: var(--nm-surface);
  color: var(--nm-teal-dark);
  font-weight: 800;
  padding: .55rem 1.1rem;
  box-shadow: 5px 5px 10px var(--nm-shadow-dark), -5px -5px 10px var(--nm-shadow-light);
  transition: all .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  color: #205e59;
  border-color: rgba(54,143,136,.35);
  box-shadow: inset 3px 3px 7px var(--nm-shadow-dark), inset -3px -3px 7px var(--nm-shadow-light);
}
.stButton > button:focus, .stDownloadButton > button:focus {
  box-shadow: inset 3px 3px 7px var(--nm-shadow-dark), inset -3px -3px 7px var(--nm-shadow-light);
  outline: 2px solid rgba(54,143,136,.25);
}
.stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"],
.stFileUploader section {
  border-radius: 14px !important;
  background: var(--nm-surface) !important;
  box-shadow: inset 3px 3px 7px var(--nm-shadow-dark), inset -3px -3px 7px var(--nm-shadow-light) !important;
}
.stFileUploader section {
  border: 1px dashed #aeb9c6 !important;
  padding: 1.1rem !important;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 5px 5px 12px var(--nm-shadow-dark), -5px -5px 12px var(--nm-shadow-light);
}
[data-testid="stMetric"] {
  background: var(--nm-surface);
  border-radius: 18px;
  padding: .9rem 1rem;
  box-shadow: 5px 5px 10px var(--nm-shadow-dark), -5px -5px 10px var(--nm-shadow-light);
}
[data-testid="stMetric"] label {
  color: var(--nm-muted) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
  color: var(--nm-teal-dark);
}
h1, h2, h3, h4 {
  color: #2b3b50;
  letter-spacing: -.025em;
}
p, li, .stMarkdown {
  color: #4e5d70;
}
hr {
  border-color: #d3d9e1;
}
[data-testid="stAlert"] {
  border-radius: 15px;
}
footer { visibility: hidden; }
@media (max-width: 800px) {
  .block-container { padding: 1.1rem 1rem 2rem; }
  .hero { padding: 1.3rem; border-radius: 22px; }
  .hero h1 { font-size: 2rem; }
}
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
                          plot_bgcolor="rgba(0,0,0,0)",font_color="#526276",
                          xaxis=dict(range=[0,1],tickformat=".0%",gridcolor="#d8dee6"),
                          yaxis=dict(title="",gridcolor="rgba(0,0,0,0)"))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig,use_container_width=True)
    return pred, name

st.markdown("""
<div class="hero"><div class="eyebrow">BIOSIGNAL INTELLIGENCE · MACHINE LEARNING</div>
<h1>🧠 NEUROMOTION</h1><p>Decoding human hand intent through eight-channel EMG feature patterns.</p></div>
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
    st.caption("EMG gesture intelligence")
    page = st.radio("NAVIGATION",["Overview","Gesture replay","Upload & predict","Model insights"])
    st.markdown("---")
    st.success("Extra Trees pipeline loaded")
    st.caption("32 inputs · 6 gesture classes")

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
                          font_color="#526276",xaxis=dict(gridcolor="#d8dee6"),yaxis=dict(gridcolor="#d8dee6"))
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
                                      plot_bgcolor="rgba(0,0,0,0)",font_color="#526276")
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
