import streamlit as st
import streamlit.components.v1 as components
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="PredPhylloPre", layout="centered")

st.markdown("""
<style>
    /* Sci-tech gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2e4b 40%, #0a3d62 70%, #1a5276 100%);
        background-attachment: fixed;
    }
    /* Subtle grid overlay for tech feel */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(46,123,207,0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(46,123,207,0.07) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }
    /* Main content card */
    .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
        padding: 2.5rem 3rem;
        box-shadow: 0 4px 32px rgba(0,0,0,0.35), 0 0 0 1px rgba(46,123,207,0.2);
        max-width: 720px;
        position: relative;
        z-index: 1;
    }
    /* Title */
    h1 {
        color: #1a3a6b !important;
        font-size: 1.55rem !important;
        font-weight: 700;
        border-bottom: 3px solid #2e7bcf;
        padding-bottom: 0.5rem;
        margin-bottom: 0.5rem;
    }
    /* Labels */
    label, .stSelectbox label, .stNumberInput label {
        color: #1a3a6b !important;
        font-weight: 500;
    }
    /* Body text */
    p, .stMarkdown p {
        color: #2c3e50 !important;
    }
    /* Selectbox & number input */
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: #f4f8ff !important;
        color: #1a3a6b !important;
        border: 1.5px solid #b0c4de !important;
        border-radius: 6px !important;
    }
    /* Number input container & inner input */
    [data-baseweb="input"] {
        background-color: #f4f8ff !important;
        border: 1.5px solid #b0c4de !important;
        border-radius: 6px !important;
    }
    [data-baseweb="input"] input {
        background-color: #f4f8ff !important;
        color: #1a3a6b !important;
    }
    /* Number input +/- buttons */
    .stNumberInput > div > div > button {
        background-color: #e8f0fb !important;
        color: #1a3a6b !important;
        border: 1px solid #b0c4de !important;
    }
    .stNumberInput > div > div > button:hover {
        background-color: #2e7bcf !important;
        color: white !important;
    }
    /* Dropdown option list */
    [data-baseweb="select"] [role="option"],
    [data-baseweb="menu"] li {
        color: #1a3a6b !important;
        background-color: #ffffff !important;
    }
    [data-baseweb="select"] [role="option"]:hover,
    [data-baseweb="menu"] li:hover {
        background-color: #e8f0fb !important;
    }
    /* Predict button */
    .stButton > button {
        background-color: #2e7bcf;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        margin-top: 0.5rem;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #1a5fa8;
        color: white;
    }
    hr { border-color: #d0dce8; }
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(function(){
  // Remove existing canvas if any (hot-reload safe)
  const old = window.parent.document.getElementById('medCanvas');
  if(old) old.remove();

  const canvas = window.parent.document.createElement('canvas');
  canvas.id = 'medCanvas';
  canvas.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:0;pointer-events:none;';
  window.parent.document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  function resize(){
    canvas.width = window.parent.innerWidth;
    canvas.height = window.parent.innerHeight;
  }
  resize();
  window.parent.addEventListener('resize', resize);

  const N = 70;
  const W = () => canvas.width, H = () => canvas.height;
  const particles = Array.from({length: N}, () => ({
    x: Math.random() * canvas.width,
    y: Math.random() * canvas.height,
    r: Math.random() * 2.5 + 1,
    vx: (Math.random() - 0.5) * 0.4,
    vy: (Math.random() - 0.5) * 0.4,
    type: Math.random() < 0.2 ? 'cross' : 'dot',
    alpha: Math.random() * 0.5 + 0.25,
  }));

  function drawCross(x, y, size, alpha){
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.moveTo(x-size, y); ctx.lineTo(x+size, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y-size); ctx.lineTo(x, y+size); ctx.stroke();
    ctx.restore();
  }

  function draw(){
    ctx.clearRect(0, 0, W(), H());
    for(let i=0;i<N;i++) for(let j=i+1;j<N;j++){
      const dx=particles[i].x-particles[j].x, dy=particles[i].y-particles[j].y;
      const d=Math.sqrt(dx*dx+dy*dy);
      if(d<140){
        ctx.save(); ctx.globalAlpha=(1-d/140)*0.3;
        ctx.strokeStyle='rgba(255,255,255,0.9)'; ctx.lineWidth=0.7;
        ctx.beginPath(); ctx.moveTo(particles[i].x,particles[i].y);
        ctx.lineTo(particles[j].x,particles[j].y); ctx.stroke(); ctx.restore();
      }
    }
    particles.forEach(p=>{
      if(p.type==='cross') drawCross(p.x, p.y, p.r*3, p.alpha);
      else {
        ctx.save(); ctx.globalAlpha=p.alpha;
        ctx.fillStyle='white';
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2); ctx.fill(); ctx.restore();
      }
      p.x+=p.vx; p.y+=p.vy;
      if(p.x<0) p.x=W(); if(p.x>W()) p.x=0;
      if(p.y<0) p.y=H(); if(p.y>H()) p.y=0;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();
</script>
""", height=0)

st.title("PredPhylloPre (Prediction of Phyllodes Tumor Based on Preoperative Features)")

st.markdown("""
    This model predicts whether a breast phyllodes tumor is **benign** or **non-benign** based on preoperative clinical features.

    This model is developed on preopertive clinical features of breast phyllodes tumor and does NOT work on other breast tumor types.

    Please enter the patient data in the input boxes below.

    This model is intended to ASSIST physicians in making clinical decisions ONLY.
""")

FA_history = st.selectbox("History of fibroadenoma:", options=['No', 'Yes'])
age = st.number_input("Age (years):", min_value=1, max_value=120, value=40)
maximum_size = st.number_input("Maximum tumor size (cm):", min_value=0.0, max_value=50.0, value=2.0, step=0.1)
NUM_OPTIONS = {"Single": 1, "Multiple": 2}
num_label = st.selectbox("Number of tumors:", options=list(NUM_OPTIONS.keys()))
num = NUM_OPTIONS[num_label]
BI_rads = st.selectbox("BI-RADS category:", options=[2, 3, 4, 5])
US_DIAGNOSIS_OPTIONS = {
    "Other benign component": 1,
    "Intraductal papilloma": 2,
    "Fibroadenoma": 3,
    "Fibroadenoma or phyllodes tumor": 4,
    "Phyllodes tumor (PTs other than those explicitly indicated as malignant)": 5,
    "Malignant components (breast cancer, sarcoma, malignant phyllodes tumor and other malignancies)": 6,
}
US_diagnosis_label = st.selectbox("Ultrasound diagnosis:", options=list(US_DIAGNOSIS_OPTIONS.keys()))
US_diagnosis = US_DIAGNOSIS_OPTIONS[US_diagnosis_label]

feature_columns = ['FA_history', 'age', 'maxmium_size', 'num', 'BI_rads', 'US_diagnosis']
input_df = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)

input_df['FA_history'] = 1 if FA_history == 'Yes' else 0
input_df['age'] = age
input_df['maxmium_size'] = maximum_size
input_df['num'] = num
input_df['BI_rads'] = BI_rads
input_df['US_diagnosis'] = US_diagnosis

model = joblib.load('models/stacking_model.pkl')
imputer = joblib.load('models/imputer.pkl')
scaler = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

if st.button("Predict"):
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=feature_columns)
    input_scaled = pd.DataFrame(scaler.transform(input_imputed), columns=feature_columns)
    input_scaled_df = input_scaled

    pred = model.predict(input_scaled_df)[0]
    pred_proba = model.predict_proba(input_scaled_df)[0]

    label = label_encoder.inverse_transform([pred])[0]
    benign_prob = pred_proba[0] * 100
    nonbenign_prob = pred_proba[1] * 100

    if label == 'B':
        st.success(f"Prediction: **Benign**")
        st.write(f"Probability of benign: {benign_prob:.1f}%")
        st.write(f"Probability of non-benign: {nonbenign_prob:.1f}%")
    else:
        st.warning(f"Prediction: **Non-benign**")
        st.write(f"Probability of non-benign: {nonbenign_prob:.1f}%")
        st.write(f"Probability of benign: {benign_prob:.1f}%")
