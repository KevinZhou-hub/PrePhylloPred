import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.title("Prediction of Phyllodes Tumor Based on Preoperative Features")

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
