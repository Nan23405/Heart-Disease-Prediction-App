import streamlit as st
import pandas as pd
import joblib

# Page Config
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


st.write("App Started")


# Custom CSS
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
}

h1 {
    color: #ff4b4b;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


#Title
st.title("Heart Disease Prediction App❤️")

st.warning(
    "⚠️ This prediction is for educational purposes only."
)


try: #*
    model = joblib.load('heart_model_LG.pkl')
    scaler = joblib.load('scaler.pkl')
    expected_columns = joblib.load('columns.pkl')

except Exception as e: #*
    st.error(f"Error loading model files: {e}")
    st.stop()

# st.title("Heart Disease Prediction App❤️")
# st.markdown("This app predicts the likelihood of a heart disease based on user input.")
col1, col2 = st.columns(2)  #*

chest_pain_options = [
    ('ATA', 'ATA (Atypical Angina)'),
    ('NAP', 'NAP (Non-Angina Pain)'),
    ('ASY', 'ASY (Asymptomatic)'),
    ('TA', 'TA (Typical Angina)'),
]

with col1: #*
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("SEX", ['M', 'F'])
    selected_chest_pain = st.selectbox(
        "Chest Pain Type",
        [label for _, label in chest_pain_options]
    )
    chest_pain = next(code for code, label in chest_pain_options if label == selected_chest_pain)
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholestrol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)

with col2: #*
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ['Normal', 'ST', 'LVH'])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ['Up', 'Flat', 'Down'])

if st.button("Predict"):
    raw_data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholestrol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_data])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[expected_columns]
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("High risk of heart stroke. Please consult a doctor.")
    else:
        st.success("Low risk of heart stroke. Keep up the healthy lifestyle!")
    
