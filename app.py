import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="CSP Clinical Portal", page_icon="⚕️", layout="wide")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


@st.cache_resource
def train_model():
    np.random.seed(42)
    num_samples = 400
    data = {
        'Age': np.random.randint(18, 80, num_samples),
        'Glucose': np.random.randint(70, 200, num_samples),
        'Blood_Pressure': np.random.randint(60, 140, num_samples),
        'BMI': np.random.uniform(15.0, 38.0, num_samples)
    }
    df = pd.DataFrame(data)
    
    def calculate_risk(row):
        points = 0
        if row['Glucose'] > 130: points += 2
        if row['Blood_Pressure'] > 90: points += 1
        if row['BMI'] > 27.5: points += 1
        return 1 if points >= 2 else 0

    df['Target_Risk'] = df.apply(calculate_risk, axis=1)
    X = df[['Age', 'Glucose', 'Blood_Pressure', 'BMI']]
    y = df['Target_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()


if not st.session_state.logged_in:
    st.title("⚕️ Clinical Screening Portal Login")
    st.subheader("Community Service Project (CSP) - Vignan Institute")
    st.write("---")
    
    username = st.text_input("Username / Field Worker ID", placeholder="e.g., worker01")
    password = st.text_input("Password", type="password", placeholder="••••••••")
    
    if st.button("Login to Portal"):
        if username == "admin" and password == "csp2026":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Username or Password. Try admin / csp2026")


else:
    st.sidebar.markdown("### 👤 Session Active")
    st.sidebar.write("User: **Field Admin**")
    if st.sidebar.button("Log Out Subsystem"):
        st.session_state.logged_in = False
        st.rerun()
        
    st.title("⚕️ AI-Based Disease Prediction System")
    st.subheader("Target Alignment: SDG 3 (Good Health and Well-being)")
    st.divider()

    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.markdown("### 📊 Patient Vitals Collection Form")
        age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=45)
        glucose = st.number_input("Fasting Blood Glucose (mg/dL)", min_value=50, max_value=300, value=120)
        bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=50, max_value=200, value=120)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=50.0, value=24.5)
        
        run_analysis = st.button("Execute Predictive Analysis ⚙️")

    with right_col:
        st.markdown("### 📋 Diagnostic Assessment Engine")
        
        if run_analysis:
            input_data = np.array([[age, glucose, bp, bmi]])
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][prediction] * 100
            
            st.markdown("#### Input Parameter Summary")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Age", f"{age}")
            m_col2.metric("Glucose", f"{glucose}")
            m_col3.metric("BP", f"{bp}")
            m_col4.metric("BMI", f"{bmi}")
            
            st.markdown("---")
            st.markdown("#### Evaluation Response")
            if prediction == 1:
                st.error("⚠️ STATUS: HIGH RISK PATIENT PROFILE DETECTED")
                st.info(f"Classification Confidence: {probability:.2f}%")
                st.warning("Recommended Intervention: Issue urgent referral to localized medical specialist checkup paths.")
            else:
                st.success("✅ STATUS: LOW PATIENT RISK DETECTED")
                st.info(f"Classification Confidence: {probability:.2f}%")
                st.success("Recommended Intervention: Standard preventative screening schedules maintained.")
        else:
            st.info("Input patient parameters inside the entry panel and press execute to view the diagnostic metrics visualization panel.")
