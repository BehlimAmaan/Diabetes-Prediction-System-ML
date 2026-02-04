import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.2);
    }
    
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.3);
    }
    
    .sidebar .sidebar-content {
        background: #f8fafc;
    }
    
    .section-title {
        color: #1565c0;
        border-bottom: 2px solid #e3f2fd;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    
    .caution-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .recommendation-box {
        background: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model
try:
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scalar.pkl")
except:
    st.error("⚠️ Model files not found. Please ensure 'model.pkl' and 'scalar.pkl' are in the 'models' folder.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:2.5rem;">🩸 Diabetes Risk Assessment System</h1>
    <p style="margin:0.5rem 0; font-size:1.2rem; opacity:0.9;">
        AI-Powered Early Detection for Better Health Outcomes
    </p>
    <div style="display: flex; gap: 1rem; margin-top: 1rem; font-size:0.9rem;">
        <span>🔬 Clinically-Informed Model</span>
        <span>📊 Real-time Analysis</span>
        <span>🛡️ Privacy-First</span>
        <span>💡 Evidence-Based</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
        <h2 style="color:#1565c0;">🩺 Health Assessment</h2>
        <p style="color:#666; font-size:0.9rem;">Complete all fields for accurate risk prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Personal Information
    st.markdown("### 👤 Personal Information")
    age = st.slider("Age (years)", 1, 120, 30, 
                   help="Age is a significant factor in diabetes risk")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    
    # Health Metrics
    st.markdown("### 📊 Health Metrics")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, 
                             help="Body Mass Index: Underweight (<18.5), Normal (18.5-24.9), Overweight (25-29.9), Obese (≥30)")
    with col2:
        hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.5,
                               help="Glycated hemoglobin: Normal (<5.7%), Prediabetes (5.7-6.4%), Diabetes (≥6.5%)")
    
    blood_glucose_level = st.slider("Blood Glucose (mg/dL)", 50, 300, 120,
                                   help="Fasting blood glucose: Normal (<100), Prediabetes (100-125), Diabetes (≥126)")
    
    # Medical History
    st.markdown("### 🏥 Medical History")
    hypertension = st.radio("Hypertension", ["No", "Yes"], 
                          help="High blood pressure increases diabetes risk")
    heart_disease = st.radio("Heart Disease", ["No", "Yes"])
    
    smoking_ui = st.selectbox(
        "Smoking History",
        ["Never smoked", "Quit smoking", "Currently smoking", 
         "Smoked earlier (not now)", "Smoked at least once"]
    )
    
    # Convert inputs
    hypertension_val = 1 if hypertension == "Yes" else 0
    heart_disease_val = 1 if heart_disease == "Yes" else 0
    
    gender_female = 1 if gender == "Female" else 0
    gender_male = 1 if gender == "Male" else 0
    gender_other = 1 if gender == "Other" else 0
    
    smoking_map = {
        "Never smoked": "never",
        "Quit smoking": "former",
        "Currently smoking": "current",
        "Smoked earlier (not now)": "not current",
        "Smoked at least once": "ever"
    }
    smoking = smoking_map[smoking_ui]
    
    smoking_current = 1 if smoking == "current" else 0
    smoking_ever = 1 if smoking == "ever" else 0
    smoking_former = 1 if smoking == "former" else 0
    smoking_never = 1 if smoking == "never" else 0
    smoking_not_current = 1 if smoking == "not current" else 0

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Cautions and Information
    st.markdown('<h3 class="section-title">⚠️ Important Information</h3>', unsafe_allow_html=True)
    
    with st.expander("🔍 Understanding Your Assessment", expanded=True):
        st.markdown("""
        **This tool provides a risk assessment based on machine learning analysis:**
        
        - **Not a diagnosis**: This is a screening tool, not a medical diagnosis
        - **For awareness**: Helps identify potential risk factors
        - **Consult professionals**: Always consult healthcare providers for proper diagnosis
        - **Regular check-ups**: Annual screenings recommended for adults over 45
        """)
    
    st.markdown('<div class="caution-box">', unsafe_allow_html=True)
    st.markdown("""
    **⚠️ Medical Disclaimer**
    
    This prediction system is for informational purposes only. It does not provide medical advice, 
    diagnosis, or treatment. Always seek the advice of your physician or other qualified health 
    provider with any questions you may have regarding a medical condition.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Quick Stats
    st.markdown("### 📈 Health Indicators")
    
    # BMI Indicator
    bmi_status = "Normal"
    bmi_color = "#4CAF50"
    if bmi < 18.5:
        bmi_status = "Underweight"
        bmi_color = "#2196F3"
    elif bmi >= 25 and bmi < 30:
        bmi_status = "Overweight"
        bmi_color = "#FF9800"
    elif bmi >= 30:
        bmi_status = "Obese"
        bmi_color = "#F44336"
    
    # HbA1c Indicator
    hba1c_status = "Normal"
    hba1c_color = "#4CAF50"
    if hba1c >= 5.7 and hba1c < 6.5:
        hba1c_status = "Prediabetes"
        hba1c_color = "#FF9800"
    elif hba1c >= 6.5:
        hba1c_status = "Diabetes"
        hba1c_color = "#F44336"
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.9rem; color:#666;">Your BMI</div>
        <div style="font-size:1.5rem; font-weight:bold; color:{bmi_color};">{bmi:.1f}</div>
        <div style="font-size:0.8rem; color:{bmi_color};">{bmi_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size:0.9rem; color:#666;">HbA1c Level</div>
        <div style="font-size:1.5rem; font-weight:bold; color:{hba1c_color};">{hba1c:.1f}%</div>
        <div style="font-size:0.8rem; color:{hba1c_color};">{hba1c_status}</div>
    </div>
    """, unsafe_allow_html=True)

# Prediction Button
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔬 Analyze Diabetes Risk", 
                              use_container_width=True,
                              type="primary")

# Prediction Logic
if predict_button:
    with st.spinner("🔄 Analyzing health data..."):
        # Prepare input data
        input_data = np.array([[
            age,
            hypertension_val,
            heart_disease_val,
            bmi,
            hba1c,
            blood_glucose_level,
            gender_female,
            gender_male,
            gender_other,
            smoking_current,
            smoking_ever,
            smoking_former,
            smoking_never,
            smoking_not_current
        ]])
        
        # Scale and predict
        input_scaled = scaler.transform(input_data)
        probability = model.predict_proba(input_scaled)[0][1]
        threshold = 0.40
        
        # Display Results
        st.markdown("---")
        st.markdown('<h2 class="section-title">📊 Assessment Results</h2>', unsafe_allow_html=True)
        
        # Risk visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk Score", 'font': {'size': 20}},
            delta = {'reference': 40, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#4CAF50'},
                    {'range': [30, 40], 'color': '#FFC107'},
                    {'range': [40, 100], 'color': '#F44336'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 40}}))
        
        fig.update_layout(height=300, margin=dict(t=50, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk Classification
        if probability >= threshold:
            st.markdown(f"""
            <div class="risk-card" style="border-color:#F44336;">
                <h3 style="color:#F44336; margin:0;">⚠️ HIGH RISK</h3>
                <p style="margin:0.5rem 0; font-size:1.1rem;">
                    Probability: <b>{probability*100:.1f}%</b>
                </p>
                <p style="margin:0; color:#666;">
                    Immediate medical consultation recommended
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="caution-box" style="color: #F44336;">
                <h4>🚨 Immediate Actions Recommended:</h4>
                <ul>
                    <li>Consult a healthcare provider if there are concerns about your blood sugar levels.</li>
                    <li>Begin monitoring blood sugar regularly.</li>
                    <li>Make changes to your diet and exercise plan.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        elif 0.30 < probability < threshold:
            st.markdown(f"""
            <div class="risk-card" style="border-color:#FF9800;">
                <h3 style="color:#FF9800; margin:0;">🟡 MODERATE RISK</h3>
                <p style="margin:0.5rem 0; font-size:1.1rem;">
                    Probability: <b>{probability*100:.1f}%</b>
                </p>
                <p style="margin:0; color:#666;">
                    Lifestyle modifications and regular monitoring advised
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="recommendation-box"style="color: #FFC107">
                <h4>📝 Preventive Recommendations:</h4>
                <ul>
                    <li>Lose 5-7% of body weight if overweight</li>
                    <li>Increase physical activity to 150 minutes/week</li>
                    <li>Follow a balanced diet with reduced sugar intake</li>
                    <li>Schedule annual diabetes screening</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
            <div class="risk-card" style="border-color:#4CAF50;">
                <h3 style="color:#4CAF50; margin:0;">✅ LOW RISK</h3>
                <p style="margin:0.5rem 0; font-size:1.1rem;">
                    Probability: <b>{probability*100:.1f}%</b>
                </p>
                <p style="margin:0; color:#666;">
                    Maintain healthy lifestyle with regular check-ups
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="recommendation-box" style="color: #4caf50;">
                <h4>💚 Maintain Healthy Habits:</h4>
                <ul>
                    <li>Continue regular physical activity</li>
                    <li>Maintain balanced nutrition</li>
                    <li>Annual health check-ups recommended</li>
                    <li>Monitor family history changes</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Risk Factors
        st.markdown("### 🔍 Identified Risk Factors")
        risk_factors = []
        
        if age >= 45:
            risk_factors.append("Age (45+)")
        if bmi >= 25:
            risk_factors.append(f"BMI ({bmi:.1f}) - Overweight/Obese")
        if hba1c >= 5.7:
            risk_factors.append(f"HbA1c ({hba1c:.1f}%) - Elevated")
        if blood_glucose_level >= 100:
            risk_factors.append(f"Blood Glucose ({blood_glucose_level} mg/dL) - Elevated")
        if hypertension_val == 1:
            risk_factors.append("Hypertension")
        if heart_disease_val == 1:
            risk_factors.append("Heart Disease")
        if smoking in ["current", "former"]:
            risk_factors.append("Smoking History")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("No significant risk factors identified in your profile.")
        
        # Download Report
        st.markdown("---")
        report_text = f"""
        Diabetes Risk Assessment Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        
        Personal Information:
        - Age: {age} years
        - Gender: {gender}
        - BMI: {bmi:.1f} ({bmi_status})
        - HbA1c: {hba1c:.1f}% ({hba1c_status})
        - Blood Glucose: {blood_glucose_level} mg/dL
        
        Medical History:
        - Hypertension: {hypertension}
        - Heart Disease: {heart_disease}
        - Smoking: {smoking_ui}
        
        Assessment Results:
        - Diabetes Risk Probability: {probability*100:.1f}%
        - Risk Level: {'High' if probability >= threshold else 'Moderate' if probability > 0.3 else 'Low'}
        
        Recommendations:
        {('Immediate medical consultation recommended' if probability >= threshold else 
          'Lifestyle modifications and monitoring advised' if probability > 0.3 else 
          'Maintain healthy lifestyle with regular check-ups')}
        
        Disclaimer: This is an AI-generated risk assessment, not a medical diagnosis.
        """
        
        st.download_button(
            label="📥 Download Assessment Report",
            data=report_text,
            file_name=f"diabetes_assessment_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666; font-size:0.8rem; padding:1rem;">
    <p>© 2024 Diabetes Risk Assessment System | For educational purposes only</p>
    <p>Data privacy is guaranteed - no personal data is stored or shared</p>
    <p>Developed with ❤️ for better community health awareness</p>
</div>
""", unsafe_allow_html=True)