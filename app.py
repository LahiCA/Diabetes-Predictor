import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io
import plotly.io as pio
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with modern design
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');
    
    /* Global Styles */
    .main {
        padding: 1.5rem 2rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Title */
    .custom-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .custom-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e0e7ff 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-family: 'Poppins', sans-serif;
        color: #4c1d95;
        font-size: 1.5rem;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {
        font-family: 'Poppins', sans-serif;
        color: #5b21b6;
        font-size: 1.1rem;
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #a78bfa;
    }
    
    /* Custom Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.15);
    }
    
    /* Result Cards */
    .result-card-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 6px solid #10b981;
        margin: 1rem 0;
    }
    
    .result-card-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 6px solid #f59e0b;
        margin: 1rem 0;
    }
    
    .result-card-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 2rem;
        border-radius: 1rem;
        border-left: 6px solid #ef4444;
        margin: 1rem 0;
    }
    
    /* Button Styling */
    .stButton > button {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Download Button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 0.75rem;
        padding: 0.75rem 2rem;
        transition: all 0.3s;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info Box Styling */
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Poppins', sans-serif;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #6b7280;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 1rem;
        border: none;
        font-family: 'Inter', sans-serif;
    }
    
    /* Section Headers */
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #1f2937;
    }
    
    /* Disclaimer Box */
    .disclaimer {
        background: #fef3c7;
        padding: 1rem;
        border-radius: 0.75rem;
        border-left: 4px solid #f59e0b;
        font-size: 0.9rem;
        color: #78350f;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'diabetes_model.pkl' and 'scaler_svm.pkl' are in the working directory.")
        return None, None

# Custom Header
st.markdown('<h1 class="custom-title">🩺 Diabetes Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="custom-subtitle">Advanced ML-powered diabetes risk assessment based on comprehensive health parameters</p>', unsafe_allow_html=True)

# Load Model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("⚠️ Unable to load model or scaler. Please check the files and try again.")
    st.info("""
            **Setup Required:**
            
            Please run the following command first:
            ```
            python diabetes_prediction.py
            ```
            This will train and save the model files.
            """)
    st.stop()

# Sidebar inputs with enhanced design
st.sidebar.markdown("# 📋 Patient Information")
st.sidebar.markdown("---")

st.sidebar.markdown("## 👤 Demographics")
age = st.sidebar.slider("Age (years)", 21, 100, 30, help="Patient's age in years")
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 20, 0, help="Total number of pregnancies")

st.sidebar.markdown("## 🔬 Health Metrics")
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 100, help="Plasma glucose concentration")
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 130, 70, help="Diastolic blood pressure")
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20, help="Triceps skin fold thickness")
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 900, 80, help="2-Hour serum insulin")
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01, help="Diabetes likelihood based on family history")

st.sidebar.markdown("## ⚖️ Body Measurements")

weight = st.sidebar.number_input(
    "Weight (kg)",
    min_value=20.0,
    max_value=300.0,
    value=70.0,
    step=0.5
)

height_cm = st.sidebar.number_input(
    "Height (cm)",
    min_value=100.0,
    max_value=220.0,
    value=170.0,
    step=0.5
)

# Convert height to meters
height_m = height_cm / 100

# Auto-calculate BMI
bmi = round(weight / (height_m ** 2), 1)

st.sidebar.markdown(
    f"**📐 Calculated BMI:** `{bmi} kg/m²`"
)


# Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("🔮 Predict Diabetes Risk", type="primary", use_container_width=True)

# Main content 
if predict_button:
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Standardize input data
    input_std = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_std)[0]

    # Get probability if available
    try:
        prediction_proba = model.predict_proba(input_std)[0]
        prob_negative = prediction_proba[0] * 100
        prob_positive = prediction_proba[1] * 100
    except: 
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive

    # Display results with enhanced design
    st.markdown("---")
    st.markdown("## 📊 Prediction Results")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Prediction box with color-coded design
        if prediction == 0:
            if prob_positive < 30:
                st.markdown(
                    '<div class="result-card-success">'
                    '<h2 style="color: #065f46; margin: 0;">✅ LOW RISK</h2>'
                    '<h3 style="color: #047857; margin-top: 0.5rem;">Non-Diabetic Status</h3>'
                    '<p style="color: #065f46; margin-top: 1rem;">Your health parameters indicate a low risk of diabetes. Continue maintaining your healthy lifestyle.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="result-card-warning">'
                    '<h2 style="color: #92400e; margin: 0;">⚠️ MODERATE RISK</h2>'
                    '<h3 style="color: #b45309; margin-top: 0.5rem;">Non-Diabetic Status</h3>'
                    '<p style="color: #92400e; margin-top: 1rem;">While not diabetic, some risk factors need attention. Consider lifestyle modifications.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )
        else:
            if prob_positive > 70:
                st.markdown(
                    '<div class="result-card-danger">'
                    '<h2 style="color: #991b1b; margin: 0;">🚨 HIGH RISK</h2>'
                    '<h3 style="color: #dc2626; margin-top: 0.5rem;">Diabetic Status Indicated</h3>'
                    '<p style="color: #991b1b; margin-top: 1rem;">Immediate medical consultation recommended. Your health metrics suggest diabetes.</p>'
                    '</div>',
                    unsafe_allow_html=True
                )

        # Probabilities with enhanced metrics
        st.markdown("### 📈 Probability Breakdown")
        pcol1, pcol2 = st.columns(2)
        
        with pcol1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="color: #10b981; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">NON-DIABETIC</div>'
                f'<div style="font-size: 2.5rem; font-weight: 700; color: #059669;">{prob_negative:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        with pcol2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="color: #ef4444; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">DIABETIC</div>'
                f'<div style="font-size: 2.5rem; font-weight: 700; color: #dc2626;">{prob_positive:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    with col2:
        # Enhanced Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_positive,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Percentage", 'font': {'size': 20, 'family': 'Poppins', 'color': '#1f2937'}},
            number={'font': {'size': 48, 'family': 'Poppins', 'color': '#1f2937'}, 'suffix': '%'},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#9ca3af"},
                'bar': {'color': "#667eea", 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 3,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 30], 'color': '#d1fae5'},
                    {'range': [30, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#fee2e2'}
                ],
                'threshold': {
                    'line': {'color': "#1f2937", 'width': 6},
                    'thickness': 0.8,
                    'value': prob_positive
                }
            }
        ))
        fig.update_layout(
            font={'color': "#1f2937", 'family': "Poppins"},
            height=350,
            margin=dict(t=60, b=10, l=20, r=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk factors with improved layout
    st.markdown("---")
    st.markdown("## 🔍 Health Analysis")

    risk_factors = []
    positive_factors = []
    suggestions = []

    # Glucose level
    if glucose > 125:
        risk_factors.append(("High Glucose Level", f"{glucose} mg/dL (Normal: 70-125)"))
        suggestions.append("Maintain a balanced diet and regular exercise to manage glucose levels.")
    elif glucose < 70:
        risk_factors.append(("Low Glucose Level", f"{glucose} mg/dL (Normal: 70-125)"))
        suggestions.append("Ensure regular meals and monitor glucose levels closely.")
    else:
        positive_factors.append(f"Glucose levels are within the normal range ({glucose} mg/dL)")

    # BMI 
    if bmi >= 30:
        risk_factors.append(("Obesity (High BMI)", f"{bmi:.1f} kg/m² (Healthy: 18.5-24.9)"))
        suggestions.append("Incorporate physical activity and a healthy diet to manage weight.")
    elif 25 <= bmi < 30:
        risk_factors.append(("Overweight Status", f"{bmi:.1f} kg/m² (Healthy: 18.5-24.9)"))
        suggestions.append("Consider lifestyle changes to achieve a healthier weight.")
    elif 18.5 <= bmi < 25:
        positive_factors.append(f"BMI is within the healthy range ({bmi:.1f} kg/m²)")
    else:
        risk_factors.append(("Underweight (Low BMI)", f"{bmi:.1f} kg/m² (Healthy: 18.5-24.9)"))
        suggestions.append("Ensure adequate nutrition and consult a healthcare provider if necessary.")

    # Blood Pressure
    if bp >= 130:
        risk_factors.append(("High Blood Pressure", f"{bp} mm Hg (Normal: 80-120)"))
        suggestions.append("Monitor blood pressure regularly and reduce sodium intake.")   
    elif bp < 80:
        risk_factors.append(("Low Blood Pressure", f"{bp} mm Hg (Normal: 80-120)"))
        suggestions.append("Stay hydrated and avoid sudden position changes.")
    else:
        positive_factors.append(f"Blood pressure is within the normal range ({bp} mm Hg)")

    # Diabetes Pedigree Function
    if dpf > 0.5:
        risk_factors.append(("Elevated Family Risk", f"DPF: {dpf:.3f}"))
        suggestions.append("Family history indicates higher risk; regular check-ups are advised.")
    
    # Age factor
    if age > 45:
        risk_factors.append(("Age Factor", f"{age} years"))
        suggestions.append("Age increases diabetes risk; maintain regular health screenings.")

    # Display risk factors in cards
    if risk_factors:
        st.markdown("### ⚠️ Identified Risk Factors")
        cols = st.columns(2)
        for idx, (factor, detail) in enumerate(risk_factors):
            with cols[idx % 2]:
                st.markdown(
                    f'<div class="metric-card" style="border-left-color: #ef4444; margin-bottom: 1rem;">'
                    f'<div style="font-weight: 600; color: #dc2626; margin-bottom: 0.5rem;">{factor}</div>'
                    f'<div style="color: #6b7280; font-size: 0.9rem;">{detail}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    if suggestions:
        st.markdown("### 💡 Recommended Actions")
        for suggestion in suggestions:
            st.info(f"➤ {suggestion}")

    if positive_factors:
        st.markdown("### ✅ Positive Health Indicators")
        for positive in positive_factors:
            st.success(f"✓ {positive}")

    # Recommendations
    st.markdown("---")
    st.markdown("## 📋 Medical Recommendations")

    if prediction == 1:
        st.error("""
        **🚨 Immediate Actions Required:**
        
        - **Consult a healthcare professional** immediately for comprehensive evaluation and management
        - Get comprehensive diabetes screening and HbA1c test
        - Monitor blood sugar levels regularly (fasting and post-meal)
        - Engage in regular physical activity (at least 150 minutes of moderate exercise per week)
        - Adopt a balanced diet rich in whole grains, fruits, vegetables, and lean proteins
        - Limit intake of refined sugars and processed foods
        - Consider working with a certified diabetes educator
        """)
    else:
        st.success("""
        **✅ Preventive Measures:**
        
        - Maintain a healthy weight through balanced diet and regular exercise
        - Monitor blood sugar levels periodically (annual screening recommended)
        - Limit intake of sugary foods and beverages
        - Stay physically active to enhance insulin sensitivity
        - Consume a diet rich in fiber, whole grains, and vegetables
        - Schedule regular health check-ups to monitor risk factors
        - Maintain proper hydration (8-10 glasses of water daily)
        - Manage stress through meditation, yoga, or other relaxation techniques
        """)

    # Enhanced disclaimer
    st.markdown("---")
    st.markdown(
        '<div class="disclaimer">'
        '<strong>⚠️ Medical Disclaimer:</strong> This prediction tool is for informational and educational purposes only. '
        'It should not replace professional medical advice, diagnosis, or treatment. Always consult with your physician '
        'or other qualified health provider regarding any medical condition or health concerns. Do not disregard professional '
        'medical advice or delay seeking it because of information from this tool.'
        '</div>',
        unsafe_allow_html=True
    )

    # PDF Download with fixed function
    st.markdown("---")
    st.markdown("## 📥 Export Your Report")

    def generate_pdf():

        def safe_multicell(pdf, text, h=5):
            pdf.set_x(10)
            pdf.multi_cell(0, h, text)

        """Generate a comprehensive PDF report with proper error handling"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        try:
            # Try to load custom fonts, fallback to standard fonts if not available
            try:
                pdf.add_font(family="DejaVuSans", style="", fname="fonts/DejaVuSans.ttf", uni=True)
                pdf.add_font(family="DejaVuSans", style="B", fname="fonts/DejaVuSans-Bold.ttf", uni=True)
                use_custom_font = True
            except:
                use_custom_font = False
            
            # Header
            if use_custom_font:
                pdf.set_font("DejaVuSans", "B", 18)
            else:
                pdf.set_font("Arial", "B", 18)
            
            pdf.set_text_color(102, 126, 234)
            pdf.cell(0, 12, "Diabetes Risk Prediction Report", ln=1, align="C")
            pdf.set_text_color(0, 0, 0)
            
            # Date
            if use_custom_font:
                pdf.set_font("DejaVuSans", "", 10)
            else:
                pdf.set_font("Arial", "", 10)
            
            pdf.cell(0, 6, f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=1, align="C")
            pdf.ln(8)

            # Patient Information Section
            if use_custom_font:
                pdf.set_font("DejaVuSans", "B", 14)
            else:
                pdf.set_font("Arial", "B", 14)
            
            pdf.set_fill_color(230, 230, 250)
            pdf.cell(0, 10, "Patient Information", ln=1, fill=True)
            pdf.ln(4)
            
            if use_custom_font:
                pdf.set_font("DejaVuSans", "", 11)
            else:
                pdf.set_font("Arial", "", 11)
            
            # Patient details in a clean format
            patient_info = [
                ("Age:", f"{age} years"),
                ("Pregnancies:", f"{pregnancies}"),
                ("Glucose:", f"{glucose} mg/dL"),
                ("Blood Pressure:", f"{bp} mm Hg"),
                ("Skin Thickness:", f"{skin} mm"),
                ("Insulin:", f"{insulin} mu U/ml"),
                ("BMI:", f"{bmi:.1f} kg/m2"),
                ("Diabetes Pedigree Function:", f"{dpf:.3f}")
            ]
            
            for label, value in patient_info:
                pdf.cell(70, 6, label, border=0)
                pdf.cell(0, 6, value, ln=1)
            
            pdf.ln(8)

            # Prediction Result Section
            if use_custom_font:
                pdf.set_font("DejaVuSans", "B", 14)
            else:
                pdf.set_font("Arial", "B", 14)
            
            pdf.set_fill_color(230, 230, 250)
            pdf.cell(0, 10, "Prediction Result", ln=1, fill=True)
            pdf.ln(4)
            
            if use_custom_font:
                pdf.set_font("DejaVuSans", "", 11)
            else:
                pdf.set_font("Arial", "", 11)

            # Determine outcome and risk level
            if prediction == 1:
                outcome = "Diabetic - High Risk"
                pdf.set_text_color(220, 38, 38)
            else:
                outcome = "Non-Diabetic"
                pdf.set_text_color(16, 185, 129)
            
            if use_custom_font:
                pdf.set_font("DejaVuSans", "B", 12)
            else:
                pdf.set_font("Arial", "B", 12)
            
            pdf.cell(0, 8, f"Outcome: {outcome}", ln=1)
            pdf.set_text_color(0, 0, 0)
            
            if use_custom_font:
                pdf.set_font("DejaVuSans", "", 11)
            else:
                pdf.set_font("Arial", "", 11)
            
            risk_text = (
                "LOW RISK" if prediction == 0 and prob_positive < 30 else
                "MODERATE RISK" if prediction == 0 else
                "HIGH RISK" if prob_positive > 70 else
                "ELEVATED RISK"
            )
            
            pdf.cell(0, 6, f"Estimated Diabetes Probability: {prob_positive:.1f}%", ln=1)
            pdf.cell(0, 6, f"Risk Category: {risk_text}", ln=1)
            pdf.ln(8)

            # Add Gauge Chart
            try:
                if use_custom_font:
                    pdf.set_font("DejaVuSans", "B", 12)
                else:
                    pdf.set_font("Arial", "B", 12)
                
                pdf.cell(0, 8, "Risk Gauge Visualization", ln=1)
                pdf.ln(2)

                # Save plotly figure to PNG bytes
                img_bytes = io.BytesIO()
                pio.write_image(fig, img_bytes, format="png", engine="kaleido")
                img_bytes.seek(0)

                # Add image to PDF
                pdf.image(img_bytes, x=15, w=180)
                pdf.ln(10)
                pdf.set_x(10)
            except Exception as e:
                pdf.cell(0, 6, "(Visualization could not be included)", ln=1)
                pdf.ln(2)

            # Risk Factors & Suggestions
            if risk_factors or suggestions or positive_factors:
                if use_custom_font:
                    pdf.set_font("DejaVuSans", "B", 14)
                else:
                    pdf.set_font("Arial", "B", 14)
                
                pdf.set_fill_color(230, 230, 250)
                pdf.cell(0, 10, "Health Analysis", ln=1, fill=True)
                pdf.ln(4)
                
                if use_custom_font:
                    pdf.set_font("DejaVuSans", "", 11)
                else:
                    pdf.set_font("Arial", "", 11)

                if risk_factors:
                    pdf.set_text_color(220, 38, 38)
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "B", 11)
                    else:
                        pdf.set_font("Arial", "B", 11)
                    
                    pdf.cell(0, 8, "Identified Risk Factors:", ln=1)
                    pdf.set_text_color(0, 0, 0)
                    
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "", 10)
                    else:
                        pdf.set_font("Arial", "", 10)
                    
                    for factor, detail in risk_factors:
                        pdf.set_x(10)
                        pdf.multi_cell(0, 5, f"  - {factor}: {detail}")
                    pdf.ln(2)

                if suggestions:
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "B", 11)
                    else:
                        pdf.set_font("Arial", "B", 11)
                    
                    pdf.cell(0, 8, "Recommended Actions:", ln=1)
                    
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "", 10)
                    else:
                        pdf.set_font("Arial", "", 10)
                    
                    for sug in suggestions:
                        pdf.set_x(10)
                        pdf.multi_cell(0, 5, f"  - {sug}")
                    pdf.ln(2)

                if positive_factors:
                    pdf.set_text_color(16, 185, 129)
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "B", 11)
                    else:
                        pdf.set_font("Arial", "B", 11)
                    
                    pdf.cell(0, 8, "Positive Health Indicators:", ln=1)
                    pdf.set_text_color(0, 0, 0)
                    
                    if use_custom_font:
                        pdf.set_font("DejaVuSans", "", 10)
                    else:
                        pdf.set_font("Arial", "", 10)
                    
                    for pos in positive_factors:
                        pdf.set_x(10)
                        pdf.multi_cell(0, 5, f"  - {pos}")

            # Disclaimer
            pdf.ln(10)
            pdf.set_fill_color(254, 243, 199)
            pdf.set_text_color(120, 53, 15)
            pdf.set_font("Arial", "B", 10)

            pdf.cell(0, 6, "MEDICAL DISCLAIMER", ln=1, fill=True)

            pdf.set_font("Arial", "", 9)
            pdf.set_text_color(0, 0, 0)
            pdf.set_x(10)
            pdf.multi_cell(
                0, 5,
                "This report is for informational purposes only and does not replace professional medical advice. "
                "Always consult with a qualified healthcare provider regarding any medical condition."
            )

            return bytes(pdf.output())
        
        except Exception as e:
            # If there's any error, create a simple text-based PDF
            st.error(f"Error generating detailed PDF: {str(e)}. Creating simplified version...")
            
            simple_pdf = FPDF()
            simple_pdf.add_page()
            simple_pdf.set_font("Arial", "B", 16)
            simple_pdf.cell(0, 10, "Diabetes Risk Prediction Report", ln=1, align="C")
            simple_pdf.set_font("Arial", "", 11)
            simple_pdf.ln(5)
            simple_pdf.multi_cell(0, 6, f"Date: {datetime.now().strftime('%B %d, %Y')}")
            simple_pdf.multi_cell(0, 6, f"Prediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
            simple_pdf.multi_cell(0, 6, f"Risk Probability: {prob_positive:.1f}%")
            
            return bytes(simple_pdf.output())

    try:
        pdf_bytes = generate_pdf()
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.download_button(
                label="📄 Download Complete PDF Report",
                data=pdf_bytes,
                file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    except Exception as e:
        st.error(f"Unable to generate PDF: {str(e)}")
        st.info("Please ensure all required dependencies are installed (fpdf, kaleido for plotly)")

else:
    # Enhanced initial page
    st.markdown("---")
    st.markdown(
        '<div class="info-box">'
        '<h3 style="margin-top: 0; color: #4c1d95;"> How to Use This Tool</h3>'
        '<p style="color: #5b21b6; margin-bottom: 0;">Enter patient health information in the sidebar and click '
        '<strong>"Predict Diabetes Risk"</strong> to receive a comprehensive analysis powered by machine learning.</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Model information cards
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            '<div class="metric-card">'
            '<div style="color: #667eea; font-size: 2.5rem; margin-bottom: 0.5rem;">🧠</div>'
            '<div style="font-weight: 600; font-size: 1.1rem; color: #1f2937;">Model Type</div>'
            '<div style="color: #6b7280; margin-top: 0.5rem;">SVM Classifier</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            '<div class="metric-card">'
            '<div style="color: #10b981; font-size: 2.5rem; margin-bottom: 0.5rem;">📊</div>'
            '<div style="font-weight: 600; font-size: 1.1rem; color: #1f2937;">Accuracy</div>'
            '<div style="color: #6b7280; margin-top: 0.5rem;">78%</div>'
            '</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            '<div class="metric-card">'
            '<div style="color: #f59e0b; font-size: 2.5rem; margin-bottom: 0.5rem;">📈</div>'
            '<div style="font-weight: 600; font-size: 1.1rem; color: #1f2937;">Training Data</div>'
            '<div style="color: #6b7280; margin-top: 0.5rem;">768 Samples</div>'
            '</div>',
            unsafe_allow_html=True
        )

    # Additional information
    st.markdown("---")
    st.markdown("### About This Predictor")
    st.markdown("""
    This diabetes risk prediction tool uses a **Support Vector Machine (SVM)** classifier trained on the 
    Pima Indians Diabetes Database. The model analyzes eight key health parameters to estimate the 
    likelihood of diabetes.
    
    **Key Features:**
    - Real-time risk assessment based on clinical parameters
    - Visual risk gauge for easy interpretation
    - Personalized health recommendations
    - Comprehensive PDF report generation
    - Evidence-based risk factor analysis
    
    **Parameters Used:**
    - **Pregnancies:** Number of times pregnant
    - **Glucose:** Plasma glucose concentration (2 hours in oral glucose tolerance test)
    - **Blood Pressure:** Diastolic blood pressure (mm Hg)
    - **Skin Thickness:** Triceps skin fold thickness (mm)
    - **Insulin:** 2-Hour serum insulin (mu U/ml)
    - **BMI:** Body mass index (weight in kg/(height in m)²)
    - **Diabetes Pedigree Function:** Genetic influence score
    - **Age:** Age in years
    """)