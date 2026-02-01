import plotly
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import io
import plotly.io as pio
from datetime import datetime

#Page Configuration
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)
#Custom CSS
st.markdown(
    """
    <style>
    .main { padding: 2rem 1rem 10rem 1rem; }
    .stAlert { padding: 0rem 1rem 0rem 1rem; border-radius: 0.5rem; }
    .h1 { color: #4B0082; font-weight: bold; padding-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True
)

#Load Model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler_svm.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Model or scaler file not found. Please ensure 'diabetes_model.pkl' and 'scaler.pkl' are in the working directory.")
        return None, None
    
#header
st.title("Diabetes Predictor 🩺", anchor=False)
st.markdown("### Predict the likelihood of diabetes based on health parameters using a trained machine learning model.")

#Load Model
model, scaler = load_model_and_scaler()

if model is None or scaler is None:
    st.error("Unable to load model or scaler. Please check the files and try again.")
    st.info("""
            Please run the following command first:
            '''
            python diabetes_prediction.py
            '''
            This will train and save modelfiles.
            """)
    st.stop()


#Sidebar inputs
st.sidebar.title("Patient Information")

st.sidebar.subheader("Demographics")
age = st.sidebar.slider("Age (years)", 21, 100, 30)
pregnancies = st.sidebar.slider("Number of Pregnancies", 0, 20, 0)

st.sidebar.subheader("Health Metrics")
glucose = st.sidebar.slider("Glucose Level (mg/dL)", 0, 200, 100)
bp = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 130, 70)
skin = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level (mu U/ml)", 0, 900, 80)
bmi = st.sidebar.slider("BMI (kg/m²)", 10.0, 70.0, 25.0, 0.1)
dpf = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)

#Predict button
st.sidebar.markdown("---")
predict_button = st.sidebar.button("Predict Diabetes", type="primary", use_container_width=True)

#Main content 
if predict_button:
    #prepare input data
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    #standardize input data
    input_std = scaler.transform(input_data)

    #make prediction
    prediction = model.predict(input_std)[0]

    #get probability if available
    try:
        prediction_proba = model.predict_proba(input_std)[0]
        prob_negative = prediction_proba[0] * 100
        prob_positive = prediction_proba[1] * 100
    except: 
        prob_positive = 100 if prediction == 1 else 0
        prob_negative = 100 - prob_positive

    #Display results
    st.markdown("---")
    st.header("Diabetes Prediction Results")

    col1, col2 = st.columns([2,1])

    with col1:
        #predction box
        if prediction == 0:
            if prob_positive <30:
                st.success("### LOW_RISK - Non Diabetic 🟢")
            else:
                st.warning("### MODERATE_RISK - Non Diabetic 🟡")
   
        else:
            if prob_positive > 70:
                st.error("### HIGH_RISK - Diabetic 🔴")

        #probabilities
        st.subheader("Probabilty Breakdown")
        pcol1, pcol2 = st.columns(2)
        pcol1.metric(" Non-Diabetic", f"{prob_negative:.2f} %")
        pcol2.metric(" Diabetic", f"{prob_positive:.2f} %")

    with col2:
        #Gauge chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = prob_positive,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Diabetes Risk Level", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "red"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prob_positive
                }
            }
        ))
        fig.update_layout(font={'color': "darkblue", 'family': "Arial"}, height=400, margin=dict(t=50, b=20, l=20, r=20) )
        st.plotly_chart(fig, use_container_width=True)
    
    #Risk factors
    st.markdown("---")
    st.subheader("Diabetes Risk Factor Analysis")

    risk_factors = []
    positive_factors = []
    suggestions = []

    #glucose level
    if glucose > 125:
        risk_factors.append("High Glucose Level")
        suggestions.append("Maintain a balanced diet and regular exercise to manage glucose levels.")
    elif glucose < 100:
        risk_factors.append("Low Glucose Level")
        suggestions.append("Ensure regular meals and monitor glucose levels.")
    else:
        positive_factors.append("Glucose levels are within the normal range.")

    #BMI 
    if bmi >= 30:
        risk_factors.append("High BMI - Obesity")
        suggestions.append("Incorporate physical activity and a healthy diet to manage weight.")
    elif 25 <= bmi < 30:
        risk_factors.append("Risk at being Overweight ")
        suggestions.append("Consider lifestyle changes to achieve a healthier weight.")
    elif 18.5 <= bmi < 24.9:
        positive_factors.append("BMI is within the healthy range.")
    else:
        risk_factors.append("Low BMI - Underweight")
        suggestions.append("Ensure adequate nutrition and consult a healthcare provider if necessary.")

    #Blood Pressure
    if bp >= 130:
        risk_factors.append("High Blood Pressure")
        suggestions.append("Monitor blood pressure regularly and reduce sodium intake.")   
    elif bp < 80:
        risk_factors.append("Low Blood Pressure")
        suggestions.append("Stay hydrated and avoid sudden position changes.")
    else:
        positive_factors.append("Blood pressure is within the normal range.")

    #Diabetes Pedigree Function
    if dpf > 0.5:
        risk_factors.append("High Diabetes Pedigree Function")
        positive_factors.append("Family history indicates higher risk; regular check-ups are advised.")
    
    #Age factor
    if age > 45:
        risk_factors.append("Advanced Age")
        suggestions.append("Age increases risk; maintain regular health screenings.")

    if risk_factors:
        st.warning("### Identified Risk Factors:")
        for factor in risk_factors:
            st.markdown(f"- {factor}")

    if suggestions:
        st.info("### Suggestions for Risk Reduction:")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")

    if positive_factors:
        st.success("### Positive Health Indicators:")
        for positive in positive_factors:
            st.markdown(f"- {positive}")


    #Recommendations
    st.markdown("---")
    st.subheader("Recommendations")

    if prediction == 1:
        st.error("""
        *** Important Actions: ***
        - Consult a healthcare professional immediately for comprehensive evaluation and management.
        - Get comprehensive bdiabetes screening.
        - Monitor blood sugar levels regularly.
        - Engage in regular physical activity (at least 150 minutes of moderate exercise per week).
        - Adopt a balanced diet rich in whole grains, fruits, and vegetables.   
        """)
    else:
        st.success("""
        *** Preventive Measures: ***
        - Maintain a healthy weight through balanced diet and regular exercise.
        - Monitor blood sugar levels periodically.
        - Limit intake of sugary foods and beverages.
        - Stay physically active to enhance insulin sensitivity.
        - Schedule regular health check-ups to monitor risk factors.
        """)

    #Disclaimer
    st.markdown("---")
    st.caption("" \
    "⚠️ **Disclaimer:** This prediction tool is for informational purposes only and should not replace professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.")

   #PDF Download
    st.markdown("---")
    st.subheader("Export Results")

    

    def generate_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_font(family="DejaVuSans", style="", fname="fonts/DejaVuSans.ttf", uni=True)
        pdf.add_font(family="DejaVuSans", style="B", fname="fonts/DejaVuSans-Bold.ttf", uni=True)
        
        #header
        pdf.set_font("DejaVuSans", "B", 16)
        pdf.cell(0, 10, "Diabetes Prediction Report", ln=1, align="C")
        pdf.ln(8)

        #patient information
        pdf.set_font("DejaVuSans", "", 11)
        pdf.cell(0, 10, "Patient Information", ln=1)
        pdf.set_font("DejaVuSans", "", 11)
        pdf.cell(60, 6, "Age:", border=0)
        pdf.cell(0, 6, f"{age} years", ln=1)
        pdf.cell(60, 6, "Pregnancies:", border=0)
        pdf.cell(0, 6, f"{pregnancies}", ln=1)
        pdf.cell(60, 6, "Glucose:", border=0)
        pdf.cell(0, 6, f"{glucose} mg/dL", ln=1)
        pdf.cell(60, 6, "Blood Pressure:", border=0)
        pdf.cell(0, 6, f"{bp} mm Hg", ln=1)
        pdf.cell(60, 6, "Skin Thickness:", border=0)
        pdf.cell(0, 6, f"{skin} mm", ln=1)
        pdf.cell(60, 6, "Insulin:", border=0)
        pdf.cell(0, 6, f"{insulin} mu U/ml", ln=1)
        pdf.cell(60, 6, "BMI:", border=0)
        pdf.cell(0, 6, f"{bmi:.1f} kg/m²", ln=1)
        pdf.cell(60, 6, "Diabetes Pedigree Function:", border=0)
        pdf.cell(0, 6, f"{dpf:.3f}", ln=1)
        pdf.ln(8)

        #prediction result
        pdf.set_font("DejaVuSans", "B", 12)
        pdf.cell(0, 8, "Prediction Result", ln=1)
        pdf.set_font("DejaVuSans", "", 11)

        outcome = "Diabetic – High Risk" if prediction == 1 else "Non-Diabetic"
        risk_text = (
            "LOW RISK" if prediction == 0 and prob_positive < 30 else
            "MODERATE RISK" if prediction == 0 else
            "HIGH RISK" if prob_positive > 70 else
            "ELEVATED RISK"
        )

        pdf.set_text_color(200, 0, 0) if prediction == 1 else pdf.set_text_color(0, 128, 0)
        pdf.cell(0, 8, f"Outcome: {outcome}", ln=1)
        pdf.set_text_color(0, 0, 0)  # reset color
        pdf.cell(0, 6, f"Estimated Diabetes Probability: {prob_positive:.1f}%", ln=1)
        pdf.cell(0, 6, f"Risk Category: {risk_text}", ln=1)
        pdf.ln(6)

        # Add Gauge Chart as image
        pdf.set_font("DejaVuSans", "B", 11)
        pdf.cell(0, 8, "Risk Gauge Visualization", ln=1)

        # Save plotly figure to PNG bytes
        img_bytes = io.BytesIO()
        fig.write_image(img_bytes, format="png", scale=2)  # higher scale = better quality
        img_bytes.seek(0)

        # Add image to PDF (centered, width 140mm ~ half page)
        pdf.image(img_bytes, x=35, y=None, w=140)

        pdf.ln(10)  # space after image

        # Risk Factors & Suggestions
        if risk_factors or suggestions or positive_factors:
            pdf.set_font("DejaVuSans", "B", 12)
            pdf.cell(0, 10, "Risk Factor Analysis", ln=1)
            pdf.set_font("DejaVuSans", "", 11)

            if risk_factors:
                pdf.set_text_color(200, 0, 0)
                pdf.cell(0, 8, "Identified Risk Factors:", ln=1)
                pdf.set_text_color(0, 0, 0)
                for factor in risk_factors:
                    pdf.multi_cell(0, 6, f"• {factor}")

            if suggestions:
                pdf.ln(4)
                pdf.cell(0, 8, "Suggested Actions:", ln=1)
                for sug in suggestions:
                    pdf.multi_cell(0, 6, f"  → {sug}")

            if positive_factors:
                pdf.ln(4)
                pdf.set_text_color(0, 128, 0)
                pdf.cell(0, 8, "Positive Indicators:", ln=1)
                pdf.set_text_color(0, 0, 0)
                for pos in positive_factors:
                    pdf.multi_cell(0, 6, f"✓ {pos}")

            pdf.ln(8)

        
        

        #disclaimer
        pdf.ln(6)
        pdf.set_font("DejaVuSans", "I", 9)
        pdf.multi_cell(
            0, 5,
            "WARNING: This report is for informational purposes only and does not replace medical advice."
        )

        return bytes(pdf.output(dest="S"))


    pdf_bytes = generate_pdf()

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="diabetes_report.pdf",
        mime="application/pdf"
    )


    
else:
    #Initial page
    st.markdown("---")
    st.info("Enter patient information in the sidebar and click 'Predict Diabetes' to see the results.", icon="ℹ️")

    col1,col2,col3 = st.columns(3)
    col1.metric("Model Type", "SVM Classifier")
    col2.metric("Accuracy", "78 %")
    col3.metric("Dataset", "768 Samples")
