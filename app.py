import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import google.generativeai as genai
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Project Nova Dashboard",
    page_icon="✨",
    layout="wide"
)

# --- Professional CSS Styling ---
page_style = """
    <style>
        /* Main Theme */
        :root {
            --success-color: #28a745;
            --error-color: #dc3545;
            --card-border-color: #dee2e6;
            --light-text-color: #6c757d;
        }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(15px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Final Card Styling */
        .decision-card {
            border-radius: 8px;
            padding: 0.5rem;
            text-align: center;
            background-color: transparent; /* MODIFIED: Removed white background */
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            transition: all 0.2s ease-in-out;
            animation: fadeIn 0.5s ease-out forwards;
            height: 100%;
        }
        .decision-card:hover {
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transform: translateY(-3px);
        }
        
        .decision-card h3 {
            margin: 0 0 0.2rem 0;
            color: var(--light-text-color);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .decision-card p {
            font-size: 2rem;
            margin: 0;
        }
        .decision-card h2 {
            margin-top: 0.2rem;
            font-size: 1.2rem;
            font-weight: 700;
        }
        .decision-card .approved-text { color: var(--success-color); }
        .decision-card .denied-text { color: var(--error-color); }
    </style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# --- Configure Google AI ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    AI_ENABLED = True
except (FileNotFoundError, KeyError):
    AI_ENABLED = False

# --- Load Resources ---
@st.cache_resource
def load_resources():
    required_files = ['nova_regression_pipeline.joblib', 'fairness_mitigator.joblib', 
                      'loan_approval_threshold.joblib', 'model_columns.joblib', 'data/partners_synthetic_upgraded.csv']
    for file in required_files:
        if not Path(file).exists():
            st.error(f"Error: `{file}` not found. Please re-run the main training script first.")
            st.stop()
    pipeline = joblib.load('nova_regression_pipeline.joblib')
    mitigator = joblib.load('fairness_mitigator.joblib')
    threshold = joblib.load('loan_approval_threshold.joblib')
    columns = joblib.load('model_columns.joblib')
    full_df = pd.read_csv('data/partners_synthetic_upgraded.csv')
    return pipeline, mitigator, threshold, columns, full_df

pipeline, mitigator, threshold, model_columns, df = load_resources()

# --- Gauge Chart Function ---
def create_advanced_gauge(score, threshold):
    colors = ["#dc3545", "#ffc107", "#28a745"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Nova Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [300, 900], 'tickfont': {'size': 10}}, 'bar': {'color': "#343a40", 'thickness': 0.2},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "#dee2e6",
            'steps': [{'range': [300, 550], 'color': colors[0]}, {'range': [550, 650], 'color': colors[1]}, {'range': [650, 900], 'color': colors[2]}],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.9, 'value': threshold}
        }
    ))
    fig.update_layout(font={'family': "Arial"}, margin=dict(l=10, r=10, t=40, b=10, pad=0), height=200)
    return fig

# --- App Header ---
st.title("✨ Project Nova: Credit Assessment Dashboard")

# --- Sidebar ---
st.sidebar.header("Find a Partner")
st.sidebar.subheader("Filter by Demographics")
selected_tiers = st.sidebar.multiselect("City Tier", options=sorted(df['city_tier'].unique()))
selected_ages = st.sidebar.multiselect("Age Group", options=sorted(df['age_group'].unique()))
selected_genders = st.sidebar.multiselect("Gender", options=sorted(df['gender'].unique()))

filtered_df = df.copy()
if selected_tiers:
    filtered_df = filtered_df[filtered_df['city_tier'].isin(selected_tiers)]
if selected_ages:
    filtered_df = filtered_df[filtered_df['age_group'].isin(selected_ages)]
if selected_genders:
    filtered_df = filtered_df[filtered_df['gender'].isin(selected_genders)]

st.sidebar.subheader("Search and Select")
search_term = st.sidebar.text_input("Search by Partner Name")

if search_term:
    filtered_df = filtered_df[filtered_df['partner_name'].str.contains(search_term, case=False)]

filtered_df['display'] = filtered_df['partner_id'].astype(str) + " - " + filtered_df['partner_name']
display_options = filtered_df['display'].tolist()

if not display_options:
    st.sidebar.warning("No partners match the current filters.")
    st.stop()

selected_display_option = st.sidebar.selectbox("Partner", display_options)
selected_partner_id = int(selected_display_option.split(" - ")[0])

# --- Get Data and Make Predictions ---
partner_data = df[df['partner_id'] == selected_partner_id]
partner_name = partner_data['partner_name'].iloc[0]
input_df = partner_data[model_columns]
nova_score = pipeline.predict(input_df)[0]
baseline_decision = "Approved" if nova_score > threshold else "Denied"
input_df_processed = pipeline.named_steps['prep'].transform(input_df)
fair_decision = "Approved" if mitigator.predict(input_df_processed)[0] == 1 else "Denied"

# --- Professional Dashboard Layout ---
st.header(f"Assessment for: {partner_name} (ID: {selected_partner_id})")

# --- Top Row: Gauge and Key Metrics ---
with st.container(border=True):
    col1, col2 = st.columns([2, 3])
    with col1:
        st.plotly_chart(create_advanced_gauge(nova_score, threshold), use_container_width=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        kpi_row1_col1, kpi_row1_col2 = st.columns(2)
        with kpi_row1_col1:
            st.metric(label="Weekly Earnings", value=f"${partner_data['weekly_earnings'].iloc[0]:.0f}")
        with kpi_row1_col2:
            st.metric(label="Avg. Rating", value=f"{partner_data['avg_rating'].iloc[0]:.2f} ⭐")
        
        kpi_row2_col1, kpi_row2_col2 = st.columns(2)
        with kpi_row2_col1:
            st.metric(label="Cancel Rate", value=f"{partner_data['cancel_rate'].iloc[0]:.1%}")
        with kpi_row2_col2:
            st.metric(label="Safety Incidents", value=f"{partner_data['safety_incidents'].iloc[0]}")

# --- Second Row of KPIs ---
with st.container(border=True):
    kpi2_col1, kpi2_col2, kpi2_col3 = st.columns(3)
    kpi2_col1.metric(label="Loyalty", value=f"{partner_data['loyalty_years'].iloc[0]} years")
    kpi2_col2.metric(label="Weekly Trips", value=f"{partner_data['trips_weekly'].iloc[0]:.0f}")
    kpi2_col3.metric(label="On-Time Percentage", value=f"{partner_data['on_time_pct'].iloc[0]:.1%}")

# --- Model Decisions ---
st.markdown("### Model Decisions")
col1, col2 = st.columns(2)
with col1:
    baseline_class = "approved" if baseline_decision == "Approved" else "denied"
    baseline_icon = '✅' if baseline_decision == "Approved" else '❌'
    st.markdown(f"""
    <div class="decision-card {baseline_class}">
        <h3>Baseline Model</h3><p>{baseline_icon}</p>
        <h2 class="decision-text {baseline_class}-text">{baseline_decision}</h2>
    </div>""", unsafe_allow_html=True)

with col2:
    fair_class = "approved" if fair_decision == "Approved" else "denied"
    fair_icon = '✅' if fair_decision == "Approved" else '❌'
    st.markdown(f"""
    <div class="decision-card {fair_class}">
        <h3>Fair Mitigated Model ✨</h3><p>{fair_icon}</p>
        <h2 class="decision-text {fair_class}-text">{fair_decision}</h2>
    </div>""", unsafe_allow_html=True)

# --- AI Feedback Section ---
if AI_ENABLED:
    st.markdown("---")
    if st.button("🤖 Get AI Financial Coach Feedback"):
        with st.spinner("Our AI coach is analyzing the data..."):
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            data_summary = partner_data[['weekly_earnings', 'avg_rating', 'cancel_rate', 'on_time_pct', 'safety_incidents', 'loyalty_years']].to_string(index=False)
            prompt = f"""
            You are a friendly financial coach for a gig economy partner. A partner's data is provided below:
            {data_summary}
            Based on this data, provide 2-3 actionable tips to help this partner improve their financial standing and 'Nova Score'.
            Focus on the areas where they can make the biggest impact. Frame the advice in a positive and supportive tone.
            Start with "Here are a few tips to help you grow:". Use bullet points.
            """
            try:
                response = model.generate_content(prompt)
                st.success("**AI Financial Coach Feedback:**")
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Sorry, there was an error generating feedback: {e}")