# frontend_improved.py
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction - Improved",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        border-left: 5px solid #c92a2a;
    }
    .medium-risk {
        background: linear-gradient(135deg, #ffd93d 0%, #f6c23e 100%);
        color: #333;
        border-left: 5px solid #f59f00;
    }
    .low-risk {
        background: linear-gradient(135deg, #6bcf7f 0%, #51cf66 100%);
        color: white;
        border-left: 5px solid #2f9e44;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.2rem; color: #666;'>
        Predict customer churn with <strong>simplified inputs</strong> and get <strong>actionable recommendations</strong>
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
    ["🔍 Single Prediction", "📊 Batch Prediction", "📈 Analytics Dashboard", "ℹ️ About"])

# Backend API URL
API_URL = "http://localhost:8000"

# Check backend connection
@st.cache_data(ttl=300)
def check_backend_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else {}
    except:
        return False, {}

# Initialize session state
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# ============================================================================
# SINGLE PREDICTION MODE
# ============================================================================

if app_mode == "🔍 Single Prediction":
    st.header("🔍 Single Customer Prediction")
    
    # Check backend connection
    backend_ok, health_info = check_backend_health()
    if not backend_ok:
        st.error("❌ Backend API is not available. Please make sure the FastAPI server is running on port 8000.")
        st.info("💡 To start the backend, run: `uvicorn backend_improved:app --reload`")
    else:
        st.success("✅ Backend API is connected!")
        if health_info.get('optimal_threshold'):
            st.info(f"📊 Using optimal threshold: {health_info['optimal_threshold']:.2f}")
    
    # Two-column layout
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📝 Customer Information")
        st.markdown("**Only enter the most important information:**")
        
        with st.form("customer_form"):
            # Critical features only
            st.markdown("### ⭐ Critical Information (Required)")
            
            info_col1, info_col2 = st.columns(2)
            
            with info_col1:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, 
                                        help="How long the customer has been with the company")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, 
                                                  value=70.0, step=0.01, help="Monthly billing amount")
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, 
                                               value=1000.0, step=0.01, help="Total amount charged")
            
            with info_col2:
                contract = st.selectbox("Contract Type", 
                    ["Month-to-month", "One year", "Two year"],
                    help="Contract duration - month-to-month has highest churn risk")
                internet_service = st.selectbox("Internet Service", 
                    ["DSL", "Fiber optic", "No"],
                    help="Type of internet service")
                payment_method = st.selectbox("Payment Method", 
                    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                    help="Payment method - automatic payments have lower churn")
            
            # Optional features (collapsed by default)
            with st.expander("⚙️ Additional Information (Optional)", expanded=False):
                opt_col1, opt_col2 = st.columns(2)
                
                with opt_col1:
                    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
                    senior_citizen = st.selectbox("Senior Citizen", [0, 1], 
                                                 format_func=lambda x: "Yes" if x == 1 else "No")
                    partner = st.selectbox("Partner", ["Yes", "No"], index=1)
                    dependents = st.selectbox("Dependents", ["Yes", "No"], index=1)
                    phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
                
                with opt_col2:
                    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"], index=1)
                    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"], index=1)
                    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"], index=1)
                    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"], index=1)
                    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"], index=1)
                    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"], index=1)
                    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"], index=1)
                    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
            
            submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)
    
    # Prediction results
    with col2:
        st.subheader("📊 Prediction Results")
        
        if submitted:
            # Prepare customer data (simplified)
            customer_data = {
                "tenure": int(tenure),
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges),
                "Contract": contract,
                "InternetService": internet_service,
                "PaymentMethod": payment_method,
                # Optional fields
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "PaperlessBilling": paperless_billing
            }
            
            try:
                # Call prediction API
                with st.spinner("🔮 Analyzing customer data..."):
                    response = requests.post(f"{API_URL}/predict", json=customer_data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results with styling
                    churn_prob = result['churn_probability']
                    risk_level = result['risk_level']
                    
                    # Risk level styling
                    if risk_level == "High":
                        risk_class = "high-risk"
                        risk_emoji = "🔴"
                    elif risk_level == "Medium":
                        risk_class = "medium-risk"
                        risk_emoji = "🟡"
                    else:
                        risk_class = "low-risk"
                        risk_emoji = "🟢"
                    
                    # Prediction box
                    st.markdown(f'<div class="prediction-box {risk_class}">', unsafe_allow_html=True)
                    st.markdown(f"### {risk_emoji} {risk_level} Risk")
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                    st.metric("Prediction", "⚠️ Will Churn" if result['churn_prediction'] else "✅ Will Stay")
                    st.metric("Confidence", result['confidence'])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = churn_prob,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk Score"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.7], 'color': "yellow"},
                                {'range': [0.7, 1], 'color': "red"}],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.7}}))
                    
                    fig_gauge.update_layout(height=250)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
                    # Key factors
                    if result.get('key_factors'):
                        st.markdown("### 🔍 Key Risk Factors")
                        for factor in result['key_factors']:
                            st.markdown(f"- ⚠️ {factor}")
                    
                    # Recommendations
                    if result.get('recommendations'):
                        st.markdown("### 💡 Actionable Recommendations")
                        for rec in result['recommendations']:
                            priority_emoji = "🔴" if rec['priority'] == "High" else "🟡" if rec['priority'] == "Medium" else "🟢"
                            st.markdown(f"""
                            <div class="recommendation-box">
                                <strong>{priority_emoji} {rec['priority']} Priority:</strong> {rec['action']}<br>
                                <em>{rec['description']}</em><br>
                                <strong>Expected Impact:</strong> {rec['expected_impact']}
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.predictions_history.append({
                        **customer_data,
                        **result,
                        'timestamp': pd.Timestamp.now()
                    })
                    
                else:
                    st.error(f"❌ API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        
        else:
            st.info("👆 Fill out the form and click 'Predict Churn' to see results")
            
            # Sample gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge",
                value = 0,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Churn Risk Score"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgreen"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "red"}]}))
            
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ============================================================================
# BATCH PREDICTION MODE
# ============================================================================

elif app_mode == "📊 Batch Prediction":
    st.header("📊 Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with customer data (minimum required columns: `tenure`, `MonthlyCharges`, `TotalCharges`, 
    `Contract`, `InternetService`, `PaymentMethod`) or enter data manually.
    """)
    
    # Option to upload or enter manually
    input_method = st.radio("Input Method", ["Upload CSV", "Enter Manually"], horizontal=True)
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Choose CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded: {len(batch_data)} customers")
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(batch_data.head(10), use_container_width=True)
                
                # Check required columns
                required_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
                missing_cols = [col for col in required_cols if col not in batch_data.columns]
                
                if missing_cols:
                    st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🚀 Predict Batch", use_container_width=True):
                        # Convert to API format
                        customers_list = batch_data.to_dict('records')
                        
                        # Call batch prediction API
                        with st.spinner(f"🔮 Predicting churn for {len(customers_list)} customers..."):
                            response = requests.post(f"{API_URL}/batch_predict", json=customers_list, timeout=30)
                        
                        if response.status_code == 200:
                            results = response.json()
                            predictions_df = pd.DataFrame(results['predictions'])
                            summary = results['summary']
                            
                            st.success(f"✅ Predictions completed for {summary['total_customers']} customers!")
                            
                            # Display results in tabs
                            tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "📈 Summary", "💡 Recommendations", "📥 Download"])
                            
                            with tab1:
                                st.subheader("Prediction Results")
                                # Add color coding
                                def color_risk(val):
                                    if val == 'High':
                                        return 'background-color: #ff6b6b; color: white'
                                    elif val == 'Medium':
                                        return 'background-color: #ffd93d; color: #333'
                                    else:
                                        return 'background-color: #6bcf7f; color: white'
                                
                                styled_df = predictions_df.style.applymap(color_risk, subset=['risk_level'])
                                st.dataframe(styled_df, use_container_width=True)
                            
                            with tab2:
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Customers", summary['total_customers'])
                                with col2:
                                    st.metric("High Risk", summary['high_risk_count'], 
                                             delta=f"{summary['high_risk_count']/summary['total_customers']*100:.1f}%")
                                with col3:
                                    st.metric("Predicted Churn", summary['predicted_churn_count'])
                                with col4:
                                    st.metric("Avg Churn Probability", f"{summary['average_churn_probability']:.1%}")
                                
                                # Visualizations
                                fig_col1, fig_col2 = st.columns(2)
                                
                                with fig_col1:
                                    # Risk distribution pie chart
                                    risk_counts = {
                                        'High': summary['high_risk_count'],
                                        'Medium': summary['medium_risk_count'],
                                        'Low': summary['low_risk_count']
                                    }
                                    fig_pie = px.pie(
                                        values=list(risk_counts.values()),
                                        names=list(risk_counts.keys()),
                                        title="Risk Level Distribution",
                                        color_discrete_map={
                                            'High': '#ff6b6b',
                                            'Medium': '#ffd93d',
                                            'Low': '#6bcf7f'
                                        }
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)
                                
                                with fig_col2:
                                    # Churn probability distribution
                                    fig_hist = px.histogram(
                                        predictions_df,
                                        x='churn_probability',
                                        nbins=20,
                                        title="Churn Probability Distribution",
                                        labels={'churn_probability': 'Churn Probability', 'count': 'Number of Customers'}
                                    )
                                    fig_hist.update_traces(marker_color='#1f77b4')
                                    st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with tab3:
                                st.subheader("Top Recommendations")
                                if summary.get('top_recommendations'):
                                    for action, count in summary['top_recommendations'].items():
                                        st.markdown(f"**{action}** - Recommended for {count} customers")
                            
                            with tab4:
                                # Download results
                                csv = predictions_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Predictions as CSV",
                                    data=csv,
                                    file_name="churn_predictions.csv",
                                    mime="text/csv"
                                )
                        else:
                            st.error(f"❌ API Error: {response.text}")
                            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
    
    else:
        # Manual entry
        st.subheader("Enter Customer Data")
        num_customers = st.number_input("Number of customers", min_value=1, max_value=50, value=3)
        
        customers_data = []
        for i in range(num_customers):
            with st.expander(f"Customer {i+1}", expanded=(i==0)):
                col1, col2 = st.columns(2)
                with col1:
                    tenure = st.number_input(f"Tenure (months)", key=f"tenure_{i}", min_value=0, max_value=100, value=12)
                    monthly_charges = st.number_input(f"Monthly Charges ($)", key=f"monthly_{i}", min_value=0.0, max_value=200.0, value=70.0)
                    total_charges = st.number_input(f"Total Charges ($)", key=f"total_{i}", min_value=0.0, max_value=10000.0, value=1000.0)
                with col2:
                    contract = st.selectbox(f"Contract", ["Month-to-month", "One year", "Two year"], key=f"contract_{i}")
                    internet_service = st.selectbox(f"Internet Service", ["DSL", "Fiber optic", "No"], key=f"internet_{i}")
                    payment_method = st.selectbox(f"Payment Method", 
                        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
                        key=f"payment_{i}")
                
                customers_data.append({
                    "tenure": int(tenure),
                    "MonthlyCharges": float(monthly_charges),
                    "TotalCharges": float(total_charges),
                    "Contract": contract,
                    "InternetService": internet_service,
                    "PaymentMethod": payment_method
                })
        
        if st.button("🚀 Predict Batch", use_container_width=True):
            with st.spinner(f"🔮 Predicting churn for {len(customers_data)} customers..."):
                response = requests.post(f"{API_URL}/batch_predict", json=customers_data, timeout=30)
            
            if response.status_code == 200:
                results = response.json()
                predictions_df = pd.DataFrame(results['predictions'])
                summary = results['summary']
                
                st.success(f"✅ Predictions completed!")
                st.dataframe(predictions_df, use_container_width=True)
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk", summary['high_risk_count'])
                with col2:
                    st.metric("Avg Probability", f"{summary['average_churn_probability']:.1%}")
                with col3:
                    st.metric("Predicted Churn", summary['predicted_churn_count'])

# ============================================================================
# ANALYTICS DASHBOARD
# ============================================================================

elif app_mode == "📈 Analytics Dashboard":
    st.header("📈 Analytics Dashboard")
    
    if not st.session_state.predictions_history:
        st.info("📊 No prediction history yet. Make some predictions first!")
    else:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(history_df))
        with col2:
            high_risk = (history_df['risk_level'] == 'High').sum()
            st.metric("High Risk Customers", high_risk)
        with col3:
            avg_prob = history_df['churn_probability'].mean()
            st.metric("Average Churn Probability", f"{avg_prob:.1%}")
        with col4:
            churn_rate = history_df['churn_prediction'].sum() / len(history_df)
            st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
        
        # Visualizations
        fig_col1, fig_col2 = st.columns(2)
        
        with fig_col1:
            # Risk level distribution
            risk_counts = history_df['risk_level'].value_counts()
            fig_bar = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Distribution",
                labels={'x': 'Risk Level', 'y': 'Count'},
                color=risk_counts.index,
                color_discrete_map={
                    'High': '#ff6b6b',
                    'Medium': '#ffd93d',
                    'Low': '#6bcf7f'
                }
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Churn probability over time
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df = history_df.sort_values('timestamp')
                fig_timeline = px.line(
                    history_df,
                    x='timestamp',
                    y='churn_probability',
                    title="Churn Probability Over Time",
                    markers=True
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
        
        with fig_col2:
            # Churn probability distribution
            fig_hist = px.histogram(
                history_df,
                x='churn_probability',
                nbins=20,
                title="Churn Probability Distribution",
                labels={'churn_probability': 'Churn Probability', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Feature analysis
            if 'Contract' in history_df.columns:
                contract_churn = history_df.groupby('Contract')['churn_probability'].mean().sort_values(ascending=False)
                fig_contract = px.bar(
                    x=contract_churn.index,
                    y=contract_churn.values,
                    title="Average Churn Probability by Contract Type",
                    labels={'x': 'Contract Type', 'y': 'Avg Churn Probability'}
                )
                st.plotly_chart(fig_contract, use_container_width=True)

# ============================================================================
# ABOUT MODE
# ============================================================================

elif app_mode == "ℹ️ About":
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts customer churn using machine learning and provides actionable recommendations 
    to help retain customers.
    
    ### ✨ Key Features
    - **Simplified Input**: Only enter the most important customer information
    - **Smart Recommendations**: Get actionable recommendations based on predictions
    - **Batch Processing**: Analyze multiple customers at once
    - **Visual Analytics**: Interactive charts and dashboards
    
    ### 📊 Critical Features Used
    The model focuses on these key factors:
    1. **Tenure** - How long the customer has been with the company
    2. **Monthly Charges** - Monthly billing amount
    3. **Total Charges** - Total amount charged
    4. **Contract Type** - Contract duration (month-to-month has highest risk)
    5. **Internet Service** - Type of internet service
    6. **Payment Method** - How the customer pays
    
    ### 🎚️ Risk Levels
    - 🔴 **High Risk** (>70%): Immediate action required
    - 🟡 **Medium Risk** (30-70%): Monitor closely
    - 🟢 **Low Risk** (<30%): Standard service
    
    ### 💡 Recommendations
    The system provides personalized recommendations such as:
    - Offering long-term contracts to month-to-month customers
    - Improving service quality for high-risk segments
    - Implementing retention programs for new customers
    - Encouraging automatic payment methods
    """)
    
    # Model info
    try:
        response = requests.get(f"{API_URL}/model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            st.subheader("🤖 Model Information")
            st.json(model_info)
    except:
        st.info("Model information not available. Make sure the backend is running.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using FastAPI, Streamlit, and Scikit-learn | 
        Customer Churn Prediction System v2.0 (Improved)
    </div>
    """,
    unsafe_allow_html=True
)

