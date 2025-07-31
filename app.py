import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import base64

# Page configuration
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #f5f7fa;
        color: #1a365d;  /* Default text color for the main app - dark blue */
    }
    
    /* Headers in main content area */
    h1 {
        color: #1a365d;
        font-family: 'Arial Black', sans-serif;
        border-bottom: 3px solid #4f46e5;
        padding-bottom: 10px;
    }
    
    h2 {
        color: #2d3748; /* Darker gray for H2 */
        font-family: 'Arial', sans-serif;
    }
    
    h3 {
        color: #4a5568; /* Medium gray for H3 */
        font-family: 'Arial', sans-serif;
    }
    
    /* Main content general text color */
    .main .block-container {
        color: #2d3748; /* General text color for main content area */
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    
    /* Metric labels (e.g., "Total Predictions") */
    [data-testid="metric-container"] label {
        color: #4a5568 !important; /* Darker grey for metric labels */
    }
    
    /* Metric values (e.g., "10", "89.2%") */
    [data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1a365d !important; /* Dark blue for metric values */
        font-weight: bold !important;
    }

    /* Metric delta values */
    [data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        color: #2d3748 !important; /* Ensure delta values are also readable, e.g., dark gray */
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1d391kg > div { /* Target the main sidebar container and its direct children */
        background-color: #1a365d;
        background-image: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
    }
    
    /* Sidebar text - ensure ALL text is light */
    .css-1d391kg, 
    .css-1d391kg .stMarkdown, 
    .css-1d391kg .stMarkdown *,
    .css-1d391kg label,
    .css-1d391kg p,
    .css-1d391kg .stText,
    .css-1d391kg .stMarkdown h1,
    .css-1d391kg .stMarkdown h2,
    .css-1d391kg .stMarkdown h3,
    .css-1d391kg .stMarkdown h4,
    .css-1d391kg .stMarkdown h5,
    .css-1d391kg .stMarkdown h6 {
        color: #ffffff !important;
    }
    
    /* Widget values in sidebar */
    .css-1d391kg .stSlider .st-ae, 
    .css-1d391kg .stSlider .st-af {
        color: #ffffff !important;
    }
    
    /* Widget input fields */
    .stTextInput input, 
    .stNumberInput input, 
    .stSelectbox .st-bd,
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a365d !important;
        border-radius: 8px !important;
    }
    
    /* Checkbox and radio button labels */
    .css-1d391kg .stCheckbox > label > div:first-child,
    .css-1d391kg .stRadio div[role="radiogroup"] > label > div:first-child {
        color: #ffffff !important;
    }

    /* Selectbox dropdown */
    div[data-baseweb="select"] div[role="option"] {
        color: #1a365d !important;
    }
    div[data-baseweb="select"] div[role="listbox"] {
        background-color: #ffffff !important;
    }

    /* Success/Warning/Error boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #4f46e5;
        background-image: linear-gradient(to right, #4f46e5, #7c3aed);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 28px;
        font-weight: bold;
        transition: all 0.3s;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.3);
    }
    
    .stButton > button:hover {
        background-image: linear-gradient(to right, #4338ca, #6d28d9);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 18px;
        margin: 12px 0;
        border-radius: 8px;
        color: #2d3748;
    }
    
    /* Feature importance box */
    .feature-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        margin: 12px 0;
        border: 1px solid #e2e8f0;
        color: #2d3748;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        transition: all 0.3s;
        color: #2d3748 !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #cbd5e0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    
    /* Gauge chart improvements */
    .gauge .title {
        fill: #4a5568 !important;
        font-weight: bold !important;
    }
    
    /* Data table improvements */
    .stDataFrame {
        border-radius: 8px !important;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1) !important;
    }
    
    /* Footer */
    .footer {
        color: #718096;
        font-size: 0.9rem;
    }
    
    /* Main content text */
    .stMarkdown, .stMarkdown p, .stText, .stText p {
        color: #2d3748 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []

# Load model and feature names
@st.cache_resource
def load_model():
    try:
        # Try to load the model - adjust the filename based on your actual saved model
        model = joblib.load('final_model_random_forest_(overfitting_prevention).pkl')
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'final_model_random_forest_(overfitting_prevention).pkl' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_feature_names():
    try:
        with open('feature_names.txt', 'r') as f:
            features = [line.strip() for line in f.readlines()]
        return features
    except FileNotFoundError:
        st.error("Feature names file (feature_names.txt) not found. Please ensure it's available.")
        # If file not found, return default features based on your analysis
        # IMPORTANT: Replace with actual feature names if you cannot create the file
        return ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates',
                'PageValues', 'SpecialDay', 'OperatingSystems', 'Browser', 'Region',
                'TrafficType', 'Weekend', 'Month_Aug', 'Month_Dec', 'Month_Feb', 'Month_Jul',
                'Month_June', 'Month_Mar', 'Month_May', 'Month_Nov', 'Month_Oct', 'Month_Sep',
                'VisitorType_New_Visitor', 'VisitorType_Other', 'VisitorType_Returning_Visitor']
    except Exception as e:
        st.error(f"Error loading feature names: {e}")
        return None

# Header with icon
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center;'>🛒 E-Commerce Purchase Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #7f8c8d; font-size: 18px;'>Predict customer purchase intent using advanced machine learning</p>", unsafe_allow_html=True)

# Sidebar for inputs
with st.sidebar:
    st.markdown("## 📊 Input Customer Data")
    st.markdown("---")
    
    # Basic Information
    st.markdown("### 👤 Visitor Information") # This one was likely problematic
    visitor_type = st.selectbox("Visitor Type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("Weekend Visit", value=False)
    
    # Temporal Information
    st.markdown("### 📅 Temporal Data") # This one was likely problematic
    month_options = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month = st.selectbox("Month", month_options)
    special_day = st.slider("Special Day", 0.0, 1.0, 0.0, 0.1,
        help="Closeness to a special day (0=No special day, 1=Special day)")
    
    # Page Interaction
    st.markdown("### 📱 Page Interactions") # This one was likely problematic
    administrative = st.number_input("Administrative Pages", 0, 50, 2)
    administrative_duration = st.number_input("Administrative Duration (seconds)", 0.0, 5000.0, 80.0)
    
    informational = st.number_input("Informational Pages", 0, 50, 0)
    informational_duration = st.number_input("Informational Duration (seconds)", 0.0, 5000.0, 0.0)
    
    product_related = st.number_input("Product Related Pages", 0, 500, 30)
    product_related_duration = st.number_input("Product Related Duration (seconds)", 0.0, 50000.0, 1200.0)
    
    # Engagement Metrics
    st.markdown("### 📈 Engagement Metrics") # This one was likely problematic
    page_values = st.slider("Page Values", 0.0, 500.0, 5.0, 0.1,
        help="Average value of pages visited (KEY PREDICTOR!)")
    bounce_rates = st.slider("Bounce Rate", 0.0, 0.3, 0.02, 0.001,
        help="Percentage of single-page visits")
    exit_rates = st.slider("Exit Rate", 0.0, 0.3, 0.04, 0.001,
        help="Percentage of exits from the site")
    
    # Technical Information
    st.markdown("### 💻 Technical Details") # This one was likely problematic
    operating_systems = st.selectbox("Operating System", list(range(1, 9)))
    browser = st.selectbox("Browser", list(range(1, 14)))
    region = st.selectbox("Region", list(range(1, 10)))
    traffic_type = st.selectbox("Traffic Type", list(range(1, 21)))

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Analytics Dashboard", "🎓 Model Insights", "📚 About"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Make a Prediction")
        st.markdown('<div class="info-box">Enter customer session data in the sidebar to predict purchase probability</div>', unsafe_allow_html=True)
        
        # Create prediction dataframe
        if st.button("🔮 Predict Purchase Probability", key="predict"):
            model = load_model()
            
            if model is not None:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'Administrative': [administrative],
                    'Administrative_Duration': [administrative_duration],
                    'Informational': [informational],
                    'Informational_Duration': [informational_duration],
                    'ProductRelated': [product_related],
                    'ProductRelated_Duration': [product_related_duration],
                    'BounceRates': [bounce_rates],
                    'ExitRates': [exit_rates],
                    'PageValues': [page_values],
                    'SpecialDay': [special_day],
                    'Month': [month],
                    'OperatingSystems': [operating_systems],
                    'Browser': [browser],
                    'Region': [region],
                    'TrafficType': [traffic_type],
                    'VisitorType': [visitor_type],
                    'Weekend': [weekend]
                })
                
                # One-hot encode categorical variables
                input_encoded = pd.get_dummies(input_data)
                
                # Align with training features
                feature_names = load_feature_names()
                if feature_names:
                    # Create a dataframe with all features, fill missing with 0
                    aligned_input = pd.DataFrame(0, index=[0], columns=feature_names)
                    for col in input_encoded.columns:
                        if col in aligned_input.columns:
                            aligned_input[col] = input_encoded[col].values
                    
                    # Make prediction
                    prediction = model.predict(aligned_input)[0]
                    probability = model.predict_proba(aligned_input)[0]
                    
                    # Store prediction
                    st.session_state.predictions.append({
                        'timestamp': datetime.now(),
                        'prediction': prediction,
                        'probability': probability[1],
                        'page_values': page_values,
                        'month': month
                    })
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Results")
                    
                    if prediction:
                        st.success("✅ **HIGH PURCHASE PROBABILITY**")
                        st.balloons()
                    else:
                        st.warning("❌ **LOW PURCHASE PROBABILITY**")
                    
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability[1] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Purchase Probability (%)"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "#4f46e5"}, # Match button color
                            'steps': [
                                {'range': [0, 25], 'color': "#cbd5e0"}, # Light grey
                                {'range': [25, 50], 'color': "#a0aec0"}, # Medium grey
                                {'range': [50, 75], 'color': "#90ee90"}, # Light green
                                {'range': [75, 100], 'color': "#3cb371"} # Medium sea green
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("Feature names file not found. Please ensure feature_names.txt is available and correctly formatted.")
            else:
                st.error("Model could not be loaded. Please check the model file.")
    
    with col2:
        st.markdown("### 🎯 Key Factors")
        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        
        # Show importance of current inputs
        # These are hardcoded examples, ideally these would come from SHAP values or model's feature importance for the specific prediction
        importance_data = {
            'Feature': ['Page Values', 'Month (Aug)', 'Bounce Rate', 'Product Pages'],
            'Your Value': [
                f"{page_values:.1f}",
                "Yes" if month == "Aug" else "No",
                f"{bounce_rates:.3f}",
                f"{product_related}"
            ],
            'Impact': ['🔴 Very High', '🟠 High', '🟡 Medium', '🟡 Medium']
        }
        
        for i, row in enumerate(importance_data['Feature']):
            st.markdown(f"**{row}**")
            st.markdown(f"Value: {importance_data['Your Value'][i]}")
            st.markdown(f"Impact: {importance_data['Impact'][i]}")
            if i < len(importance_data['Feature']) - 1:
                st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations
        st.markdown("### 💡 Quick Tips")
        if page_values < 50:
            st.info("💰 Low page values detected. Customer might benefit from personalized high-value product recommendations.")
        if month == "Aug":
            st.success("📅 August visit - Peak conversion month!")
        if bounce_rates > 0.1:
            st.warning("⚠️ High bounce rate. Consider improving landing page engagement.")

with tab2:
    st.markdown("## 📊 Analytics Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(st.session_state.predictions))
    with col2:
        if st.session_state.predictions:
            avg_prob = np.mean([p['probability'] for p in st.session_state.predictions])
            st.metric("Avg Purchase Probability", f"{avg_prob:.1%}")
        else:
             st.metric("Avg Purchase Probability", "N/A")
    with col3:
        if st.session_state.predictions:
            high_prob = sum(1 for p in st.session_state.predictions if p['probability'] > 0.5)
            st.metric("High Probability Sessions", high_prob)
        else:
            st.metric("High Probability Sessions", 0)
    with col4:
        if st.session_state.predictions:
            aug_sessions = sum(1 for p in st.session_state.predictions if p['month'] == 'Aug')
            st.metric("August Sessions", aug_sessions)
        else:
            st.metric("August Sessions", 0)
    
    # Visualizations
    if st.session_state.predictions:
        col1, col2 = st.columns(2)
        
        with col1:
            # Prediction history
            df_predictions = pd.DataFrame(st.session_state.predictions)
            fig = px.line(df_predictions, x='timestamp', y='probability',
                          title='Purchase Probability Over Time',
                          labels={'probability': 'Purchase Probability', 'timestamp': 'Time'})
            fig.update_yaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribution of predictions
            fig = px.histogram(df_predictions, x='probability', nbins=20,
                               title='Distribution of Purchase Probabilities',
                               labels={'probability': 'Purchase Probability', 'count': 'Frequency'})
            fig.update_xaxes(tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)
        
        # Page Values vs Probability scatter
        fig = px.scatter(df_predictions, x='page_values', y='probability',
                         title='Page Values vs Purchase Probability',
                         labels={'page_values': 'Page Values', 'probability': 'Purchase Probability'},
                         trendline="ols")
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet. Make some predictions to see analytics!")

with tab3:
    st.markdown("## 🎓 Model Insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Feature importance visualization
        feature_importance = {
            'Feature': ['PageValues', 'Month_Aug', 'BounceRates', 'ProductRelated', 
                        'Month_Oct', 'Administrative_Duration', 'OperatingSystems', 
                        'Weekend', 'Month_Dec', 'Administrative'],
            'Importance': [0.5773, 0.1204, 0.0663, 0.0482, 0.0467, 
                            0.0257, 0.0183, 0.0171, 0.0139, 0.0127]
        }
        
        df_importance = pd.DataFrame(feature_importance)
        
        fig = px.bar(df_importance, x='Importance', y='Feature', orientation='h',
                     title='Top 10 Feature Importances',
                     color='Importance',
                     color_continuous_scale='viridis')
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔑 Key Findings")
        st.markdown("""
        <div class="feature-box">
        <h4>Top Predictors:</h4>
        
        1. **Page Values (57.7%)** - The #1 predictor by far
        2. **August Month (12.0%)** - Strong seasonal effect
        3. **Bounce Rate (6.6%)** - User engagement matters
        
        <h4>Insights:</h4>
        
        • Page Values alone explains more than half of purchase decisions
        • Strong seasonality with August as peak month
        • Session duration less important than page quality
        • Technical details (OS, Browser) have minimal impact
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance metrics
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "89.2%", "+2.3%")
    with col2:
        st.metric("Precision", "87.5%", "+1.8%")
    with col3:
        st.metric("Recall", "85.3%", "+3.1%")
    with col4:
        st.metric("F1-Score", "86.4%", "+2.5%")

with tab4:
    st.markdown("## 📚 About This Application")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Purpose
        This application predicts whether an e-commerce website visitor will make a purchase based on their Browse behavior and session characteristics.
        
        ### 🔧 Technology Stack
        - **Model**: Random Forest Classifier
        - **Accuracy**: ~89%
        - **Key Innovation**: Identifies high-value visitors in real-time
        
        ### 📊 Data Sources
        The model was trained on real e-commerce session data including:
        - Page interaction metrics
        - Session timing information
        - Technical specifications
        - User engagement indicators
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Use Cases
        1. **Real-time Personalization** - Customize user experience for high-intent visitors
        2. **Marketing Optimization** - Focus resources on high-probability segments
        2. **Inventory Management** - Prepare for seasonal peaks (especially August)
        4. **Conversion Rate Optimization** - Identify and fix high bounce rate issues
        
        ### 🚀 Future Enhancements
        - A/B testing integration
        - Real-time model updates
        - Customer segmentation
        - Revenue prediction
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
    <p>Built with ❤️ using Streamlit and Scikit-learn</p>
    <p>© 2024 E-Commerce Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #95a5a6;'>
    <small>For support or questions, contact: analytics@ecommerce.com | Version 1.0</small>
</div>
""", unsafe_allow_html=True)