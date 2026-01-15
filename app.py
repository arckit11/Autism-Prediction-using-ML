"""
NeuroSense - AI-Powered Autism Support Tool
Built with Streamlit, Python, and Machine Learning
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Import custom modules
import config
from data_preprocessing import DataPreprocessor
from visualizations import (
    create_probability_gauge,
    create_feature_importance_chart,
    create_behavioral_score_breakdown,
    create_risk_distribution_chart,
    create_demographic_summary,
    create_comparison_radar
)
from utils import (
    interpret_prediction,
    format_input_summary,
    export_results_to_dict,
    validate_input_ranges,
    get_disclaimer_text
)

# Page Configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout=config.PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .info-box {
        background: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .success-box {
        background: #E8F5E9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .danger-box {
        background: #FFEBEE;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F44336;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'results' not in st.session_state:
    st.session_state.results = None

# Load Model
@st.cache_resource
def load_model():
    """Load the trained ML model"""
    try:
        model_path = Path(config.MODEL_PATH)
        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            return None
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Initialize preprocessor
preprocessor = DataPreprocessor()

# ============================================================================
# HEADER SECTION
# ============================================================================
st.markdown('<h1 class="main-header">🧠 NeuroSense</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Autism Spectrum Disorder Support Tool</p>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Welcome to NeuroSense!</strong><br>
    This application uses advanced machine learning to provide autism screening support based on 
    behavioral patterns and demographic information. Our AI model translates complex data into 
    interpretable insights through interactive visualizations.
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio(
        "Select Section:",
        ["Assessment", "About", "How It Works", "Disclaimer"],
        index=0
    )
    
    st.divider()
    
    st.header("ℹ️ Quick Info")
    st.info("""
    **Assessment Time:** 5-10 minutes
    
    **Questions:** 10 behavioral + demographics
    
    **Technology:** Machine Learning (Random Forest)
    
    **Accuracy:** Trained on validated screening data
    """)
    
    st.divider()
    
    if st.session_state.prediction_made:
        if st.button("🔄 Start New Assessment"):
            st.session_state.prediction_made = False
            st.session_state.results = None
            st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

if page == "Assessment":
    if not st.session_state.prediction_made:
        st.header("📝 Autism Screening Assessment")
        
        st.markdown("""
        Please answer the following questions based on observed behaviors and characteristics.
        Each question should be answered with **Yes (1)** or **No (0)**.
        """)
        
        # Create form for input
        with st.form("assessment_form"):
            st.subheader("🎯 Behavioral Assessment (A1-A10)")
            st.caption("Answer based on observed patterns and behaviors")
            
            # Behavioral questions in a grid layout
            A_scores = []
            
            # Question descriptions
            questions = [
                "Does the individual make eye contact during conversation?",
                "Does the individual respond when their name is called?",
                "Does the individual engage in pretend or imaginative play?",
                "Does the individual show interest in other children?",
                "Does the individual point to show interest in something?",
                "Does the individual bring objects to show you?",
                "Does the individual respond to social smiles?",
                "Does the individual notice when others are hurt or upset?",
                "Does the individual imitate others' actions?",
                "Does the individual respond to simple requests?"
            ]
            
            cols = st.columns(2)
            for i in range(1, 11):
                with cols[(i-1) % 2]:
                    score = st.selectbox(
                        f"**Q{i}:** {questions[i-1]}",
                        options=[0, 1],
                        format_func=lambda x: "No" if x == 0 else "Yes",
                        key=f"A{i}",
                        index=0
                    )
                    A_scores.append(score)
            
            st.divider()
            
            st.subheader("👤 Demographic Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox(
                    "Gender",
                    options=["m", "f"],
                    format_func=lambda x: "Male" if x == "m" else "Female"
                )
                
                age = st.number_input(
                    "Age (in years)",
                    min_value=1,
                    max_value=100,
                    value=25,
                    step=1
                )
                
                ethnicity = st.selectbox(
                    "Ethnicity",
                    options=["White-European", "Asian", "Middle Eastern", "Black", 
                            "South Asian", "Hispanic", "Latino", "Others", "?"],
                    index=0
                )
                
                jaundice = st.selectbox(
                    "Jaundice at birth?",
                    options=["no", "yes"],
                    format_func=lambda x: x.capitalize()
                )
                
                family_history = st.selectbox(
                    "Family history of ASD?",
                    options=["no", "yes"],
                    format_func=lambda x: x.capitalize()
                )
            
            with col2:
                country = st.text_input(
                    "Country of Residence",
                    value="United States",
                    placeholder="Enter country name"
                )
                
                used_app_before = st.selectbox(
                    "Used screening app before?",
                    options=["no", "yes"],
                    format_func=lambda x: x.capitalize()
                )
                
                age_desc = st.selectbox(
                    "Age Group",
                    options=["18 and more", "4-11 years", "12-17 years"]
                )
                
                relation = st.selectbox(
                    "Relation to individual",
                    options=["Self", "Parent", "Relative", "Health care professional"]
                )
            
            st.divider()
            
            # Submit button
            submitted = st.form_submit_button("🔍 Analyze Assessment", use_container_width=True)
            
            if submitted:
                # Build input data
                input_data = {
                    "A1_Score": A_scores[0],
                    "A2_Score": A_scores[1],
                    "A3_Score": A_scores[2],
                    "A4_Score": A_scores[3],
                    "A5_Score": A_scores[4],
                    "A6_Score": A_scores[5],
                    "A7_Score": A_scores[6],
                    "A8_Score": A_scores[7],
                    "A9_Score": A_scores[8],
                    "A10_Score": A_scores[9],
                    "age": float(age),
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "jaundice": jaundice,
                    "austim": family_history,
                    "contry_of_res": country,
                    "used_app_before": used_app_before,
                    "result": float(sum(A_scores)),  # Sum of behavioral scores as result
                    "age_desc": age_desc,
                    "relation": relation
                }
                
                # Validate input
                is_valid, errors = validate_input_ranges(input_data)
                
                if not is_valid:
                    st.error("Input validation failed:")
                    for error in errors:
                        st.error(f"- {error}")
                else:
                    # Preprocess
                    input_df = preprocessor.preprocess_input(input_data)
                    
                    # Make prediction
                    if model is not None:
                        with st.spinner("🤖 AI is analyzing the assessment..."):
                            # The model pipeline expects:
                            # - Numeric features: A1-A10, age, jaundice, austim, used_app_before, result
                            # - Categorical features: gender, ethnicity, contry_of_res, age_desc, relation
                            
                            # Convert yes/no to 1/0 for numeric pipeline features
                            input_df_encoded = input_df.copy()
                            for col in ['jaundice', 'austim', 'used_app_before']:
                                input_df_encoded[col] = input_df_encoded[col].map({'yes': 1, 'no': 0})
                            
                            # Ensure column order matches model expectations
                            input_df_encoded = input_df_encoded[config.ALL_FEATURES]
                            
                            prediction = model.predict(input_df_encoded)[0]
                            
                            if hasattr(model, "predict_proba"):
                                probability = model.predict_proba(input_df_encoded)[0][1]
                            else:
                                probability = float(prediction)
                            
                            # Calculate behavioral score
                            behavioral_score = preprocessor.calculate_behavioral_score(input_data)
                            
                            # Get interpretation
                            interpretation = interpret_prediction(probability, behavioral_score)
                            
                            # Store results
                            st.session_state.results = {
                                'input_data': input_data,
                                'prediction': prediction,
                                'probability': probability,
                                'interpretation': interpretation,
                                'behavioral_score': behavioral_score
                            }
                            st.session_state.prediction_made = True
                            st.rerun()
                    else:
                        st.error("Model not loaded. Cannot make predictions.")
    
    else:
        # Display results
        results = st.session_state.results
        interpretation = results['interpretation']
        
        st.header("📊 Assessment Results")
        
        # Risk Level Banner
        risk_level = interpretation['risk_level']
        probability = results['probability']
        
        if risk_level == "Low":
            st.markdown(f"""
            <div class="success-box">
                <h2>✅ {risk_level} Risk Detected</h2>
                <p>Probability: <strong>{probability*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Medium":
            st.markdown(f"""
            <div class="warning-box">
                <h2>⚠️ {risk_level} Risk Detected</h2>
                <p>Probability: <strong>{probability*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="danger-box">
                <h2>🔴 {risk_level} Risk Detected</h2>
                <p>Probability: <strong>{probability*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Risk Level",
                risk_level,
                delta=None
            )
        
        with col2:
            st.metric(
                "Behavioral Score",
                f"{results['behavioral_score']}/10",
                delta=None
            )
        
        with col3:
            st.metric(
                "Confidence",
                f"{probability*100:.1f}%",
                delta=None
            )
        
        st.divider()
        
        # Visualizations
        st.header("📈 Interactive Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Risk Analysis",
            "📊 Behavioral Breakdown",
            "🔍 Feature Importance",
            "👤 Demographics"
        ])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Probability Gauge
                gauge_fig = create_probability_gauge(probability, risk_level)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Risk Distribution
                dist_fig = create_risk_distribution_chart(probability)
                st.plotly_chart(dist_fig, use_container_width=True)
        
        with tab2:
            # Behavioral Score Breakdown
            breakdown_fig = create_behavioral_score_breakdown(results['input_data'])
            st.plotly_chart(breakdown_fig, use_container_width=True)
            
            # Radar Chart
            radar_fig = create_comparison_radar(results['input_data'])
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with tab3:
            # Feature Importance
            if hasattr(model, 'feature_importances_'):
                importance_fig = create_feature_importance_chart(
                    model,
                    config.ALL_FEATURES
                )
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
                    
                    st.info("""
                    **Understanding Feature Importance:**
                    This chart shows which factors had the most influence on the prediction.
                    Higher values indicate features that contributed more to the model's decision.
                    """)
            else:
                st.info("Feature importance not available for this model type.")
        
        with tab4:
            # Demographic Summary
            demo_fig = create_demographic_summary(results['input_data'])
            st.plotly_chart(demo_fig, use_container_width=True)
        
        st.divider()
        
        # AI Insights
        st.header("🤖 AI-Generated Insights")
        
        st.markdown("### 💡 Key Findings")
        for insight in interpretation['insights']:
            st.markdown(f"- {insight}")
        
        st.divider()
        
        st.markdown("### 📋 Recommendations")
        for recommendation in interpretation['recommendations']:
            st.markdown(recommendation)
        
        st.divider()
        
        # Export Results
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as JSON
            export_data = export_results_to_dict(
                results['input_data'],
                results['prediction'],
                results['probability'],
                interpretation
            )
            
            st.download_button(
                label="📥 Download Results (JSON)",
                data=pd.DataFrame([export_data]).to_json(indent=2),
                file_name="neurosense_results.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Export as CSV
            export_df = pd.DataFrame([{
                'Risk_Level': risk_level,
                'Probability': f"{probability*100:.1f}%",
                'Behavioral_Score': f"{results['behavioral_score']}/10",
                **results['input_data']
            }])
            
            st.download_button(
                label="📥 Download Results (CSV)",
                data=export_df.to_csv(index=False),
                file_name="neurosense_results.csv",
                mime="text/csv",
                use_container_width=True
            )

elif page == "About":
    st.header("ℹ️ About NeuroSense")
    
    st.markdown("""
    ### What is NeuroSense?
    
    NeuroSense is an **AI-powered autism support tool** designed to assist in the early screening 
    of Autism Spectrum Disorder (ASD) characteristics. Built with cutting-edge machine learning 
    technology, it provides:
    
    - 🤖 **Real-time AI predictions** using trained machine learning models
    - 📊 **Interactive visualizations** to understand assessment results
    - 💡 **Interpretable insights** that translate ML outputs into actionable information
    - 📈 **Data-driven analysis** based on validated screening methodologies
    
    ### Technology Stack
    
    - **Python** - Core programming language
    - **Streamlit** - Interactive web application framework
    - **Scikit-learn** - Machine learning model training
    - **Plotly** - Interactive data visualizations
    - **Pandas & NumPy** - Data processing and analysis
    
    ### Features
    
    ✅ **Comprehensive Assessment** - 10 behavioral questions + demographic information
    
    ✅ **ML-Powered Predictions** - Random Forest classifier trained on screening data
    
    ✅ **Visual Analytics** - Gauges, charts, radar plots, and distribution graphs
    
    ✅ **Interpretable AI** - Clear explanations and recommendations
    
    ✅ **Export Capabilities** - Download results in JSON or CSV format
    
    ### Development
    
    Built as part of an AI-powered autism support initiative, combining:
    - Data preprocessing pipelines
    - Machine learning model training
    - Real-time prediction systems
    - Interactive visualization dashboards
    """)

elif page == "How It Works":
    st.header("⚙️ How NeuroSense Works")
    
    st.markdown("""
    ### The Process
    
    NeuroSense follows a systematic approach to autism screening:
    
    #### 1️⃣ Data Collection
    - **Behavioral Assessment**: 10 standardized questions (A1-A10) based on validated screening tools
    - **Demographic Information**: Age, gender, medical history, and family background
    
    #### 2️⃣ Data Preprocessing
    - Input validation and cleaning
    - Feature engineering and transformation
    - Normalization for model compatibility
    
    #### 3️⃣ Machine Learning Prediction
    - **Model Type**: Random Forest Classifier
    - **Training Data**: Validated autism screening datasets
    - **Output**: Probability score (0-1) indicating ASD likelihood
    
    #### 4️⃣ Interpretation & Insights
    - Risk level classification (Low/Medium/High)
    - Behavioral score analysis
    - Feature importance evaluation
    - Personalized recommendations
    
    #### 5️⃣ Visualization
    - Interactive charts and graphs
    - Probability gauges
    - Feature breakdown analysis
    - Comparative radar plots
    
    ### Understanding the Results
    
    **Risk Levels:**
    - 🟢 **Low Risk** (< 30%): Few ASD characteristics observed
    - 🟡 **Medium Risk** (30-60%): Some characteristics warrant attention
    - 🔴 **High Risk** (> 60%): Multiple characteristics suggest professional evaluation
    
    **Behavioral Score:**
    - Sum of all A1-A10 responses (0-10 scale)
    - Higher scores indicate more observed characteristics
    - Used in conjunction with ML probability for comprehensive assessment
    
    ### Model Performance
    
    The machine learning model has been trained on validated screening data with:
    - Cross-validation for robustness
    - Feature importance analysis
    - Probability calibration
    - Regular updates and improvements
    """)

elif page == "Disclaimer":
    st.header("⚠️ Important Disclaimer")
    st.markdown(get_disclaimer_text())
    
    st.divider()
    
    st.markdown("""
    ### Privacy & Data
    
    - ✅ All assessments are processed locally
    - ✅ No data is stored or transmitted to external servers
    - ✅ Results are only saved if you choose to export them
    - ✅ Your privacy is our priority
    
    ### Limitations
    
    This tool has limitations:
    - Not a replacement for professional diagnosis
    - Based on screening data, not diagnostic criteria
    - May not capture all aspects of ASD
    - Cultural and linguistic factors may affect results
    - Should be used as one part of a comprehensive evaluation
    
    ### When to Seek Professional Help
    
    Consult a healthcare professional if:
    - You receive a Medium or High risk result
    - You have ongoing concerns about development
    - You notice regression in skills or abilities
    - You want a comprehensive diagnostic evaluation
    - You need guidance on interventions and support
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.divider()

st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>NeuroSense</strong> - AI-Powered Autism Support Tool</p>
    <p>Built with ❤️ using Machine Learning | Python | Streamlit | Scikit-learn</p>
    <p style="font-size: 0.9rem;">For educational and research purposes only • Not a medical diagnostic tool</p>
</div>
""", unsafe_allow_html=True)
