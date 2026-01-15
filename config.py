"""
Configuration settings for NeuroSense application
"""

# Model Configuration
MODEL_PATH = "best_autism_model.pkl"

# Feature Definitions
BEHAVIORAL_FEATURES = [
    "A1_Score", "A2_Score", "A3_Score", "A4_Score", "A5_Score",
    "A6_Score", "A7_Score", "A8_Score", "A9_Score", "A10_Score"
]

DEMOGRAPHIC_FEATURES = [
    "age", "gender", "ethnicity", "jaundice", "austim", 
    "contry_of_res", "used_app_before", "result", "age_desc", "relation"
]

ALL_FEATURES = BEHAVIORAL_FEATURES + DEMOGRAPHIC_FEATURES

# Feature Descriptions for Interpretability
FEATURE_DESCRIPTIONS = {
    "A1_Score": "Social interaction and communication patterns",
    "A2_Score": "Response to social cues and eye contact",
    "A3_Score": "Understanding of social contexts",
    "A4_Score": "Ability to engage in conversation",
    "A5_Score": "Imaginative play and creativity",
    "A6_Score": "Repetitive behaviors or routines",
    "A7_Score": "Sensory sensitivities",
    "A8_Score": "Focus and attention patterns",
    "A9_Score": "Emotional regulation",
    "A10_Score": "Adaptive functioning skills",
    "age": "Age in years",
    "gender": "Biological sex",
    "ethnicity": "Ethnic background",
    "jaundice": "History of jaundice at birth",
    "austim": "Family history of autism",
    "contry_of_res": "Country of residence",
    "used_app_before": "Previous screening experience",
    "result": "Sum of behavioral assessment scores",
    "age_desc": "Age category",
    "relation": "Relationship to individual being assessed"
}

# Visualization Theme
PLOTLY_THEME = "plotly_white"
COLOR_SCHEME = {
    "primary": "#4A90E2",
    "secondary": "#50C878",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "success": "#2ECC71",
    "info": "#3498DB"
}

# Risk Thresholds
LOW_RISK_THRESHOLD = 0.3
MEDIUM_RISK_THRESHOLD = 0.6
HIGH_RISK_THRESHOLD = 0.8

# App Configuration
APP_TITLE = "NeuroSense - AI-Powered Autism Support Tool"
APP_ICON = "🧠"
PAGE_LAYOUT = "wide"
