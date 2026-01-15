"""
Utility functions for NeuroSense application
"""
import pandas as pd
import config


def interpret_prediction(probability, behavioral_score):
    """
    Generate interpretable insights from prediction
    
    Args:
        probability: Prediction probability (0-1)
        behavioral_score: Total behavioral assessment score
        
    Returns:
        dict: Interpretation with risk level, insights, and recommendations
    """
    # Determine risk level
    if probability < config.LOW_RISK_THRESHOLD:
        risk_level = "Low"
        risk_color = "success"
    elif probability < config.MEDIUM_RISK_THRESHOLD:
        risk_level = "Medium"
        risk_color = "warning"
    else:
        risk_level = "High"
        risk_color = "danger"
    
    # Generate insights
    insights = []
    
    if behavioral_score >= 7:
        insights.append("The behavioral assessment indicates multiple areas of concern across social interaction, communication, and behavioral patterns.")
    elif behavioral_score >= 4:
        insights.append("The behavioral assessment shows some areas that may warrant further attention.")
    else:
        insights.append("The behavioral assessment indicates relatively few concerns in the screened areas.")
    
    # Add probability-based insight
    if probability >= 0.8:
        insights.append(f"The model predicts a {probability*100:.1f}% probability of ASD characteristics, which is considered high confidence.")
    elif probability >= 0.5:
        insights.append(f"The model predicts a {probability*100:.1f}% probability of ASD characteristics, suggesting moderate concern.")
    else:
        insights.append(f"The model predicts a {probability*100:.1f}% probability of ASD characteristics, which is relatively low.")
    
    # Generate recommendations
    recommendations = []
    
    if risk_level == "High":
        recommendations.append("🏥 **Strongly recommend** consultation with a qualified healthcare professional or developmental specialist.")
        recommendations.append("📋 Consider comprehensive diagnostic evaluation (ADOS-2, ADI-R).")
        recommendations.append("👨‍👩‍👧 Explore early intervention services and support resources.")
    elif risk_level == "Medium":
        recommendations.append("👨‍⚕️ **Recommend** discussing results with a pediatrician or healthcare provider.")
        recommendations.append("📊 Consider follow-up screening in 6-12 months.")
        recommendations.append("📚 Learn more about developmental milestones and ASD characteristics.")
    else:
        recommendations.append("✅ Results suggest low likelihood, but continue monitoring development.")
        recommendations.append("📅 Regular developmental check-ups are still important.")
        recommendations.append("🔍 Stay informed about developmental milestones.")
    
    # Universal recommendations
    recommendations.append("⚠️ **Important**: This tool is for screening purposes only and does not replace professional diagnosis.")
    
    return {
        "risk_level": risk_level,
        "risk_color": risk_color,
        "probability_percentage": probability * 100,
        "behavioral_score": behavioral_score,
        "max_score": len(config.BEHAVIORAL_FEATURES),
        "insights": insights,
        "recommendations": recommendations
    }


def get_feature_explanation(feature_name, value):
    """
    Get human-readable explanation for a feature value
    
    Args:
        feature_name: Name of the feature
        value: Value of the feature
        
    Returns:
        str: Explanation text
    """
    description = config.FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
    
    if feature_name in config.BEHAVIORAL_FEATURES:
        if value == 1:
            return f"✓ {description}: Characteristic observed"
        else:
            return f"○ {description}: Characteristic not observed"
    else:
        return f"{description}: {value}"


def format_input_summary(input_data):
    """
    Format input data into readable summary
    
    Args:
        input_data: dict with all input features
        
    Returns:
        str: Formatted summary
    """
    summary_lines = ["### Input Summary\n"]
    
    # Behavioral scores
    summary_lines.append("**Behavioral Assessment (A1-A10):**")
    for i in range(1, 11):
        feature = f"A{i}_Score"
        value = input_data[feature]
        summary_lines.append(f"- {get_feature_explanation(feature, value)}")
    
    # Demographics
    summary_lines.append("\n**Demographic Information:**")
    for feature in config.DEMOGRAPHIC_FEATURES:
        value = input_data[feature]
        summary_lines.append(f"- {get_feature_explanation(feature, value)}")
    
    return "\n".join(summary_lines)


def export_results_to_dict(input_data, prediction, probability, interpretation):
    """
    Export all results to a structured dictionary
    
    Args:
        input_data: Input features
        prediction: Model prediction (0 or 1)
        probability: Prediction probability
        interpretation: Interpretation dict from interpret_prediction
        
    Returns:
        dict: Complete results
    """
    return {
        "timestamp": pd.Timestamp.now().isoformat(),
        "input_data": input_data,
        "prediction": int(prediction),
        "probability": float(probability),
        "risk_level": interpretation["risk_level"],
        "behavioral_score": interpretation["behavioral_score"],
        "insights": interpretation["insights"],
        "recommendations": interpretation["recommendations"]
    }


def validate_input_ranges(input_data):
    """
    Validate that input values are within expected ranges
    
    Args:
        input_data: dict with input features
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []
    
    # Check behavioral scores are 0 or 1
    for i in range(1, 11):
        feature = f"A{i}_Score"
        if feature in input_data:
            value = input_data[feature]
            if value not in [0, 1]:
                errors.append(f"{feature} must be 0 or 1, got {value}")
    
    # Check categorical values
    valid_genders = ["m", "f"]
    if input_data.get("gender") not in valid_genders:
        errors.append(f"gender must be one of {valid_genders}")
    
    valid_yes_no = ["yes", "no"]
    for feature in ["jaundice", "austim", "used_app_before"]:
        if input_data.get(feature) not in valid_yes_no:
            errors.append(f"{feature} must be 'yes' or 'no'")
    
    return len(errors) == 0, errors


def get_disclaimer_text():
    """
    Get standard disclaimer text
    
    Returns:
        str: Disclaimer text
    """
    return """
    ### ⚠️ Important Disclaimer
    
    **NeuroSense is a screening tool for educational and research purposes only.**
    
    - This application does **NOT** provide medical diagnosis
    - Results should **NOT** be used as a substitute for professional medical advice
    - Always consult qualified healthcare professionals for proper evaluation
    - Early intervention and professional assessment are crucial for accurate diagnosis
    - This tool is based on machine learning models trained on screening data
    
    **If you have concerns about autism spectrum disorder, please consult:**
    - Pediatrician or family doctor
    - Developmental pediatrician
    - Child psychologist or psychiatrist
    - Licensed clinical psychologist specializing in ASD
    """
