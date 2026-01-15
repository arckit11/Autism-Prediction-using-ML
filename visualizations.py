"""
Interactive visualization utilities for NeuroSense
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import config


def create_probability_gauge(probability, risk_level):
    """
    Create a gauge chart showing prediction probability
    
    Args:
        probability: Prediction probability (0-1)
        risk_level: Risk level string (Low/Medium/High)
        
    Returns:
        plotly.graph_objects.Figure
    """
    # Determine color based on risk level
    if risk_level == "Low":
        color = config.COLOR_SCHEME["success"]
    elif risk_level == "Medium":
        color = config.COLOR_SCHEME["warning"]
    else:
        color = config.COLOR_SCHEME["danger"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Risk Probability - {risk_level} Risk", 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#E8F5E9'},
                {'range': [30, 60], 'color': '#FFF9C4'},
                {'range': [60, 100], 'color': '#FFEBEE'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="white",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(model, feature_names):
    """
    Create horizontal bar chart of feature importances
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        plotly.graph_objects.Figure
    """
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    
    # Create DataFrame for easier sorting
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df['Importance'],
        y=df['Feature'],
        orientation='h',
        marker=dict(
            color=df['Importance'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        text=df['Importance'].round(3),
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Feature Importance Analysis",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        template=config.PLOTLY_THEME
    )
    
    return fig


def create_behavioral_score_breakdown(input_data):
    """
    Create bar chart showing individual behavioral scores
    
    Args:
        input_data: dict with A1-A10 scores
        
    Returns:
        plotly.graph_objects.Figure
    """
    scores = [input_data[f"A{i}_Score"] for i in range(1, 11)]
    questions = [f"Q{i}" for i in range(1, 11)]
    
    colors = [config.COLOR_SCHEME["danger"] if s == 1 else config.COLOR_SCHEME["success"] 
              for s in scores]
    
    fig = go.Figure(go.Bar(
        x=questions,
        y=scores,
        marker=dict(color=colors),
        text=scores,
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Behavioral Assessment Breakdown (A1-A10 Scores)",
        xaxis_title="Assessment Questions",
        yaxis_title="Score (0 or 1)",
        yaxis=dict(range=[0, 1.2]),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        template=config.PLOTLY_THEME
    )
    
    return fig


def create_risk_distribution_chart(probability):
    """
    Create a pie chart showing risk distribution
    
    Args:
        probability: Prediction probability (0-1)
        
    Returns:
        plotly.graph_objects.Figure
    """
    risk_prob = probability * 100
    no_risk_prob = (1 - probability) * 100
    
    fig = go.Figure(data=[go.Pie(
        labels=['ASD Risk', 'No ASD Risk'],
        values=[risk_prob, no_risk_prob],
        hole=.4,
        marker=dict(colors=[config.COLOR_SCHEME["danger"], config.COLOR_SCHEME["success"]]),
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig.update_layout(
        title="Risk Distribution",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        template=config.PLOTLY_THEME,
        showlegend=True
    )
    
    return fig


def create_demographic_summary(input_data):
    """
    Create visual summary of demographic information
    
    Args:
        input_data: dict with demographic features
        
    Returns:
        plotly.graph_objects.Figure
    """
    demographics = {
        'Gender': input_data['gender'],
        'Age Group': input_data['age_desc'],
        'Jaundice': input_data['jaundice'],
        'Family History': input_data['austim'],
        'Previous Screening': input_data['used_app_before'],
        'Relation': input_data['relation']
    }
    
    # Create a table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Demographic Factor</b>', '<b>Value</b>'],
            fill_color=config.COLOR_SCHEME["primary"],
            align='left',
            font=dict(color='white', size=14)
        ),
        cells=dict(
            values=[list(demographics.keys()), list(demographics.values())],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        )
    )])
    
    fig.update_layout(
        title="Demographic Information",
        height=300,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def create_comparison_radar(input_data, avg_scores=None):
    """
    Create radar chart comparing individual scores to average
    
    Args:
        input_data: dict with behavioral scores
        avg_scores: dict with average scores (optional)
        
    Returns:
        plotly.graph_objects.Figure
    """
    categories = [f"A{i}" for i in range(1, 11)]
    individual_scores = [input_data[f"A{i}_Score"] for i in range(1, 11)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=individual_scores,
        theta=categories,
        fill='toself',
        name='Individual Scores',
        line_color=config.COLOR_SCHEME["primary"]
    ))
    
    if avg_scores:
        avg_values = [avg_scores.get(f"A{i}_Score", 0.5) for i in range(1, 11)]
        fig.add_trace(go.Scatterpolar(
            r=avg_values,
            theta=categories,
            fill='toself',
            name='Average Scores',
            line_color=config.COLOR_SCHEME["secondary"],
            opacity=0.6
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Behavioral Assessment Radar",
        height=450,
        margin=dict(l=20, r=20, t=60, b=20),
        template=config.PLOTLY_THEME
    )
    
    return fig
