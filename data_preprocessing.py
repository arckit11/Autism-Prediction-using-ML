"""
Data preprocessing utilities for NeuroSense
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import config


class DataPreprocessor:
    """Handles data preprocessing and validation"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def validate_input(self, data):
        """
        Validate input data for required features
        
        Args:
            data: DataFrame or dict containing input features
            
        Returns:
            bool: True if valid, False otherwise
        """
        if isinstance(data, dict):
            data = pd.DataFrame([data])
            
        required_features = config.ALL_FEATURES
        missing_features = set(required_features) - set(data.columns)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        return True
    
    def preprocess_input(self, input_data):
        """
        Preprocess input data for model prediction
        
        Args:
            input_data: dict or DataFrame with raw input values
            
        Returns:
            DataFrame: Preprocessed data ready for prediction
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        # Validate input
        self.validate_input(df)
        
        return df
    
    def calculate_behavioral_score(self, input_data):
        """
        Calculate total behavioral assessment score
        
        Args:
            input_data: dict or DataFrame with behavioral scores
            
        Returns:
            int: Sum of all A1-A10 scores
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        behavioral_cols = config.BEHAVIORAL_FEATURES
        total_score = df[behavioral_cols].sum(axis=1).values[0]
        
        return int(total_score)
    
    def get_feature_summary(self, input_data):
        """
        Generate a summary of input features
        
        Args:
            input_data: dict or DataFrame with input values
            
        Returns:
            dict: Summary statistics and feature breakdown
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        else:
            df = input_data.copy()
            
        behavioral_score = self.calculate_behavioral_score(df)
        
        summary = {
            "behavioral_score": behavioral_score,
            "max_possible_score": len(config.BEHAVIORAL_FEATURES),
            "behavioral_percentage": (behavioral_score / len(config.BEHAVIORAL_FEATURES)) * 100,
            "demographic_info": {
                "gender": df["gender"].values[0],
                "age_group": df["age_desc"].values[0],
                "jaundice_history": df["jaundice"].values[0],
                "family_history": df["austim"].values[0]
            }
        }
        
        return summary


def load_and_prepare_data(filepath):
    """
    Load and prepare training data
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(filepath)
    
    # Separate features and target
    X = df[config.ALL_FEATURES]
    y = df["Class/ASD"] if "Class/ASD" in df.columns else df["target"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def clean_data(df):
    """
    Clean and handle missing values in dataset
    
    Args:
        df: DataFrame to clean
        
    Returns:
        DataFrame: Cleaned data
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df
