import pandas as pd
import numpy as np
import pickle
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def load_model(model_path, scaler_path=None, features_path=None):
    """
    Load a saved model and its associated files
    """
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler if available
    scaler = None
    if scaler_path and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Load feature names if available
    features = None
    if features_path and os.path.exists(features_path):
        with open(features_path, 'r') as f:
            features = json.load(f)
    
    return model, scaler, features

def predict_rainfall(model, data, scaler=None, features=None):
    """
    Make rainfall prediction using the trained model
    """
    # Ensure data has the right features
    if features is not None:
        if 'RR' in features and 'RR' in data.columns:
            print("Menghapus 'RR' dari data input karena ini adalah nilai yang akan diprediksi.")
            features = [f for f in features if f != 'RR']
        
        data = data[features]
    
    # Scale data if scaler is provided
    if scaler is not None:
        data = pd.DataFrame(
            scaler.transform(data),
            columns=data.columns
        )
    
    # Make prediction
    prediction = model.predict(data)
    
    # Get prediction probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(data)
    
    return prediction, probabilities

def convert_categorical_to_numerical(df, cat_columns):
    """
    Convert categorical columns to numerical
    """
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            # Create a mapping dictionary
            unique_values = df[col].unique()
            mapping = {val: i for i, val in enumerate(unique_values)}
            
            # Apply mapping
            df[f"{col}_code"] = df[col].map(mapping)
    
    return df

def handle_wind_direction(df, direction_columns):
    """
    Convert wind direction to sine and cosine components
    """
    for col in direction_columns:
        if col in df.columns:
            # Convert to radians
            radians = df[col] * np.pi / 180
            
            # Create sine and cosine components
            df[f"{col}_sin"] = np.sin(radians)
            df[f"{col}_cos"] = np.cos(radians)
    
    return df

def create_rainfall_categories(value, bins, labels):
    """
    Categorize rainfall value based on bins and labels
    """
    for i, upper_bound in enumerate(bins[1:]):
        if value < upper_bound:
            return labels[i]
    return labels[-1]

def prepare_data_for_prediction(input_data, features, scaler=None):
    """
    Prepare input data for prediction
    """
    # Create DataFrame
    df = pd.DataFrame([input_data])
    
    # Remove RR from features if present
    if 'RR' in features and 'RR' in df.columns:
        print("Menghapus 'RR' dari data input karena ini adalah nilai yang akan diprediksi.")
        features = [f for f in features if f != 'RR']
        df = df.drop('RR', axis=1, errors='ignore')
    
    # Handle wind direction if present
    direction_columns = [col for col in df.columns if 'ddd' in col.lower()]
    df = handle_wind_direction(df, direction_columns)
    
    # Select only required features
    df = df[features]
    
    # Scale data if scaler is provided
    if scaler is not None:
        df = pd.DataFrame(
            scaler.transform(df),
            columns=df.columns
        )
    
    return df

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from model if available
    """
    if hasattr(model, 'feature_importances_'):
        return dict(zip(feature_names, model.feature_importances_))
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) == 1:
            return dict(zip(feature_names, model.coef_))
        else:
            return dict(zip(feature_names, np.abs(model.coef_).mean(axis=0)))
    else:
        return None

def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Create a confusion matrix plot
    """
    cm = pd.crosstab(y_true, y_pred, rownames=['Aktual'], colnames=['Prediksi'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks + 0.5, class_names, rotation=45)
        plt.yticks(tick_marks + 0.5, class_names)
    
    return plt.gcf()