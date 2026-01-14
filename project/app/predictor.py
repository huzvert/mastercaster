import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from feature_engineer import engineer_features_for_date, get_feature_names

class PM25Predictor:
    """Handler for PM2.5 prediction using trained ML models."""
    
    def __init__(self, data_path, models_path):
        """
        Initialize predictor with data and model paths.
        
        Parameters:
        -----------
        data_path : str
            Path to merged_data_extended.csv
        models_path : str
            Path to models directory
        """
        self.data_path = data_path
        self.models_path = models_path
        self.df = None
        self.scaler = None
        self.model = None
        self.model_name = None
        
        # Load data
        self.load_data()
        
        # Load scaler
        self.load_scaler()
    
    def load_data(self):
        """Load the extended dataset."""
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        print(f"✅ Loaded data: {self.df['date'].min()} to {self.df['date'].max()}")
    
    def load_scaler(self):
        """Load the feature scaler."""
        scaler_path = f"{self.models_path}/scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Loaded scaler from {scaler_path}")
    
    def load_model(self, model_name):
        """
        Load a specific trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model ('lasso', 'xgboost', 'lightgbm', 'random_forest', 
            'gradient_boosting', 'ridge', 'svr')
        """
        model_path = f"{self.models_path}/{model_name}_model.pkl"
        self.model = joblib.load(model_path)
        self.model_name = model_name
        print(f"✅ Loaded {model_name} model")
    
    def validate_date(self, prediction_date):
        """
        Validate that the prediction date has sufficient historical data.
        
        Parameters:
        -----------
        prediction_date : str or datetime
            Date for which to predict next day's PM2.5
        
        Returns:
        --------
        tuple : (bool, str)
            (is_valid, error_message)
        """
        prediction_date = pd.to_datetime(prediction_date)
        
        # Check if date exists in dataset
        if prediction_date not in self.df['date'].values:
            return False, f"Date {prediction_date.date()} not found in dataset."
        
        # Check if we have at least 31 days of prior data
        date_idx = self.df[self.df['date'] == prediction_date].index[0]
        if date_idx < 30:
            min_valid_date = self.df['date'].iloc[30].date()
            return False, f"Insufficient historical data. Please select a date after {min_valid_date}."
        
        # Check if we can predict next day (date shouldn't be the last day)
        if prediction_date >= self.df['date'].max():
            max_valid_date = (self.df['date'].max() - timedelta(days=1)).date()
            return False, f"Cannot predict beyond available data. Please select a date before {max_valid_date}."
        
        return True, ""
    
    def predict(self, prediction_date):
        """
        Predict PM2.5 for the next day based on selected date.
        
        Parameters:
        -----------
        prediction_date : str or datetime
            Date for which to make prediction (will predict next day)
        
        Returns:
        --------
        dict : Prediction results
            {
                'prediction_date': date used for prediction,
                'target_date': date being predicted,
                'predicted_pm25': predicted value,
                'actual_pm25': actual value (if available),
                'model_name': name of model used,
                'features': engineered features (for debugging),
                'is_projected': whether target date uses projected data
            }
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        # Validate date
        is_valid, error_msg = self.validate_date(prediction_date)
        if not is_valid:
            raise ValueError(error_msg)
        
        prediction_date = pd.to_datetime(prediction_date)
        target_date = prediction_date + timedelta(days=1)
        
        # Engineer features for prediction date
        print(f"🔧 Engineering features for {prediction_date.date()}...")
        features = engineer_features_for_date(self.df, prediction_date)
        
        # Get feature names in correct order
        feature_names = get_feature_names()
        
        # Ensure all features are present and in correct order
        feature_vector = features[feature_names].values.reshape(1, -1)
        
        # Handle any NaN values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make prediction
        predicted_pm25 = self.model.predict(feature_vector_scaled)[0]
        
        # Get actual value if available
        actual_pm25 = None
        if target_date in self.df['date'].values:
            actual_pm25 = self.df[self.df['date'] == target_date]['pm2_5_mean'].values[0]
        
        # Check if target date is in projected range (May 2 - Dec 31, 2025)
        is_projected = target_date >= pd.to_datetime('2025-05-02')
        
        result = {
            'prediction_date': prediction_date.date(),
            'target_date': target_date.date(),
            'predicted_pm25': float(predicted_pm25),
            'actual_pm25': float(actual_pm25) if actual_pm25 is not None else None,
            'model_name': self.model_name,
            'features': features.to_dict(),
            'is_projected': is_projected
        }
        
        return result
    
    def get_aqi_category(self, pm25_value):
        """
        Get AQI category and color based on PM2.5 value.
        
        Parameters:
        -----------
        pm25_value : float
            PM2.5 concentration in µg/m³
        
        Returns:
        --------
        dict : AQI information
            {
                'category': category name,
                'color': hex color code,
                'health_implications': health message
            }
        """
        if pm25_value <= 12.0:
            return {
                'category': 'Good',
                'color': '#00E400',
                'health_implications': 'Air quality is satisfactory, and air pollution poses little or no risk.'
            }
        elif pm25_value <= 35.4:
            return {
                'category': 'Moderate',
                'color': '#FFFF00',
                'health_implications': 'Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution.'
            }
        elif pm25_value <= 55.4:
            return {
                'category': 'Unhealthy for Sensitive Groups',
                'color': '#FF7E00',
                'health_implications': 'Members of sensitive groups may experience health effects. The general public is less likely to be affected.'
            }
        elif pm25_value <= 150.4:
            return {
                'category': 'Unhealthy',
                'color': '#FF0000',
                'health_implications': 'Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects.'
            }
        elif pm25_value <= 250.4:
            return {
                'category': 'Very Unhealthy',
                'color': '#8F3F97',
                'health_implications': 'Health alert: The risk of health effects is increased for everyone.'
            }
        else:
            return {
                'category': 'Hazardous',
                'color': '#7E0023',
                'health_implications': 'Health warning of emergency conditions: everyone is more likely to be affected.'
            }
    
    def get_available_date_range(self):
        """
        Get the valid date range for predictions.
        
        Returns:
        --------
        tuple : (min_date, max_date)
        """
        min_date = self.df['date'].iloc[30].date()  # Need 30 days prior
        max_date = (self.df['date'].max() - timedelta(days=1)).date()
        return min_date, max_date
