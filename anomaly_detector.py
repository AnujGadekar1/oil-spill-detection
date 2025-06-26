# src/core/anomaly_detector.py
"""
Anomaly Detection Engine
Advanced machine learning models for detecting unusual vessel behavior patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import scipy.stats as stats

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Advanced anomaly detection system for maritime vessel behavior"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize models
        self.isolation_forest = None
        self.dbscan_spatial = None
        self.dbscan_behavioral = None
        self.random_forest = None
        self.pca = None
        
        # Scalers for different feature types
        self.movement_scaler = StandardScaler()
        self.spatial_scaler = RobustScaler()
        self.temporal_scaler = StandardScaler()
        
        # Model state
        self.is_trained = False
        self.feature_columns = None
        self.training_stats = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for anomaly detection"""
        return {
            'isolation_forest': {
                'contamination': 0.1,
                'n_estimators': 100,
                'max_samples': 'auto',
                'random_state': 42
            },
            'dbscan_spatial': {
                'eps': 0.5,
                'min_samples': 5,
                'metric': 'euclidean'
            },
            'dbscan_behavioral': {
                'eps': 0.8,
                'min_samples': 3,
                'metric': 'euclidean'
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'random_state': 42
            },
            'pca': {
                'n_components': 0.95,  # Retain 95% of variance
                'random_state': 42
            },
            'thresholds': {
                'anomaly_score': 0.7,
                'risk_score': 0.8,
                'ensemble_threshold': 0.6
            }
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and select features for anomaly detection"""
        
        # Define feature groups
        movement_features = [
            'speed', 'course', 'speed_change', 'speed_acceleration', 'speed_variance',
            'course_change', 'course_rate', 'distance_nm', 'reported_vs_calculated_speed'
        ]
        
        behavioral_features = [
            'stop_indicator', 'high_speed_indicator', 'sharp_turn_indicator',
            'speed_mean_5', 'speed_mean_10', 'speed_mean_20',
            'course_std_5', 'course_std_10', 'course_std_20',
            'position_drift_5', 'position_drift_10', 'position_drift_20'
        ]
        
        spatial_features = [
            'latitude', 'longitude', 'vessel_density', 'distance_from_shore', 'in_shipping_lane'
        ]
        
        temporal_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'is_night'
        ]
        
        # Select available features
        available_features = []
        for feature_group in [movement_features, behavioral_features, spatial_features, temporal_features]:
            available_features.extend([f for f in feature_group if f in df.columns])
        
        self.feature_columns = available_features
        
        # Extract features and handle missing values
        feature_df = df[self.feature_columns].copy()
        
        # Fill missing values with appropriate strategies
        for col in feature_df.columns:
            if col in movement_features + behavioral_features:
                feature_df[col] = feature_df[col].fillna(feature_df[col].median())
            elif col in spatial_features:
                feature_df[col] = feature_df[col].fillna(feature_df[col].mean())
            else:  # temporal features
                feature_df[col] = feature_df[col].fillna(0)
        
        logger.info(f"Prepared {len(available_features)} features for anomaly detection")
        
        return feature_df
    
    def detect_statistical_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies using statistical methods"""
        
        statistical_scores = []
        
        for _, vessel_data in df.groupby('vessel_id'):
            if len(vessel_data) < 5:  # Need minimum data points
                vessel_scores = [0.0] * len(vessel_data)
            else:
                # Z-score based anomaly detection
                speed_zscore = np.abs(stats.zscore(vessel_data['speed']))
                course_change_zscore = np.abs(stats.zscore(vessel_data['course_change']))
                
                # Combine scores
                combined_zscore = np.maximum(speed_zscore, course_change_zscore)
                
                # Convert to anomaly scores (0-1)
                vessel_scores = np.clip(combined_zscore / 3.0, 0, 1).tolist()
            
            statistical_scores.extend(vessel_scores)
        
        df['statistical_anomaly_score'] = statistical_scores
        df['statistical_anomaly'] = df['statistical_anomaly_score'] > 0.5
        
        return df
    
    def train_models(self, df: pd.DataFrame, labels: Optional[pd.Series] = None):
        """Train all anomaly detection models"""
        
        logger.info("Training anomaly detection models...")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        # Scale features
        scaled_features =
