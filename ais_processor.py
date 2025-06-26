# src/core/ais_processor.py
"""
AIS Data Processor
Handles Automatic Identification System data processing and feature extraction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class VesselInfo:
    """Vessel information structure"""
    vessel_id: str
    mmsi: str
    vessel_type: str
    length: float
    width: float
    flag_state: str

@dataclass
class AISRecord:
    """Single AIS record structure"""
    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed: float
    course: float
    heading: Optional[float] = None
    status: Optional[str] = None

class AISDataProcessor:
    """Processes and analyzes AIS (Automatic Identification System) data"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.vessel_registry = {}
        
    def _default_config(self) -> Dict:
        """Default configuration for AIS processing"""
        return {
            'min_speed_threshold': 0.5,  # knots
            'max_speed_threshold': 30.0,  # knots
            'position_accuracy_threshold': 0.001,  # degrees
            'time_window_hours': 24,
            'feature_window_size': 10
        }
    
    def register_vessel(self, vessel_info: VesselInfo):
        """Register vessel information"""
        self.vessel_registry[vessel_info.vessel_id] = vessel_info
        logger.info(f"Registered vessel {vessel_info.vessel_id}")
    
    def validate_ais_record(self, record: AISRecord) -> bool:
        """Validate AIS record for data quality"""
        # Basic validation checks
        if not (-90 <= record.latitude <= 90):
            return False
        if not (-180 <= record.longitude <= 180):
            return False
        if record.speed < 0 or record.speed > self.config['max_speed_threshold']:
            return False
        if not (0 <= record.course <= 360):
            return False
        
        return True
    
    def clean_ais_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess AIS data"""
        logger.info(f"Cleaning AIS data: {len(df)} records")
        
        original_count = len(df)
        
        # Remove invalid coordinates
        df = df[(df['latitude'].between(-90, 90)) & 
                (df['longitude'].between(-180, 180))]
        
        # Remove invalid speeds
        df = df[df['speed'].between(0, self.config['max_speed_threshold'])]
        
        # Remove invalid courses
        df = df[df['course'].between(0, 360)]
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['vessel_id', 'timestamp'])
        
        # Sort by vessel and timestamp
        df = df.sort_values(['vessel_id', 'timestamp'])
        
        cleaned_count = len(df)
        logger.info(f"Cleaned data: {cleaned_count}/{original_count} records retained")
        
        return df
    
    def calculate_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate movement-based features for each vessel"""
        
        def process_vessel_group(group):
            """Process features for a single vessel"""
            group = group.sort_values('timestamp')
            
            # Time-based features
            group['time_diff'] = group['timestamp'].diff().dt.total_seconds() / 3600  # hours
            
            # Speed features
            group['speed_change'] = group['speed'].diff().abs()
            group['speed_acceleration'] = group['speed_change'] / group['time_diff']
            group['speed_variance'] = group['speed'].rolling(
                window=self.config['feature_window_size'], min_periods=2
            ).var()
            
            # Course features
            group['course_change'] = group['course'].diff()
            # Handle course wraparound (359° to 1° = 2° change, not 358°)
            group['course_change'] = group['course_change'].apply(
                lambda x: min(abs(x), 360 - abs(x)) if not pd.isna(x) else 0
            )
            group['course_rate'] = group['course_change'] / group['time_diff']
            
            # Position features
            group['lat_change'] = group['latitude'].diff()
            group['lon_change'] = group['longitude'].diff()
            
            # Distance calculation (Haversine approximation)
            group['distance_nm'] = self._calculate_distance(
                group['latitude'].shift(1), group['longitude'].shift(1),
                group['latitude'], group['longitude']
            )
            
            # Speed consistency
            group['reported_vs_calculated_speed'] = abs(
                group['speed'] - (group['distance_nm'] / group['time_diff'])
            )
            
            # Behavioral patterns
            group['stop_indicator'] = (group['speed'] < self.config['min_speed_threshold']).astype(int)
            group['high_speed_indicator'] = (group['speed'] > 20).astype(int)
            group['sharp_turn_indicator'] = (group['course_change'] > 45).astype(int)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                group[f'speed_mean_{window}'] = group['speed'].rolling(window, min_periods=1).mean()
                group[f'course_std_{window}'] = group['course'].rolling(window, min_periods=1).std()
                group[f'position_drift_{window}'] = (
                    group['lat_change'].rolling(window, min_periods=1).std() +
                    group['lon_change'].rolling(window, min_periods=1).std()
                )
            
            return group
        
        # Apply feature calculation to each vessel
        result = df.groupby('vessel_id').apply(process_vessel_group)
        result = result.reset_index(drop=True)
        
        # Fill NaN values with appropriate defaults
        numeric_columns = result.select_dtypes(include=[np.number]).columns
        result[numeric_columns] = result[numeric_columns].fillna(0)
        
        logger.info(f"Calculated movement features for {result['vessel_id'].nunique()} vessels")
        
        return result
    
    def calculate_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate spatial and geographic features"""
        
        # Proximity to other vessels
        def calculate_vessel_density(group):
            """Calculate local vessel density"""
            lat, lon = group['latitude'].iloc[-1], group['longitude'].iloc[-1]
            
            # Find nearby vessels (within 5 nautical miles)
            nearby_vessels = 0
            for other_vessel in df['vessel_id'].unique():
                if other_vessel == group['vessel_id'].iloc[0]:
                    continue
                    
                other_data = df[df['vessel_id'] == other_vessel]
                if len(other_data) > 0:
                    other_lat, other_lon = other_data.iloc[-1][['latitude', 'longitude']]
                    distance = self._calculate_distance(lat, lon, other_lat, other_lon)
                    if distance < 5:  # nautical miles
                        nearby_vessels += 1
            
            return nearby_vessels
        
        # Add spatial features
        vessel_density = df.groupby('vessel_id').apply(calculate_vessel_density)
        df['vessel_density'] = df['vessel_id'].map(vessel_density)
        
        # Distance from shore (simplified - would use actual coastline data)
        df['distance_from_shore'] = self._estimate_shore_distance(
            df['latitude'], df['longitude']
        )
        
        # Shipping lane proximity (simplified)
        df['in_shipping_lane'] = self._check_shipping_lane_proximity(
            df['latitude'], df['longitude']
        )
        
        return df
    
    def calculate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based behavioral features"""
        
        # Time of day features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] < 6) | (df['hour'] > 18)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def generate_sample_data(self, num_vessels: int = 100, days: int = 7) -> pd.DataFrame:
        """Generate realistic sample AIS data for testing"""
        np.random.seed(42)
        
        # Define maritime regions
        regions = {
            'North Sea': {'lat': (54, 58), 'lon': (2, 6)},
            'English Channel': {'lat': (49, 51), 'lon': (-2, 2)},
            'Baltic Sea': {'lat': (54, 60), 'lon': (10, 20)}
        }
        
        vessel_types = ['Cargo', 'Tanker', 'Container', 'Bulk Carrier', 'Oil Tanker', 'Fishing']
        
        data = []
        
        for vessel_id in range(1, num_vessels + 1):
            vessel_type = np.random.choice(vessel_types)
            region = np.random.choice(list(regions.keys()))
            region_coords = regions[region]
            
            # Base position in selected region
            base_lat = np.random.uniform(*region_coords['lat'])
            base_lon = np.random.uniform(*region_coords['lon'])
            
            # Vessel characteristics based on type
            if vessel_type in ['Oil Tanker', 'Tanker']:
                base_speed = np.random.uniform(8, 15)
                speed_variance = 2
            elif vessel_type == 'Container':
                base_speed = np.random.uniform(15, 22)
                speed_variance = 3
            else:
                base_speed = np.random.uniform(10, 18)
                speed_variance = 2.5
            
            # Generate trajectory
            current_lat, current_lon = base_lat, base_lon
            current_course = np.random.uniform(0, 360)
            
            for hour in range(days * 24):
                timestamp = datetime.now() - timedelta(hours=days*24 - hour)
                
                # Simulate different behaviors
                behavior_type = np.random.choice(
                    ['normal', 'anchored', 'maneuvering', 'anomalous'], 
                    p=[0.80, 0.10, 0.07, 0.03]
                )
                
                if behavior_type == 'normal':
                    # Normal cruising
                    speed = np.random.normal(base_speed, speed_variance)
                    speed = max(0, min(speed, 25))
                    course_change = np.random.normal(0, 5)
                    
                elif behavior_type == 'anchored':
                    # Anchored or very slow
                    speed = np.random.uniform(0, 1)
                    course_change = np.random.uniform(-10, 10)
                    
                elif behavior_type == 'maneuvering':
                    # Port operations or maneuvering
                    speed = np.random.uniform(2, 8)
                    course_change = np.random.uniform(-30, 30)
                    
                else:  # anomalous
                    # Unusual behavior (potential spill scenario)
                    if np.random.random() > 0.5:
                        # Sudden stop
                        speed = np.random.uniform(0, 2)
                        course_change = np.random.uniform(-45, 45)
                    else:
                        # Erratic movement
                        speed = np.random.uniform(0, 20)
                        course_change = np.random.uniform(-90, 90)
                
                # Update course
                current_course = (current_course + course_change) % 360
                
                # Update position based on speed and course
                # Simplified movement calculation
                distance_nm = speed * 1  # nautical miles per hour
                lat_change = distance_nm * np.cos(np.radians(current_course)) / 60
                lon_change = distance_nm * np.sin(np.radians(current_course)) / (60 * np.cos(np.radians(current_lat)))
                
                current_lat += lat_change
                current_lon += lon_change
                
                # Add some noise to position
                current_lat += np.random.normal(0, 0.001)
                current_lon += np.random.normal(0, 0.001)
                
                data.append({
                    'vessel_id': f'V{vessel_id:03d}',
                    'mmsi': f'{200000000 + vessel_id}',
                    'timestamp': timestamp,
                    'latitude': current_lat,
                    'longitude': current_lon,
                    'speed': speed,
                    'course': current_course,
                    'vessel_type': vessel_type,
                    'region': region,
                    'behavior_type': behavior_type,
                    'is_anomalous': behavior_type == 'anomalous'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} AIS records for {num_vessels} vessels over {days} days")
        
        return df
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Calculate distance between two points in nautical miles (Haversine formula)"""
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return 0
        
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in nautical miles
        earth_radius_nm = 3440.065
        
        return earth_radius_nm * c
    
    def _estimate_shore_distance(self, latitudes, longitudes) -> pd.Series:
        """Estimate distance from shore (simplified implementation)"""
        # This is a simplified implementation
        # In production, you would use actual coastline data
        
        # Assume center of regions are furthest from shore
        shore_distances = []
        
        for lat, lon in zip(latitudes, longitudes):
            # Very simplified calculation based on known maritime regions
            if 54 <= lat <= 58 and 2 <= lon <= 6:  # North Sea
                distance = min(abs(lat - 56), abs(lon - 4)) * 60  # Rough approximation
            elif 49 <= lat <= 51 and -2 <= lon <= 2:  # English Channel
                distance = min(abs(lat - 50), abs(lon - 0)) * 60
            else:
                distance = 50  # Default distance
            
            shore_distances.append(distance)
        
        return pd.Series(shore_distances)
    
    def _check_shipping_lane_proximity(self, latitudes, longitudes) -> pd.Series:
        """Check if vessel is in major shipping lanes"""
        # Simplified shipping lane detection
        in_lane = []
        
        for lat, lon in zip(latitudes, longitudes):
            # Major shipping lanes (simplified)
            is_in_lane = (
                (49 <= lat <= 51 and -1 <= lon <= 1) or  # Dover Strait
                (55 <= lat <= 57 and 3 <= lon <= 5) or   # North Sea main route
                (54 <= lat <= 56 and 10 <= lon <= 12)    # Baltic approach
            )
            in_lane.append(1 if is_in_lane else 0)
        
        return pd.Series(in_lane)
    
    def process_real_time_stream(self, ais_record: AISRecord) -> Dict:
        """Process a single real-time AIS record"""
        # Validate the record
        if not self.validate_ais_record(ais_record):
            logger.warning(f"Invalid AIS record for vessel {ais_record.vessel_id}")
            return None
        
        # Convert to DataFrame for processing
        record_df = pd.DataFrame([{
            'vessel_id': ais_record.vessel_id,
            'timestamp': ais_record.timestamp,
            'latitude': ais_record.latitude,
            'longitude': ais_record.longitude,
            'speed': ais_record.speed,
            'course': ais_record.course
        }])
        
        # Process features (simplified for real-time)
        processed_record = {
            'vessel_id': ais_record.vessel_id,
            'timestamp': ais_record.timestamp,
            'position': {'lat': ais_record.latitude, 'lon': ais_record.longitude},
            'speed': ais_record.speed,
            'course': ais_record.course,
            'processed_at': datetime.now()
        }
        
        return processed_record
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            'registered_vessels': len(self.vessel_registry),
            'config': self.config,
            'last_updated': datetime.now()
        }
