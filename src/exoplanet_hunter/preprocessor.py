"""
Preprocessing utilities for transit light curve data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import signal
from scipy.stats import median_abs_deviation
from typing import Tuple, Optional, List
import warnings


class TransitPreprocessor:
    """
    Preprocessing pipeline for transit light curve data.
    """
    
    def __init__(self, 
                 detrend_method: str = "biweight",
                 outlier_sigma: float = 5.0,
                 normalization_method: str = "robust"):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        detrend_method : str
            Method for detrending ('biweight', 'savgol', 'median')
        outlier_sigma : float
            Sigma threshold for outlier removal
        normalization_method : str
            Normalization method ('standard', 'robust', 'minmax')
        """
        self.detrend_method = detrend_method
        self.outlier_sigma = outlier_sigma
        self.normalization_method = normalization_method
        self.scaler = None
        self.imputer = None
        
    def remove_outliers(self, time: np.ndarray, flux: np.ndarray, 
                       flux_err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Remove outliers using sigma clipping.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux array
        flux_err : np.ndarray, optional
            Flux error array
            
        Returns:
        --------
        Tuple of cleaned time, flux, and flux_err arrays
        """
        # Calculate robust statistics
        median_flux = np.median(flux)
        mad_flux = median_abs_deviation(flux, scale='normal')
        
        # Identify outliers
        outlier_mask = np.abs(flux - median_flux) < self.outlier_sigma * mad_flux
        
        # Apply mask
        clean_time = time[outlier_mask]
        clean_flux = flux[outlier_mask]
        clean_flux_err = flux_err[outlier_mask] if flux_err is not None else None
        
        removed_count = len(time) - len(clean_time)
        if removed_count > 0:
            print(f"Removed {removed_count} outliers ({removed_count/len(time)*100:.2f}%)")
        
        return clean_time, clean_flux, clean_flux_err
    
    def detrend_flux(self, time: np.ndarray, flux: np.ndarray) -> np.ndarray:
        """
        Detrend the flux using the specified method.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux array
            
        Returns:
        --------
        np.ndarray
            Detrended flux
        """
        if self.detrend_method == "biweight":
            return self._biweight_detrend(time, flux)
        elif self.detrend_method == "savgol":
            return self._savgol_detrend(flux)
        elif self.detrend_method == "median":
            return self._median_detrend(flux)
        else:
            raise ValueError(f"Unknown detrend method: {self.detrend_method}")
    
    def _biweight_detrend(self, time: np.ndarray, flux: np.ndarray, 
                         window_length: float = 0.5) -> np.ndarray:
        """
        Detrend using biweight filter (simplified version).
        
        Parameters:
        -----------
        time : np.ndarray
            Time array
        flux : np.ndarray
            Flux array
        window_length : float
            Window length in days
            
        Returns:
        --------
        np.ndarray
            Detrended flux
        """
        # Simple implementation - use rolling median as approximation
        try:
            from scipy.ndimage import median_filter
            # Estimate window size in data points
            time_span = time.max() - time.min()
            window_size = int(window_length / time_span * len(time))
            window_size = max(3, min(window_size, len(flux) // 4))  # Ensure reasonable window size
            
            trend = median_filter(flux, size=window_size)
            detrended = flux / trend
            return detrended
        except Exception as e:
            warnings.warn(f"Biweight detrending failed, using median: {e}")
            return self._median_detrend(flux)
    
    def _savgol_detrend(self, flux: np.ndarray) -> np.ndarray:
        """
        Detrend using Savitzky-Golay filter.
        """
        try:
            window_length = min(51, len(flux) // 4)
            if window_length % 2 == 0:
                window_length += 1
            window_length = max(5, window_length)
            
            trend = signal.savgol_filter(flux, window_length, 3)
            detrended = flux / trend
            return detrended
        except Exception as e:
            warnings.warn(f"Savgol detrending failed, using median: {e}")
            return self._median_detrend(flux)
    
    def _median_detrend(self, flux: np.ndarray) -> np.ndarray:
        """
        Simple median detrending.
        """
        median_flux = np.median(flux)
        return flux / median_flux
    
    def normalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """
        Normalize flux values.
        
        Parameters:
        -----------
        flux : np.ndarray
            Flux array
            
        Returns:
        --------
        np.ndarray
            Normalized flux
        """
        flux_reshaped = flux.reshape(-1, 1)
        
        if self.normalization_method == "standard":
            if self.scaler is None:
                self.scaler = StandardScaler()
                normalized = self.scaler.fit_transform(flux_reshaped)
            else:
                normalized = self.scaler.transform(flux_reshaped)
        elif self.normalization_method == "robust":
            if self.scaler is None:
                self.scaler = RobustScaler()
                normalized = self.scaler.fit_transform(flux_reshaped)
            else:
                normalized = self.scaler.transform(flux_reshaped)
        elif self.normalization_method == "minmax":
            # Simple min-max normalization
            min_val, max_val = flux.min(), flux.max()
            normalized = (flux_reshaped - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
        
        return normalized.flatten()
    
    def create_time_features(self, time: np.ndarray) -> pd.DataFrame:
        """
        Create time-based features for ML.
        
        Parameters:
        -----------
        time : np.ndarray
            Time array
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time features
        """
        # Convert to relative time from start
        rel_time = time - time.min()
        
        features = pd.DataFrame({
            'time': time,
            'rel_time': rel_time,
            'time_squared': rel_time**2,
            'time_cubed': rel_time**3,
        })
        
        # Add periodic features (assuming orbital periods might be present)
        for period in [1.0, 2.0, 5.0, 10.0]:  # Common periods in days
            features[f'sin_{period}d'] = np.sin(2 * np.pi * rel_time / period)
            features[f'cos_{period}d'] = np.cos(2 * np.pi * rel_time / period)
        
        return features
    
    def process_light_curve(self, df: pd.DataFrame, 
                           create_features: bool = True) -> pd.DataFrame:
        """
        Full preprocessing pipeline for a light curve.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame with 'time', 'flux', and optionally 'flux_err'
        create_features : bool
            Whether to create additional time features
            
        Returns:
        --------
        pd.DataFrame
            Processed DataFrame
        """
        print("Starting light curve preprocessing...")
        
        # Extract arrays
        time = df['time'].values
        flux = df['flux'].values
        flux_err = df['flux_err'].values if 'flux_err' in df.columns else None
        
        # Remove NaN values
        valid_mask = np.isfinite(time) & np.isfinite(flux)
        if flux_err is not None:
            valid_mask &= np.isfinite(flux_err)
            
        time = time[valid_mask]
        flux = flux[valid_mask]
        if flux_err is not None:
            flux_err = flux_err[valid_mask]
        
        print(f"After NaN removal: {len(time)} data points")
        
        # Remove outliers
        time, flux, flux_err = self.remove_outliers(time, flux, flux_err)
        
        # Detrend
        print(f"Detrending using {self.detrend_method} method...")
        detrended_flux = self.detrend_flux(time, flux)
        
        # Normalize
        print(f"Normalizing using {self.normalization_method} method...")
        normalized_flux = self.normalize_flux(detrended_flux)
        
        # Create output DataFrame
        processed_df = pd.DataFrame({
            'time': time,
            'flux': normalized_flux,
            'original_flux': flux,
            'detrended_flux': detrended_flux
        })
        
        if flux_err is not None:
            processed_df['flux_err'] = flux_err
        
        # Add time features if requested
        if create_features:
            time_features = self.create_time_features(time)
            processed_df = pd.concat([processed_df, time_features.drop('time', axis=1)], axis=1)
        
        print(f"Preprocessing complete. Final dataset shape: {processed_df.shape}")
        
        return processed_df