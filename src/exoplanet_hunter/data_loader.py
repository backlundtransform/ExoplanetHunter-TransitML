"""
Data loading utilities for transit light curves using lightkurve.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from pathlib import Path
import warnings

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    warnings.warn(
        "lightkurve not available. Some data loading functionality will be limited.",
        ImportWarning
    )


class TransitDataLoader:
    """
    A class for loading and managing transit light curve data using lightkurve.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the data loader.
        
        Parameters:
        -----------
        cache_dir : str, optional
            Directory to cache downloaded data
        """
        self.cache_dir = cache_dir
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    def search_target_pixel_files(self, target: str, mission: str = "TESS") -> List:
        """
        Search for target pixel files for a given target.
        
        Parameters:
        -----------
        target : str
            Target identifier (TIC ID, KIC ID, or name)
        mission : str
            Mission name (TESS, Kepler, K2)
            
        Returns:
        --------
        List of search results
        """
        if not LIGHTKURVE_AVAILABLE:
            print("Warning: lightkurve not available. Cannot search for target pixel files.")
            return []
            
        try:
            search_result = lk.search_targetpixelfile(target, mission=mission)
            return search_result
        except Exception as e:
            print(f"Error searching for target {target}: {e}")
            return []
    
    def download_light_curve(self, target: str, mission: str = "TESS", 
                           quality_bitmask: str = "default") -> Optional:
        """
        Download and return a light curve for the specified target.
        
        Parameters:
        -----------
        target : str
            Target identifier
        mission : str
            Mission name
        quality_bitmask : str
            Quality mask to apply
            
        Returns:
        --------
        lightkurve.LightCurve or None
        """
        if not LIGHTKURVE_AVAILABLE:
            print("Warning: lightkurve not available. Cannot download light curves.")
            return None
            
        try:
            # Search for light curve files
            search_result = lk.search_lightcurve(target, mission=mission)
            
            if len(search_result) == 0:
                print(f"No light curve data found for target {target}")
                return None
            
            # Download the light curve collection
            lc_collection = search_result.download_all(quality_bitmask=quality_bitmask)
            
            if len(lc_collection) == 0:
                print(f"No light curve data could be downloaded for target {target}")
                return None
            
            # Stitch together multiple sectors/quarters
            lc = lc_collection.stitch()
            
            return lc
            
        except Exception as e:
            print(f"Error downloading light curve for target {target}: {e}")
            return None
    
    def load_multiple_targets(self, targets: List[str], mission: str = "TESS") -> dict:
        """
        Load light curves for multiple targets.
        
        Parameters:
        -----------
        targets : List[str]
            List of target identifiers
        mission : str
            Mission name
            
        Returns:
        --------
        dict
            Dictionary mapping target names to light curves
        """
        light_curves = {}
        
        for target in targets:
            print(f"Loading light curve for {target}...")
            lc = self.download_light_curve(target, mission=mission)
            if lc is not None:
                light_curves[target] = lc
            else:
                print(f"Failed to load light curve for {target}")
        
        return light_curves
    
    def extract_features_from_lightcurve(self, lc) -> pd.DataFrame:
        """
        Extract basic features from a light curve for ML analysis.
        
        Parameters:
        -----------
        lc : lightkurve.LightCurve or DataFrame
            Input light curve or DataFrame with time, flux columns
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with time, flux, and flux_err columns
        """
        if isinstance(lc, pd.DataFrame):
            # Already a DataFrame, just validate columns
            if 'time' in lc.columns and 'flux' in lc.columns:
                return lc.copy()
            else:
                raise ValueError("DataFrame must contain 'time' and 'flux' columns")
        
        if not LIGHTKURVE_AVAILABLE:
            raise ValueError("lightkurve not available and input is not a DataFrame")
        
        # Remove NaN values
        lc = lc.remove_nans()
        
        # Create DataFrame with basic features
        df = pd.DataFrame({
            'time': lc.time.value,
            'flux': lc.flux.value,
            'flux_err': lc.flux_err.value if hasattr(lc, 'flux_err') else np.zeros_like(lc.flux.value)
        })
        
        return df
    
    def get_known_exoplanet_targets(self, max_targets: int = 10) -> List[str]:
        """
        Get a list of known exoplanet host stars for testing.
        
        Parameters:
        -----------
        max_targets : int
            Maximum number of targets to return
            
        Returns:
        --------
        List[str]
            List of target identifiers
        """
        # Known exoplanet host stars with TESS data
        known_targets = [
            "TOI-715",  # Super-Earth in habitable zone
            "TOI-849",  # Hot Neptune
            "TOI-3884", # Hot Jupiter
            "TOI-2109", # Ultra-hot Jupiter
            "TOI-674",  # Sub-Neptune
            "TOI-1685", # Hot Jupiter
            "TOI-1842", # Warm Jupiter
            "TOI-2046", # Hot Jupiter
            "TOI-1728", # Hot Jupiter
            "TOI-1518", # Hot Jupiter
        ]
        
        return known_targets[:max_targets]