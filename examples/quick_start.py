#!/usr/bin/env python3
"""
Quick start example for ExoplanetHunter-TransitML.

This script demonstrates basic usage of the package for exoplanet detection.
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from exoplanet_hunter import (
    TransitDataLoader,
    TransitPreprocessor, 
    TransitClassifier,
    GroupedTimeSeriesValidator
)
import pandas as pd
import numpy as np


def main():
    """Main function demonstrating basic usage."""
    print("ExoplanetHunter-TransitML Quick Start Example")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    data_loader = TransitDataLoader()
    preprocessor = TransitPreprocessor(
        detrend_method="biweight",
        outlier_sigma=5.0,
        normalization_method="robust"
    )
    classifier = TransitClassifier(model_type="random_forest")
    validator = GroupedTimeSeriesValidator(n_splits=3)
    
    # 2. Load sample data
    print("\n2. Loading sample exoplanet data...")
    targets = data_loader.get_known_exoplanet_targets(max_targets=2)
    print(f"Selected targets: {targets}")
    
    try:
        light_curves = data_loader.load_multiple_targets(targets, mission="TESS")
        print(f"Successfully loaded {len(light_curves)} light curves")
        
        if len(light_curves) == 0:
            print("No real data available. Creating synthetic data for demonstration...")
            light_curves = create_synthetic_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic data for demonstration...")
        light_curves = create_synthetic_data()
    
    # 3. Preprocess data
    print("\n3. Preprocessing light curves...")
    processed_data = {}
    
    for target, lc in light_curves.items():
        if hasattr(lc, 'time'):  # Real lightkurve data
            df = data_loader.extract_features_from_lightcurve(lc)
        else:  # Synthetic data
            df = lc
            
        processed_df = preprocessor.process_light_curve(df, create_features=True)
        processed_df['target'] = target
        processed_data[target] = processed_df
    
    # 4. Combine data and create labels
    print("\n4. Creating training dataset...")
    combined_df = pd.concat(processed_data.values(), ignore_index=True)
    
    # Create synthetic transit labels for demonstration
    all_labels = []
    for target in combined_df['target'].unique():
        target_data = combined_df[combined_df['target'] == target]
        labels = classifier.create_transit_labels(
            target_data['time'].values,
            target_data['flux'].values,
            transit_threshold=0.01,
            window_size=5
        )
        all_labels.extend(labels)
    
    combined_df['has_transit'] = all_labels
    print(f"Dataset shape: {combined_df.shape}")
    print(f"Transit rate: {combined_df['has_transit'].mean()*100:.2f}%")
    
    # 5. Train model
    print("\n5. Training machine learning model...")
    training_results = classifier.train(combined_df, target_column='has_transit')
    print(f"Training accuracy: {training_results['training_accuracy']:.3f}")
    
    # 6. Cross-validation
    print("\n6. Performing cross-validation...")
    feature_cols = [col for col in combined_df.columns 
                   if col not in ['time', 'target', 'has_transit', 'original_flux']]
    
    X = combined_df[feature_cols].values
    y = combined_df['has_transit'].values
    groups = combined_df['target'].values
    
    cv_results = validator.validate_model(
        classifier.model, X, y, groups=groups,
        scoring_metrics=['accuracy', 'precision', 'recall', 'f1']
    )
    
    print("\nCross-validation results:")
    summary = cv_results['summary']
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        if mean_key in summary:
            print(f"  {metric.capitalize()}: {summary[mean_key]:.3f} ± {summary[std_key]:.3f}")
    
    # 7. Make predictions
    print("\n7. Making predictions on sample data...")
    predictions, probabilities = classifier.predict(combined_df.head(100))
    transit_detections = predictions.sum()
    print(f"Detected {transit_detections} transits in first 100 data points")
    print(f"Average transit probability: {probabilities.mean():.3f}")
    
    print("\n" + "=" * 50)
    print("Quick start example completed successfully!")
    print("For more detailed analysis, see the Jupyter notebook in notebooks/")
    

def create_synthetic_data():
    """Create synthetic light curve data for demonstration."""
    print("Creating synthetic light curve data...")
    
    synthetic_data = {}
    
    for i, target in enumerate(["Synthetic-1", "Synthetic-2"]):
        # Create time series
        time = np.linspace(0, 30, 1000)  # 30 days, 1000 points
        
        # Base flux with some noise and trends
        flux = np.ones_like(time) + 0.001 * np.random.randn(len(time))
        flux += 0.002 * np.sin(2 * np.pi * time / 10)  # Long-term variation
        
        # Add some transit-like dips
        transit_times = [5 + i*7 for i in range(4)]  # Every 7 days
        for t_transit in transit_times:
            mask = np.abs(time - t_transit) < 0.1  # Transit duration
            flux[mask] *= 0.99  # 1% dip
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time,
            'flux': flux,
            'flux_err': 0.001 * np.ones_like(time)
        })
        
        synthetic_data[target] = df
    
    return synthetic_data


if __name__ == "__main__":
    main()