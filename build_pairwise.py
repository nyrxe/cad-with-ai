#!/usr/bin/env python3
"""
Build Pairwise Thinning Dataset
Creates pairwise comparisons between original and eroded models with mass/volume data
Enhanced with tolerance system, error handling, and robust data cleaning
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Tolerance constants for handling floating point precision
EPSILON_ELEMENT = 1e-6
EPSILON_VOLUME = 1e-12
EPSILON_MASS = 1e-9
EPSILON_STRESS = 1e-3
EPSILON_PERCENT = 1e-3  # 0.001% tolerance for percentage calculations

def load_dataset(csv_path="dataset_with_mass_volume.csv"):
    """Load the dataset with mass and volume data"""
    print("=== LOADING DATASET WITH MASS/VOLUME ===")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Model types: {df['Model_Type'].value_counts().to_dict()}")
    print(f"Unique models: {df['Model'].nunique()}")
    print(f"Unique parts: {df['Part'].nunique()}")
    return df

def safe_divide(numerator, denominator, epsilon=1e-12):
    """Safe division that prevents division by zero and handles small values"""
    if abs(denominator) < epsilon:
        return 0.0
    return numerator / denominator

def build_pairwise_dataset(df):
    """Build pairwise comparison dataset with robust error handling"""
    print("\n=== BUILDING PAIRWISE DATASET ===")
    
    pairwise_data = []
    skipped_pairs = 0
    skipped_reasons = {
        'wrong_count': 0,
        'missing_data': 0,
        'invalid_reduction': 0,
        'infinite_values': 0,
        'nan_values': 0
    }
    
    # Group by Model and Part
    for (model, part), group in df.groupby(['Model', 'Part']):
        if len(group) != 2:
            print(f"Skipping {model}-{part}: {len(group)} records (need exactly 2)")
            skipped_pairs += 1
            skipped_reasons['wrong_count'] += 1
            continue
        
        # Find original and eroded rows
        orig_row = group[group['Model_Type'] == 'original']
        ero_row = group[group['Model_Type'] == 'eroded']
        
        if len(orig_row) == 0 or len(ero_row) == 0:
            print(f"Skipping {model}-{part}: missing original or eroded")
            skipped_pairs += 1
            skipped_reasons['missing_data'] += 1
            continue
        
        orig = orig_row.iloc[0]
        ero = ero_row.iloc[0]
        
        # Check for required fields with NaN/infinity checks
        required_fields = ['ElementCount', 'vonMises_p95_Pa', 'vonMises_max_Pa', 
                          'vonMises_mean_Pa', 'vonMises_std_Pa', 'volume_m3', 'mass_kg']
        
        missing_fields = []
        infinite_fields = []
        
        for field in required_fields:
            if pd.isna(orig[field]) or pd.isna(ero[field]):
                missing_fields.append(field)
            elif np.isinf(orig[field]) or np.isinf(ero[field]):
                infinite_fields.append(field)
        
        if missing_fields:
            print(f"Skipping {model}-{part}: missing fields {missing_fields}")
            skipped_pairs += 1
            skipped_reasons['missing_data'] += 1
            continue
            
        if infinite_fields:
            print(f"Skipping {model}-{part}: infinite values in {infinite_fields}")
            skipped_pairs += 1
            skipped_reasons['infinite_values'] += 1
            continue
        
        # Calculate metrics with safe division
        orig_el = float(orig['ElementCount'])
        ero_el = float(ero['ElementCount'])
        
        # Element reduction with tolerance
        elem_reduction_pct = safe_divide(orig_el - ero_el, orig_el, EPSILON_ELEMENT) * 100
        
        # Check for meaningful reduction (allow tiny differences)
        if elem_reduction_pct < EPSILON_PERCENT:
            print(f"Skipping {model}-{part}: negligible reduction {elem_reduction_pct:.6f}%")
            skipped_pairs += 1
            skipped_reasons['invalid_reduction'] += 1
            continue
        
        # Stress calculations with safe division
        orig_p95 = float(orig['vonMises_p95_Pa'])
        ero_p95 = float(ero['vonMises_p95_Pa'])
        delta_p95_pct = safe_divide(ero_p95 - orig_p95, orig_p95, EPSILON_STRESS) * 100
        
        orig_max = float(orig['vonMises_max_Pa'])
        ero_max = float(ero['vonMises_max_Pa'])
        delta_max_pct = safe_divide(ero_max - orig_max, orig_max, EPSILON_STRESS) * 100
        
        orig_mean = float(orig['vonMises_mean_Pa'])
        ero_mean = float(ero['vonMises_mean_Pa'])
        delta_mean_pct = safe_divide(ero_mean - orig_mean, orig_mean, EPSILON_STRESS) * 100
        
        # Check for NaN or infinite results
        calculated_values = [elem_reduction_pct, delta_p95_pct, delta_max_pct, delta_mean_pct]
        if any(np.isnan(val) or np.isinf(val) for val in calculated_values):
            print(f"Skipping {model}-{part}: NaN/infinite calculated values")
            skipped_pairs += 1
            skipped_reasons['nan_values'] += 1
            continue
        
        # Create pairwise record
        pair_record = {
            # Identifiers
            'model_id': model,
            'part_id': part,
            
            # Original data (context)
            'orig_el': orig_el,
            'orig_p95': orig_p95,
            'orig_mean': orig_mean,
            'orig_std': float(orig['vonMises_std_Pa']),
            'orig_volume_m3': float(orig['volume_m3']),
            'orig_mass_kg': float(orig['mass_kg']),
            'pitch_m': float(orig['pitch_m']),
            'density_kg_m3': float(orig['density_kg_m3']),
            
            # Eroded data
            'ero_el': ero_el,
            'ero_volume_m3': float(ero['volume_m3']),
            'ero_mass_kg': float(ero['mass_kg']),
            
            # Calculated metrics
            'elem_reduction_pct': elem_reduction_pct,
            'delta_p95_pct': delta_p95_pct,
            'delta_max_pct': delta_max_pct,
            'delta_mean_pct': delta_mean_pct,
            
            # Anomaly detection
            'anomaly': delta_p95_pct < 0
        }
        
        pairwise_data.append(pair_record)
    
    pairwise_df = pd.DataFrame(pairwise_data)
    print(f"Created {len(pairwise_df)} pairwise comparisons")
    print(f"Skipped {skipped_pairs} pairs")
    print(f"Skipped reasons: {skipped_reasons}")
    
    return pairwise_df

def analyze_pairwise_data(pairwise_df):
    """Analyze the pairwise dataset and print summary statistics"""
    print("\n=== PAIRWISE DATA ANALYSIS ===")
    
    if len(pairwise_df) == 0:
        print("❌ No pairwise data found!")
        return
    
    # Basic counts
    total_pairs = len(pairwise_df)
    anomalies = pairwise_df['anomaly'].sum()
    anomaly_rate = anomalies / total_pairs * 100
    
    print(f"Total pairs: {total_pairs}")
    print(f"Anomalies (delta_p95_pct < 0): {anomalies} ({anomaly_rate:.1f}%)")
    
    # Element reduction statistics
    elem_reduction = pairwise_df['elem_reduction_pct']
    print(f"\nElement Reduction Statistics:")
    print(f"  Mean: {elem_reduction.mean():.1f}%")
    print(f"  Std: {elem_reduction.std():.1f}%")
    print(f"  Min: {elem_reduction.min():.1f}%")
    print(f"  Max: {elem_reduction.max():.1f}%")
    
    # P95 stress change statistics
    delta_p95 = pairwise_df['delta_p95_pct']
    print(f"\nP95 Stress Change Statistics:")
    print(f"  Mean: {delta_p95.mean():.1f}%")
    print(f"  Std: {delta_p95.std():.1f}%")
    print(f"  Min: {delta_p95.min():.1f}%")
    print(f"  Max: {delta_p95.max():.1f}%")
    
    # Max stress change statistics
    delta_max = pairwise_df['delta_max_pct']
    print(f"\nMax Stress Change Statistics:")
    print(f"  Mean: {delta_max.mean():.1f}%")
    print(f"  Std: {delta_max.std():.1f}%")
    print(f"  Min: {delta_max.min():.1f}%")
    print(f"  Max: {delta_max.max():.1f}%")
    
    # Mean stress change statistics
    delta_mean = pairwise_df['delta_mean_pct']
    print(f"\nMean Stress Change Statistics:")
    print(f"  Mean: {delta_mean.mean():.1f}%")
    print(f"  Std: {delta_mean.std():.1f}%")
    print(f"  Min: {delta_mean.min():.1f}%")
    print(f"  Max: {delta_mean.max():.1f}%")
    
    # Mass and volume analysis
    print(f"\nMass/Volume Analysis:")
    mass_reduction = (pairwise_df['orig_mass_kg'] - pairwise_df['ero_mass_kg']) / pairwise_df['orig_mass_kg'] * 100
    volume_reduction = (pairwise_df['orig_volume_m3'] - pairwise_df['ero_volume_m3']) / pairwise_df['orig_volume_m3'] * 100
    
    print(f"  Mass reduction - Mean: {mass_reduction.mean():.1f}%, Std: {mass_reduction.std():.1f}%")
    print(f"  Volume reduction - Mean: {volume_reduction.mean():.1f}%, Std: {volume_reduction.std():.1f}%")
    
    # Model and part analysis
    print(f"\nDataset Composition:")
    print(f"  Unique models: {pairwise_df['model_id'].nunique()}")
    print(f"  Unique parts: {pairwise_df['part_id'].nunique()}")
    
    # Show top anomalies
    if anomalies > 0:
        print(f"\nTop 5 Anomalies (largest negative delta_p95_pct):")
        top_anomalies = pairwise_df[pairwise_df['anomaly']].nlargest(5, 'delta_p95_pct')
        for _, row in top_anomalies.iterrows():
            print(f"  {row['model_id']}-{row['part_id']}: {row['delta_p95_pct']:.1f}%")
    
    return pairwise_df

def clean_data_for_visualization(df, clip_percentiles=(1, 99)):
    """Clean data for visualization by clipping extreme outliers"""
    print(f"\n=== CLEANING DATA FOR VISUALIZATION ===")
    
    # Create a copy for clipping
    df_clipped = df.copy()
    
    # Columns to clip
    numeric_cols = ['elem_reduction_pct', 'delta_p95_pct', 'delta_max_pct', 'delta_mean_pct']
    
    clipping_info = {}
    
    for col in numeric_cols:
        if col in df_clipped.columns:
            # Calculate percentiles
            lower_percentile = np.percentile(df_clipped[col], clip_percentiles[0])
            upper_percentile = np.percentile(df_clipped[col], clip_percentiles[1])
            
            # Count outliers
            outliers_low = (df_clipped[col] < lower_percentile).sum()
            outliers_high = (df_clipped[col] > upper_percentile).sum()
            
            # Clip values
            df_clipped[col] = np.clip(df_clipped[col], lower_percentile, upper_percentile)
            
            clipping_info[col] = {
                'lower_bound': lower_percentile,
                'upper_bound': upper_percentile,
                'outliers_low': outliers_low,
                'outliers_high': outliers_high
            }
            
            print(f"  {col}: clipped {outliers_low + outliers_high} outliers")
            print(f"    Range: [{lower_percentile:.2f}, {upper_percentile:.2f}]")
    
    return df_clipped, clipping_info

def create_visualizations(pairwise_df):
    """Create visualizations of the pairwise data with outlier handling"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Clean data for visualization
    df_clipped, clipping_info = clean_data_for_visualization(pairwise_df)
    
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pairwise Thinning Analysis with Mass/Volume Data (Clipped for Visualization)', fontsize=16)
        
        # 1. Element reduction vs P95 stress change
        scatter = axes[0, 0].scatter(df_clipped['elem_reduction_pct'], df_clipped['delta_p95_pct'], 
                                    alpha=0.6, c=df_clipped['anomaly'], cmap='RdYlBu', s=20)
        axes[0, 0].set_xlabel('Element Reduction (%)')
        axes[0, 0].set_ylabel('P95 Stress Change (%)')
        axes[0, 0].set_title('Element Reduction vs P95 Stress Change')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        try:
            z = np.polyfit(df_clipped['elem_reduction_pct'], df_clipped['delta_p95_pct'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(df_clipped['elem_reduction_pct'].min(), 
                                df_clipped['elem_reduction_pct'].max(), 100)
            axes[0, 0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except:
            print("  Warning: Could not fit trend line")
        
        # 2. Mass reduction vs Stress change
        mass_reduction = safe_divide(df_clipped['orig_mass_kg'] - df_clipped['ero_mass_kg'], 
                                   df_clipped['orig_mass_kg'], EPSILON_MASS) * 100
        axes[0, 1].scatter(mass_reduction, df_clipped['delta_p95_pct'], alpha=0.6, s=20)
        axes[0, 1].set_xlabel('Mass Reduction (%)')
        axes[0, 1].set_ylabel('P95 Stress Change (%)')
        axes[0, 1].set_title('Mass Reduction vs P95 Stress Change')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution of element reductions
        axes[0, 2].hist(df_clipped['elem_reduction_pct'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('Element Reduction (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Distribution of Element Reductions')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Distribution of P95 stress changes
        axes[1, 0].hist(df_clipped['delta_p95_pct'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('P95 Stress Change (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of P95 Stress Changes')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Anomaly analysis
        anomaly_counts = df_clipped['anomaly'].value_counts()
        if len(anomaly_counts) > 0:
            labels = ['Normal', 'Anomaly'] if len(anomaly_counts) == 2 else ['Anomaly'] if anomaly_counts.index[0] else ['Normal']
            axes[1, 1].pie(anomaly_counts.values, labels=labels, autopct='%1.1f%%')
        axes[1, 1].set_title('Anomaly Distribution')
        
        # 6. Correlation heatmap
        try:
            corr_cols = ['elem_reduction_pct', 'delta_p95_pct', 'delta_max_pct', 'delta_mean_pct',
                        'orig_el', 'orig_p95', 'orig_mean', 'orig_std', 'orig_mass_kg']
            available_cols = [col for col in corr_cols if col in df_clipped.columns]
            corr_data = df_clipped[available_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2], 
                       fmt='.2f', cbar_kws={'shrink': 0.8})
            axes[1, 2].set_title('Feature Correlation Matrix')
        except Exception as e:
            print(f"  Warning: Could not create correlation heatmap: {e}")
            axes[1, 2].text(0.5, 0.5, 'Correlation matrix\nnot available', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Feature Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig('pairwise_thinning_with_mass_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'pairwise_thinning_with_mass_analysis.png'")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Continuing without plots...")

def clean_finite_data(df):
    """Remove rows with infinite or NaN values"""
    print("\n=== CLEANING FINITE DATA ===")
    
    initial_count = len(df)
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_mask = np.isinf(df[numeric_cols]).any(axis=1)
    infinite_count = infinite_mask.sum()
    
    # Check for NaN values
    nan_mask = df.isnull().any(axis=1)
    nan_count = nan_mask.sum()
    
    # Remove problematic rows
    clean_mask = ~(infinite_mask | nan_mask)
    df_clean = df[clean_mask].copy()
    
    removed_count = initial_count - len(df_clean)
    
    print(f"  Initial rows: {initial_count}")
    print(f"  Infinite values: {infinite_count}")
    print(f"  NaN values: {nan_count}")
    print(f"  Removed rows: {removed_count}")
    print(f"  Clean rows: {len(df_clean)}")
    
    return df_clean

def main():
    """Main function with robust error handling"""
    print("=== BUILDING PAIRWISE THINNING DATASET ===")
    
    try:
        # Load dataset with mass/volume data
        df = load_dataset()
        
        # Build pairwise dataset
        pairwise_df = build_pairwise_dataset(df)
        
        if len(pairwise_df) == 0:
            print("❌ No valid pairwise data found!")
            return
        
        # Clean finite data
        pairwise_clean = clean_finite_data(pairwise_df)
        
        if len(pairwise_clean) == 0:
            print("❌ No clean data remaining after filtering!")
            return
        
        # Analyze the data
        analyze_pairwise_data(pairwise_clean)
        
        # Create visualizations
        create_visualizations(pairwise_clean)
        
        # Save results
        # Save finite dataset
        finite_path = "pairwise_finite.csv"
        pairwise_clean.to_csv(finite_path, index=False)
        print(f"\n✅ Finite dataset saved to: {finite_path}")
        
        # Save clipped dataset for EDA
        df_clipped, _ = clean_data_for_visualization(pairwise_clean)
        eda_path = "pairwise_eda_clipped.csv"
        df_clipped.to_csv(eda_path, index=False)
        print(f"✅ Clipped dataset saved to: {eda_path}")
        
        # Final summary
        print(f"\n📊 FINAL SUMMARY:")
        print(f"  Total valid pairs: {len(pairwise_clean)}")
        print(f"  Anomalies: {pairwise_clean['anomaly'].sum()}")
        print(f"  Mean element reduction: {pairwise_clean['elem_reduction_pct'].mean():.1f}%")
        print(f"  Mean P95 stress change: {pairwise_clean['delta_p95_pct'].mean():.1f}%")
        print(f"  Element reduction range: [{pairwise_clean['elem_reduction_pct'].min():.1f}%, {pairwise_clean['elem_reduction_pct'].max():.1f}%]")
        print(f"  P95 stress change range: [{pairwise_clean['delta_p95_pct'].min():.1f}%, {pairwise_clean['delta_p95_pct'].max():.1f}%]")
        print(f"  Dataset ready for ML training!")
        
        return pairwise_clean
        
    except Exception as e:
        print(f"❌ Error in main function: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    main()
