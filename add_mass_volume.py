#!/usr/bin/env python3
"""
Add Mass and Volume Calculations to Stress Dataset
Loads pitch data from NPZ files and calculates volume/mass for each part
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from safe_print import safe_print

def load_existing_csv(csv_path="voxel_out/combined_stress_summary.csv"):
    """Load the existing stress summary CSV"""
    print("=== LOADING EXISTING CSV ===")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique models: {df['Model'].nunique()}")
    return df

def load_pitch_data(df, voxel_out_dir="voxel_out"):
    """Load pitch data from NPZ files for each model"""
    print("\n=== LOADING PITCH DATA FROM NPZ FILES ===")
    
    pitch_data = {}
    successful_models = 0
    failed_models = []
    
    unique_models = df['Model'].unique()
    print(f"Processing {len(unique_models)} unique models...")
    
    for model in unique_models:
        npz_path = os.path.join(voxel_out_dir, model, "voxels_filled_indices_colors.npz")
        
        if os.path.exists(npz_path):
            try:
                # Load NPZ file
                data = np.load(npz_path)
                pitch_arr = data["pitch"]
                
                # Extract pitch value
                if pitch_arr.size == 1:
                    pitch_m = float(pitch_arr.item())
                else:
                    pitch_m = float(pitch_arr.ravel()[0])
                
                pitch_data[model] = pitch_m
                successful_models += 1
                
                if successful_models <= 5:  # Show first 5 for debugging
                    print(f"  {model}: pitch = {pitch_m:.6f} m")
                    
            except Exception as e:
                print(f"  Error loading {model}: {e}")
                failed_models.append(model)
        else:
            print(f"  NPZ file not found: {npz_path}")
            failed_models.append(model)
    
    print(f"\nPitch loading results:")
    print(f"  Successful: {successful_models}/{len(unique_models)} models")
    print(f"  Failed: {len(failed_models)} models")
    
    if failed_models:
        print(f"  Failed models: {failed_models[:5]}{'...' if len(failed_models) > 5 else ''}")
    
    return pitch_data

def create_density_map():
    """Create density mapping for different materials"""
    density_map = {
        "(31,119,180,255)": 2700,    # Blue - Aluminum
        "(158,218,229,255)": 2700,   # Light blue - Aluminum  
        "(140,86,75,255)": 2700,      # Brown - Aluminum
        "(214,39,40,255)": 7800,      # Red - Steel
        "(152,223,138,255)": 2700,   # Green - Aluminum
        "(199,199,199,255)": 2700,   # Gray - Aluminum
    }
    
    print(f"\nDensity mapping:")
    for color, density in density_map.items():
        print(f"  {color}: {density} kg/m³")
    
    return density_map

def calculate_mass_volume(df, pitch_data, density_map):
    """Calculate volume and mass for each row"""
    print("\n=== CALCULATING MASS AND VOLUME ===")
    
    # Initialize new columns
    df['pitch_m'] = np.nan
    df['volume_m3'] = np.nan
    df['density_kg_m3'] = np.nan
    df['mass_kg'] = np.nan
    
    # Process each row
    for idx, row in df.iterrows():
        model = row['Model']
        color_rgba = row['Color_RGBA']
        element_count = row['ElementCount']
        
        # Get pitch for this model
        if model in pitch_data:
            pitch_m = pitch_data[model]
            df.at[idx, 'pitch_m'] = pitch_m
            
            # Calculate volume
            volume_m3 = element_count * (pitch_m ** 3)
            df.at[idx, 'volume_m3'] = volume_m3
            
            # Get density
            density_kg_m3 = density_map.get(color_rgba, 2700)  # Default to aluminum
            df.at[idx, 'density_kg_m3'] = density_kg_m3
            
            # Calculate mass
            mass_kg = volume_m3 * density_kg_m3
            df.at[idx, 'mass_kg'] = mass_kg
    
    # Count successful calculations
    successful_calculations = df['mass_kg'].notna().sum()
    print(f"Successfully calculated mass/volume for {successful_calculations}/{len(df)} rows")
    
    return df

def analyze_results(df):
    """Analyze the results and print statistics"""
    print("\n=== RESULTS ANALYSIS ===")
    
    # Filter to rows with successful calculations
    valid_df = df[df['mass_kg'].notna()].copy()
    
    if len(valid_df) == 0:
        print("[X] No valid calculations found!")
        return
    
    print(f"Valid calculations: {len(valid_df)} rows")
    
    # Pitch statistics
    pitch_stats = valid_df['pitch_m'].describe()
    print(f"\nPitch statistics (m):")
    print(f"  Mean: {pitch_stats['mean']:.6f}")
    print(f"  Std: {pitch_stats['std']:.6f}")
    print(f"  Min: {pitch_stats['min']:.6f}")
    print(f"  Max: {pitch_stats['max']:.6f}")
    
    # Volume statistics
    volume_stats = valid_df['volume_m3'].describe()
    print(f"\nVolume statistics (m³):")
    print(f"  Mean: {volume_stats['mean']:.6f}")
    print(f"  Std: {volume_stats['std']:.6f}")
    print(f"  Min: {volume_stats['min']:.6f}")
    print(f"  Max: {volume_stats['max']:.6f}")
    
    # Mass statistics
    mass_stats = valid_df['mass_kg'].describe()
    print(f"\nMass statistics (kg):")
    print(f"  Mean: {mass_stats['mean']:.3f}")
    print(f"  Std: {mass_stats['std']:.3f}")
    print(f"  Min: {mass_stats['min']:.3f}")
    print(f"  Max: {mass_stats['max']:.3f}")
    
    # Density distribution
    density_counts = valid_df['density_kg_m3'].value_counts()
    print(f"\nDensity distribution:")
    for density, count in density_counts.items():
        material = "Steel" if density == 7800 else "Aluminum"
        print(f"  {density} kg/m³ ({material}): {count} parts")
    
    # Model type analysis
    print(f"\nBy model type:")
    for model_type in valid_df['Model_Type'].unique():
        type_data = valid_df[valid_df['Model_Type'] == model_type]
        print(f"  {model_type}: {len(type_data)} parts")
        print(f"    Mean mass: {type_data['mass_kg'].mean():.3f} kg")
        print(f"    Mean volume: {type_data['volume_m3'].mean():.6f} m³")
    
    return valid_df

def show_sample_data(df, n_samples=5):
    """Show random sample rows for inspection"""
    print(f"\n=== RANDOM SAMPLE ROWS ({n_samples}) ===")
    
    # Get random sample
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=42)
    
    # Select key columns for display
    display_cols = ['Model', 'Part', 'Model_Type', 'ElementCount', 'pitch_m', 
                   'volume_m3', 'density_kg_m3', 'mass_kg', 'vonMises_max_Pa']
    
    # Show sample
    for idx, row in sample_df.iterrows():
        print(f"\nRow {idx}:")
        for col in display_cols:
            if col in row:
                value = row[col]
                if isinstance(value, float):
                    if col in ['pitch_m', 'volume_m3']:
                        print(f"  {col}: {value:.6f}")
                    elif col == 'mass_kg':
                        print(f"  {col}: {value:.3f}")
                    elif col == 'vonMises_max_Pa':
                        print(f"  {col}: {value:.1f}")
                    else:
                        print(f"  {col}: {value}")
                else:
                    print(f"  {col}: {value}")

def create_visualizations(df):
    """Create visualizations of the mass/volume data"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    import matplotlib.pyplot as plt
    
    # Filter to valid data
    valid_df = df[df['mass_kg'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid data for visualization")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Mass and Volume Analysis', fontsize=16)
    
    # 1. Volume vs Mass
    axes[0, 0].scatter(valid_df['volume_m3'], valid_df['mass_kg'], alpha=0.6)
    axes[0, 0].set_xlabel('Volume (m³)')
    axes[0, 0].set_ylabel('Mass (kg)')
    axes[0, 0].set_title('Volume vs Mass')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Element count vs Mass
    axes[0, 1].scatter(valid_df['ElementCount'], valid_df['mass_kg'], alpha=0.6)
    axes[0, 1].set_xlabel('Element Count')
    axes[0, 1].set_ylabel('Mass (kg)')
    axes[0, 1].set_title('Element Count vs Mass')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Mass distribution
    axes[0, 2].hist(valid_df['mass_kg'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Mass (kg)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Mass Distribution')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Volume distribution
    axes[1, 0].hist(valid_df['volume_m3'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Volume (m³)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Volume Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Mass by model type
    model_type_mass = valid_df.groupby('Model_Type')['mass_kg'].mean()
    axes[1, 1].bar(model_type_mass.index, model_type_mass.values, alpha=0.7)
    axes[1, 1].set_ylabel('Mean Mass (kg)')
    axes[1, 1].set_title('Mean Mass by Model Type')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Density distribution
    density_counts = valid_df['density_kg_m3'].value_counts()
    axes[1, 2].bar(density_counts.index, density_counts.values, alpha=0.7)
    axes[1, 2].set_xlabel('Density (kg/m³)')
    axes[1, 2].set_ylabel('Count')
    axes[1, 2].set_title('Density Distribution')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mass_volume_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure instead of showing it
    
    print("Visualizations saved as 'mass_volume_analysis.png'")

def main():
    """Main function"""
    print("=== ADDING MASS AND VOLUME CALCULATIONS ===")
    
    # Load existing CSV
    df = load_existing_csv()
    
    # Load pitch data from NPZ files
    pitch_data = load_pitch_data(df)
    
    # Create density mapping
    density_map = create_density_map()
    
    # Calculate mass and volume
    df = calculate_mass_volume(df, pitch_data, density_map)
    
    # Analyze results
    valid_df = analyze_results(df)
    
    # Show sample data
    show_sample_data(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save results
    output_path = "dataset_with_mass_volume.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[OK] Results saved to: {output_path}")
    
    # Final summary
    successful_models = len(pitch_data)
    total_models = df['Model'].nunique()
    successful_rows = df['mass_kg'].notna().sum()
    total_rows = len(df)
    
    print(f"\n[#] FINAL SUMMARY:")
    print(f"  Models with pitch data: {successful_models}/{total_models}")
    print(f"  Rows with mass/volume: {successful_rows}/{total_rows}")
    print(f"  Mean pitch: {df['pitch_m'].mean():.6f} m")
    print(f"  Mean volume: {df['volume_m3'].mean():.6f} m³")
    print(f"  Mean mass: {df['mass_kg'].mean():.3f} kg")

if __name__ == "__main__":
    main()
