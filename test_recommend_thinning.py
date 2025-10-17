#!/usr/bin/env python3
"""
Test the recommend_thinning.py script with real data
"""

import pandas as pd
import numpy as np
import os

def create_original_parts_from_real_data():
    """Create original_parts.csv from your real dataset_with_mass_volume.csv"""
    print("=== CREATING ORIGINAL PARTS FROM REAL DATA ===")
    
    # Load your real dataset
    df = pd.read_csv("dataset_with_mass_volume.csv")
    print(f"Loaded {len(df)} total records")
    
    # Filter for original parts only
    original_df = df[df['Model_Type'] == 'original'].copy()
    print(f"Found {len(original_df)} original parts")
    
    # Create the required format for recommend_thinning.py
    original_parts = []
    
    for _, row in original_df.iterrows():
        # Skip if missing critical data
        if (pd.isna(row['ElementCount']) or 
            pd.isna(row['vonMises_mean_Pa']) or 
            pd.isna(row['vonMises_std_Pa'])):
            continue
        
        part_data = {
            'model_id': row['Model'],
            'part_id': row['Part'],
            'orig_el': int(row['ElementCount']),
            'orig_p95': row['vonMises_p95_Pa'],
            'orig_mean': row['vonMises_mean_Pa'],
            'orig_std': row['vonMises_std_Pa'],
            'orig_mass_kg': row.get('mass_kg', 1.0),
            'orig_volume_m3': row.get('volume_m3', 0.001),
            'pitch_m': row.get('pitch_m', 0.001),
            'density_kg_m3': row.get('density_kg_m3', 2700)
        }
        
        original_parts.append(part_data)
    
    # Create DataFrame and save
    original_parts_df = pd.DataFrame(original_parts)
    original_parts_df.to_csv("original_parts.csv", index=False)
    
    print(f"✅ Created original_parts.csv with {len(original_parts_df)} real parts")
    print(f"Models: {original_parts_df['model_id'].nunique()}")
    print(f"Parts: {original_parts_df['part_id'].nunique()}")
    
    return original_parts_df

def test_recommend_thinning():
    """Test the recommend_thinning.py script"""
    print("\n=== TESTING RECOMMEND_THINNING.PY ===")
    
    # Check if required files exist
    required_files = [
        "ai_models/GradientBoosting_model.pkl",
        "ai_models/scalers.pkl",
        "pairwise_finite.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    
    # Create original parts dataset
    original_parts_df = create_original_parts_from_real_data()
    
    if len(original_parts_df) == 0:
        print("❌ No valid original parts found")
        return False
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Parts: {len(original_parts_df)}")
    print(f"  Models: {original_parts_df['model_id'].nunique()}")
    print(f"  Mean elements: {original_parts_df['orig_el'].mean():.0f}")
    print(f"  Mean stress: {original_parts_df['orig_mean'].mean()/1e6:.1f} MPa")
    print(f"  Mean mass: {original_parts_df['orig_mass_kg'].mean():.3f} kg")
    
    print(f"\n🚀 Ready to test recommend_thinning.py!")
    print(f"Run: python recommend_thinning.py")
    print(f"Or press Enter to run it now...")
    
    return True

if __name__ == "__main__":
    test_recommend_thinning()

