#!/usr/bin/env python3
"""
Pairwise Thinning Analysis
Builds pairwise comparison dataset and trains ML model to predict stress changes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_path="voxel_out/combined_stress_summary.csv"):
    """Load the combined stress summary data"""
    print("=== LOADING DATA ===")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Model types: {df['Model_Type'].value_counts().to_dict()}")
    return df

def build_pairwise_dataset(df):
    """Build pairwise comparison dataset"""
    print("\n=== BUILDING PAIRWISE DATASET ===")
    
    pairwise_data = []
    
    # Group by Model and Part
    for (model, part), group in df.groupby(['Model', 'Part']):
        if len(group) != 2:
            print(f"Skipping {model}-{part}: {len(group)} records (need exactly 2)")
            continue
        
        # Find original and eroded
        orig_row = group[group['Model_Type'] == 'original']
        ero_row = group[group['Model_Type'] == 'eroded']
        
        if len(orig_row) == 0 or len(ero_row) == 0:
            print(f"Skipping {model}-{part}: missing original or eroded")
            continue
        
        orig = orig_row.iloc[0]
        ero = ero_row.iloc[0]
        
        # Calculate metrics
        elem_reduction_pct = (orig['ElementCount'] - ero['ElementCount']) / orig['ElementCount'] * 100
        delta_p95_pct = (ero['vonMises_p95_Pa'] - orig['vonMises_p95_Pa']) / orig['vonMises_p95_Pa'] * 100
        delta_max_pct = (ero['vonMises_max_Pa'] - orig['vonMises_max_Pa']) / orig['vonMises_max_Pa'] * 100
        
        # Check for valid reduction
        if elem_reduction_pct <= 0:
            print(f"Skipping {model}-{part}: invalid reduction {elem_reduction_pct:.1f}%")
            continue
        
        # Create pairwise record
        pair_record = {
            'Model': model,
            'Part': part,
            'orig_el': orig['ElementCount'],
            'ero_el': ero['ElementCount'],
            'orig_p95': orig['vonMises_p95_Pa'],
            'ero_p95': ero['vonMises_p95_Pa'],
            'orig_max': orig['vonMises_max_Pa'],
            'ero_max': ero['vonMises_max_Pa'],
            'orig_mean': orig['vonMises_mean_Pa'],
            'ero_mean': ero['vonMises_mean_Pa'],
            'orig_std': orig['vonMises_std_Pa'],
            'ero_std': ero['vonMises_std_Pa'],
            'elem_reduction_pct': elem_reduction_pct,
            'delta_p95_pct': delta_p95_pct,
            'delta_max_pct': delta_max_pct,
            'anomaly': delta_p95_pct < 0
        }
        
        pairwise_data.append(pair_record)
    
    pairwise_df = pd.DataFrame(pairwise_data)
    print(f"Created {len(pairwise_df)} pairwise comparisons")
    
    return pairwise_df

def analyze_pairwise_data(pairwise_df):
    """Analyze the pairwise dataset"""
    print("\n=== PAIRWISE DATA ANALYSIS ===")
    
    # Basic stats
    print(f"Total pairs: {len(pairwise_df)}")
    print(f"Unique models: {pairwise_df['Model'].nunique()}")
    print(f"Unique parts: {pairwise_df['Part'].nunique()}")
    print(f"Anomalies (delta_p95_pct < 0): {pairwise_df['anomaly'].sum()}")
    print(f"Anomaly rate: {pairwise_df['anomaly'].mean()*100:.1f}%")
    
    # Element reduction stats
    print(f"\nElement Reduction Statistics:")
    print(f"  Mean: {pairwise_df['elem_reduction_pct'].mean():.1f}%")
    print(f"  Std: {pairwise_df['elem_reduction_pct'].std():.1f}%")
    print(f"  Min: {pairwise_df['elem_reduction_pct'].min():.1f}%")
    print(f"  Max: {pairwise_df['elem_reduction_pct'].max():.1f}%")
    
    # P95 stress change stats
    print(f"\nP95 Stress Change Statistics:")
    print(f"  Mean: {pairwise_df['delta_p95_pct'].mean():.1f}%")
    print(f"  Std: {pairwise_df['delta_p95_pct'].std():.1f}%")
    print(f"  Min: {pairwise_df['delta_p95_pct'].min():.1f}%")
    print(f"  Max: {pairwise_df['delta_p95_pct'].max():.1f}%")
    
    # Max stress change stats
    print(f"\nMax Stress Change Statistics:")
    print(f"  Mean: {pairwise_df['delta_max_pct'].mean():.1f}%")
    print(f"  Std: {pairwise_df['delta_max_pct'].std():.1f}%")
    print(f"  Min: {pairwise_df['delta_max_pct'].min():.1f}%")
    print(f"  Max: {pairwise_df['delta_max_pct'].max():.1f}%")
    
    return pairwise_df

def create_visualizations(pairwise_df):
    """Create visualizations of the pairwise data"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Pairwise Thinning Analysis', fontsize=16)
    
    # 1. Element reduction vs P95 stress change
    axes[0, 0].scatter(pairwise_df['elem_reduction_pct'], pairwise_df['delta_p95_pct'], alpha=0.6)
    axes[0, 0].set_xlabel('Element Reduction (%)')
    axes[0, 0].set_ylabel('P95 Stress Change (%)')
    axes[0, 0].set_title('Element Reduction vs P95 Stress Change')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(pairwise_df['elem_reduction_pct'], pairwise_df['delta_p95_pct'], 1)
    p = np.poly1d(z)
    axes[0, 0].plot(pairwise_df['elem_reduction_pct'], p(pairwise_df['elem_reduction_pct']), "r--", alpha=0.8)
    
    # 2. Element reduction vs Max stress change
    axes[0, 1].scatter(pairwise_df['elem_reduction_pct'], pairwise_df['delta_max_pct'], alpha=0.6)
    axes[0, 1].set_xlabel('Element Reduction (%)')
    axes[0, 1].set_ylabel('Max Stress Change (%)')
    axes[0, 1].set_title('Element Reduction vs Max Stress Change')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Distribution of element reductions
    axes[0, 2].hist(pairwise_df['elem_reduction_pct'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 2].set_xlabel('Element Reduction (%)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Element Reductions')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution of P95 stress changes
    axes[1, 0].hist(pairwise_df['delta_p95_pct'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('P95 Stress Change (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of P95 Stress Changes')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Anomaly analysis
    anomaly_counts = pairwise_df['anomaly'].value_counts()
    axes[1, 1].pie(anomaly_counts.values, labels=['Normal', 'Anomaly'], autopct='%1.1f%%')
    axes[1, 1].set_title('Anomaly Distribution')
    
    # 6. Correlation heatmap
    corr_data = pairwise_df[['elem_reduction_pct', 'delta_p95_pct', 'delta_max_pct', 
                           'orig_el', 'orig_p95', 'orig_mean', 'orig_std']].corr()
    sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('pairwise_thinning_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'pairwise_thinning_analysis.png'")

def train_ml_model(pairwise_df):
    """Train ML model to predict stress changes"""
    print("\n=== TRAINING ML MODEL ===")
    
    # Prepare features and target
    feature_columns = ['elem_reduction_pct', 'orig_el', 'orig_p95', 'orig_mean', 'orig_std']
    X = pairwise_df[feature_columns].values
    y = pairwise_df['delta_p95_pct'].values
    
    print(f"Feature columns: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split by Model (grouped split to avoid leakage)
    unique_models = pairwise_df['Model'].unique()
    train_models, test_models = train_test_split(unique_models, test_size=0.2, random_state=42)
    
    train_mask = pairwise_df['Model'].isin(train_models)
    test_mask = pairwise_df['Model'].isin(test_models)
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Train set: {len(X_train)} samples from {len(train_models)} models")
    print(f"Test set: {len(X_test)} samples from {len(test_models)} models")
    
    # Train GradientBoostingRegressor
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\nModel Performance:")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Train MAE: {train_mae:.3f}%")
    print(f"  Test MAE: {test_mae:.3f}%")
    
    # Feature importances
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importances:")
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Save model
    import joblib
    joblib.dump(model, 'pairwise_thinning_model.pkl')
    print(f"\nModel saved as 'pairwise_thinning_model.pkl'")
    
    return model, importance_df

def main():
    """Main analysis pipeline"""
    print("=== PAIRWISE THINNING ANALYSIS ===")
    
    # Load data
    df = load_and_prepare_data()
    
    # Build pairwise dataset
    pairwise_df = build_pairwise_dataset(df)
    
    # Analyze data
    pairwise_df = analyze_pairwise_data(pairwise_df)
    
    # Save pairwise dataset
    pairwise_df.to_csv('pairwise_thinning_dataset.csv', index=False)
    print(f"\nPairwise dataset saved as 'pairwise_thinning_dataset.csv'")
    
    # Create visualizations
    create_visualizations(pairwise_df)
    
    # Train ML model
    model, importance_df = train_ml_model(pairwise_df)
    
    print(f"\n✅ Analysis complete!")
    print(f"Pairwise dataset: {len(pairwise_df)} comparisons")
    print(f"Anomalies: {pairwise_df['anomaly'].sum()}")
    print(f"Model R²: {r2_score(pairwise_df['delta_p95_pct'], model.predict(pairwise_df[['elem_reduction_pct', 'orig_el', 'orig_p95', 'orig_mean', 'orig_std']])):.3f}")

if __name__ == "__main__":
    main()
