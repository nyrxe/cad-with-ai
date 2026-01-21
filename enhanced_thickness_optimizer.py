#!/usr/bin/env python3
"""
Enhanced Thickness Optimizer
Uses pairwise_finite.csv data to train ML models for thickness optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class EnhancedThicknessOptimizer:
    """Enhanced AI system for thickness optimization using pairwise data"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.results = {}
        self.feature_columns = None
        
    def load_pairwise_data(self, csv_path="pairwise_finite.csv"):
        """Load pairwise finite dataset"""
        print("=== LOADING PAIRWISE FINITE DATA ===")
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            print("Please run build_pairwise.py first to create the dataset.")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} pairwise comparisons")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['elem_reduction_pct', 'delta_p95_pct', 'delta_max_pct', 'delta_mean_pct']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Check for mass/volume data
        mass_volume_cols = ['orig_mass_kg', 'ero_mass_kg', 'orig_volume_m3', 'ero_volume_m3']
        available_mv_cols = [col for col in mass_volume_cols if col in df.columns]
        print(f"Available mass/volume columns: {available_mv_cols}")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for ML training"""
        print("\n=== PREPARING FEATURES ===")
        
        # Base features
        base_features = ['elem_reduction_pct', 'orig_el', 'orig_p95', 'orig_mean', 'orig_std']
        
        # Add mass/volume features if available
        feature_columns = base_features.copy()
        
        if 'orig_mass_kg' in df.columns and 'ero_mass_kg' in df.columns:
            # Calculate mass reduction percentage
            df['mass_reduction_pct'] = (df['orig_mass_kg'] - df['ero_mass_kg']) / df['orig_mass_kg'] * 100
            feature_columns.append('mass_reduction_pct')
            feature_columns.append('orig_mass_kg')
            print("  Added mass features")
        
        if 'orig_volume_m3' in df.columns and 'ero_volume_m3' in df.columns:
            # Calculate volume reduction percentage
            df['volume_reduction_pct'] = (df['orig_volume_m3'] - df['ero_volume_m3']) / df['orig_volume_m3'] * 100
            feature_columns.append('volume_reduction_pct')
            feature_columns.append('orig_volume_m3')
            print("  Added volume features")
        
        # Add pitch and density if available
        if 'pitch_m' in df.columns:
            feature_columns.append('pitch_m')
            print("  Added pitch feature")
        
        if 'density_kg_m3' in df.columns:
            feature_columns.append('density_kg_m3')
            print("  Added density feature")
        
        print(f"Feature columns: {feature_columns}")
        
        # Create feature matrix
        X = df[feature_columns].values
        y = df['delta_p95_pct'].values  # Target: P95 stress change
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        
        return X, y, feature_columns, df
    
    def train_models(self, X, y, feature_columns):
        """Train multiple ML models"""
        print("\n=== TRAINING MODELS ===")
        
        # Split data - adjust test size for small datasets
        n_samples = len(X)
        if n_samples < 5:
            # For very small datasets, use a smaller test set or skip split
            test_size = max(0.2, 1.0 / n_samples) if n_samples > 2 else 0.0
            if test_size == 0.0:
                # Use all data for training if too small
                X_train, X_test, y_train, y_test = X, X, y, y
                print(f"  Warning: Dataset too small ({n_samples} samples), using all data for training")
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Define models - adjust parameters for small datasets
        min_samples = min(2, len(X_train))  # Adjust for small datasets
        
        models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=50 if len(X_train) < 10 else 100, 
                max_depth=5 if len(X_train) < 10 else 10, 
                min_samples_split=min_samples,
                random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=50 if len(X_train) < 10 else 100, 
                learning_rate=0.1, 
                max_depth=3 if len(X_train) < 10 else 6,
                random_state=42
            ),
            'Ridge': Ridge(alpha=1.0),
            'LinearRegression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            
            # Cross-validation - adjust folds for small datasets
            n_train_samples = len(X_train)
            if n_train_samples < 5:
                # Use Leave-One-Out CV for very small datasets
                cv = LeaveOneOut() if n_train_samples > 1 else None
                if cv is None:
                    cv_scores = np.array([np.nan])
                    print(f"  Warning: Too few samples ({n_train_samples}) for cross-validation")
                else:
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='r2')
            else:
                # Use min(5, n_samples) folds
                n_folds = min(5, n_train_samples)
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=n_folds, scoring='r2')
            
            cv_mean = cv_scores.mean() if not np.isnan(cv_scores).all() else np.nan
            cv_std = cv_scores.std() if not np.isnan(cv_scores).all() else np.nan
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            print(f"  Train R²: {train_r2:.3f}")
            print(f"  Test R²: {test_r2:.3f}")
            print(f"  Train RMSE: {train_rmse:.3f}")
            print(f"  Test RMSE: {test_rmse:.3f}")
            print(f"  Train MAE: {train_mae:.3f}")
            print(f"  Test MAE: {test_mae:.3f}")
            if not np.isnan(cv_scores).all():
                print(f"  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            else:
                print(f"  CV R²: N/A (insufficient data)")
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                self.feature_importance[name] = dict(zip(feature_columns, importance))
                print(f"  Top features: {sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Select best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} (Test R² = {results[best_model_name]['test_r2']:.3f})")
        
        # Store all models
        self.models = {name: results[name]['model'] for name, model in models.items()}
        self.results = results
        
        return results
    
    def create_visualizations(self, df, X, y, feature_columns):
        """Create comprehensive visualizations"""
        print("\n=== CREATING VISUALIZATIONS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Enhanced Thickness Optimization Analysis', fontsize=16)
        
        # 1. Element reduction vs P95 stress change (filtered 0-20%)
        elem_filtered = df[(df['elem_reduction_pct'] >= 0) & (df['elem_reduction_pct'] <= 20)]
        axes[0, 0].scatter(elem_filtered['elem_reduction_pct'], elem_filtered['delta_p95_pct'], alpha=0.6, s=20)
        axes[0, 0].set_xlabel('Element Reduction (%)')
        axes[0, 0].set_ylabel('P95 Stress Change (%)')
        axes[0, 0].set_title('Element Reduction vs P95 Stress Change (0-20%)')
        axes[0, 0].set_xlim(0, 20)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add trend line
        try:
            if len(elem_filtered) > 1:
                z = np.polyfit(elem_filtered['elem_reduction_pct'], elem_filtered['delta_p95_pct'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(0, 20, 100)
                axes[0, 0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
        except:
            print("  Warning: Could not fit trend line for element reduction")
        
        # 2. Mass reduction vs stress change (if available, filtered 0-20%)
        if 'mass_reduction_pct' in df.columns:
            mass_filtered = df[(df['mass_reduction_pct'] >= 0) & (df['mass_reduction_pct'] <= 20)]
            axes[0, 1].scatter(mass_filtered['mass_reduction_pct'], mass_filtered['delta_p95_pct'], alpha=0.6, s=20)
            axes[0, 1].set_xlabel('Mass Reduction (%)')
            axes[0, 1].set_ylabel('P95 Stress Change (%)')
            axes[0, 1].set_title('Mass Reduction vs P95 Stress Change (0-20%)')
            axes[0, 1].set_xlim(0, 20)
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Mass data\nnot available', ha='center', va='center', 
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Mass Reduction vs P95 Stress Change')
        
        # 3. Volume reduction vs stress change (if available, filtered 0-20%)
        if 'volume_reduction_pct' in df.columns:
            volume_filtered = df[(df['volume_reduction_pct'] >= 0) & (df['volume_reduction_pct'] <= 20)]
            axes[0, 2].scatter(volume_filtered['volume_reduction_pct'], volume_filtered['delta_p95_pct'], alpha=0.6, s=20)
            axes[0, 2].set_xlabel('Volume Reduction (%)')
            axes[0, 2].set_ylabel('P95 Stress Change (%)')
            axes[0, 2].set_title('Volume Reduction vs P95 Stress Change (0-20%)')
            axes[0, 2].set_xlim(0, 20)
            axes[0, 2].grid(True, alpha=0.3)
        else:
            axes[0, 2].text(0.5, 0.5, 'Volume data\nnot available', ha='center', va='center', 
                           transform=axes[0, 2].transAxes)
            axes[0, 2].set_title('Volume Reduction vs P95 Stress Change')
        
        # 4. Model performance comparison
        model_names = list(self.results.keys())
        test_r2_scores = [self.results[name]['test_r2'] for name in model_names]
        bars = axes[1, 0].bar(model_names, test_r2_scores, alpha=0.7)
        axes[1, 0].set_ylabel('Test R² Score')
        axes[1, 0].set_title('Model Performance Comparison')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, test_r2_scores):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                          f'{score:.3f}', ha='center', va='bottom')
        
        # 5. Feature importance (best model)
        if self.best_model_name in self.feature_importance:
            features = list(self.feature_importance[self.best_model_name].keys())
            importance = list(self.feature_importance[self.best_model_name].values())
            axes[1, 1].barh(features, importance, alpha=0.7)
            axes[1, 1].set_xlabel('Feature Importance')
            axes[1, 1].set_title(f'Feature Importance ({self.best_model_name})')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available', ha='center', va='center', 
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
        
        # 6. Prediction vs Actual (best model)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_scaled = self.scalers['main'].transform(X_test)
        y_pred = self.best_model.predict(X_test_scaled)
        
        axes[1, 2].scatter(y_test, y_pred, alpha=0.6, s=20)
        axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', alpha=0.8)
        axes[1, 2].set_xlabel('Actual P95 Stress Change (%)')
        axes[1, 2].set_ylabel('Predicted P95 Stress Change (%)')
        axes[1, 2].set_title(f'Prediction vs Actual ({self.best_model_name})')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        axes[1, 2].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[1, 2].transAxes, 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('enhanced_thickness_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'enhanced_thickness_optimization_analysis.png'")
    
    def save_models(self, save_dir="ai_models"):
        """Save trained models and scalers"""
        print(f"\n=== SAVING MODELS ===")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            print(f"  Saved {name} model to {model_path}")
        
        # Save scalers
        scaler_path = os.path.join(save_dir, "scalers.pkl")
        joblib.dump(self.scalers, scaler_path)
        print(f"  Saved scalers to {scaler_path}")
        
        # Save feature columns for the recommendation system
        feature_columns_path = os.path.join(save_dir, "feature_columns.pkl")
        joblib.dump(self.feature_columns, feature_columns_path)
        print(f"  Saved feature columns to {feature_columns_path}")
        
        # Save feature importance
        if self.feature_importance:
            importance_path = os.path.join(save_dir, "feature_importance.pkl")
            joblib.dump(self.feature_importance, importance_path)
            print(f"  Saved feature importance to {importance_path}")
        
        # Save results summary
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
        
        summary_path = os.path.join(save_dir, "model_results_summary.pkl")
        joblib.dump(results_summary, summary_path)
        print(f"  Saved results summary to {summary_path}")
        
        print(f"All models saved to {save_dir}/")
    
    def print_final_summary(self):
        """Print final training summary"""
        print(f"\n📊 FINAL TRAINING SUMMARY:")
        print(f"Best model: {self.best_model_name}")
        print(f"Test R²: {self.results[self.best_model_name]['test_r2']:.3f}")
        print(f"Test RMSE: {self.results[self.best_model_name]['test_rmse']:.3f}")
        print(f"Test MAE: {self.results[self.best_model_name]['test_mae']:.3f}")
        
        print(f"\nAll model results:")
        for name, result in self.results.items():
            print(f"  {name}: R²={result['test_r2']:.3f}, RMSE={result['test_rmse']:.3f}, MAE={result['test_mae']:.3f}")
        
        if self.best_model_name in self.feature_importance:
            print(f"\nTop features ({self.best_model_name}):")
            sorted_features = sorted(self.feature_importance[self.best_model_name].items(), 
                                   key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.3f}")

def main():
    """Main training pipeline"""
    print("=== ENHANCED THICKNESS OPTIMIZER ===")
    
    # Initialize optimizer
    optimizer = EnhancedThicknessOptimizer()
    
    # Load pairwise data
    df = optimizer.load_pairwise_data()
    if df is None:
        return
    
    # Prepare features
    X, y, feature_columns, df = optimizer.prepare_features(df)
    
    # Train models
    results = optimizer.train_models(X, y, feature_columns)
    
    # Create visualizations
    optimizer.create_visualizations(df, X, y, feature_columns)
    
    # Save models
    optimizer.save_models()
    
    # Print final summary
    optimizer.print_final_summary()
    
    print(f"\n✅ Enhanced thickness optimization complete!")
    print(f"Use the saved models for thickness optimization predictions.")

if __name__ == "__main__":
    main()
