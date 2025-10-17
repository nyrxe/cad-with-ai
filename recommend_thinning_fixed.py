#!/usr/bin/env python3
"""
Fixed Thinning Recommendation System
Uses trained GradientBoosting model to recommend optimal thickness reductions
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ThinningRecommender:
    """AI-powered thinning recommendation system"""
    
    def __init__(self, model_dir="ai_models"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_models()
    
    def load_models(self):
        """Load trained GradientBoosting model and scaler"""
        print("=== LOADING TRAINED MODELS ===")
        
        try:
            # Load GradientBoosting model
            model_path = os.path.join(self.model_dir, "GradientBoosting_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = joblib.load(model_path)
            print(f"✅ Loaded GradientBoosting model from {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scalers.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['main']
            print(f"✅ Loaded scaler from {scaler_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please run enhanced_thickness_optimizer.py first to train the models.")
            return False
    
    def get_feature_schema(self):
        """Get the feature schema from pairwise_finite.csv"""
        print("=== LOADING FEATURE SCHEMA ===")
        
        try:
            # Load a sample of pairwise_finite.csv to get feature schema
            sample_df = pd.read_csv("pairwise_finite.csv", nrows=1)
            
            # Use the EXACT same features as used in training
            # This should match the features used in enhanced_thickness_optimizer.py
            expected_features = [
                'elem_reduction_pct', 'orig_el', 'orig_p95', 'orig_mean', 'orig_std',
                'orig_volume_m3', 'orig_mass_kg', 'pitch_m', 'density_kg_m3',
                'ero_el', 'ero_volume_m3', 'ero_mass_kg'
            ]
            
            # Check which features are actually available
            available_features = [col for col in expected_features if col in sample_df.columns]
            missing_features = [col for col in expected_features if col not in sample_df.columns]
            
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                print(f"Available features: {list(sample_df.columns)}")
            
            self.feature_columns = available_features
            print(f"✅ Feature schema: {self.feature_columns}")
            print(f"Total features: {len(self.feature_columns)}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading feature schema: {e}")
            return False
    
    def load_original_parts(self, csv_path):
        """Load original parts data"""
        print(f"\n=== LOADING ORIGINAL PARTS: {csv_path} ===")
        
        if not os.path.exists(csv_path):
            print(f"❌ File not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} original parts")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['model_id', 'part_id', 'orig_el', 'orig_mean', 'orig_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Check for optional columns
        optional_cols = ['orig_mass_kg', 'orig_volume_m3', 'pitch_m', 'density_kg_m3']
        available_optional = [col for col in optional_cols if col in df.columns]
        print(f"Available optional columns: {available_optional}")
        
        return df
    
    def create_feature_row(self, part_data, reduction_pct):
        """Create feature row for a given reduction percentage"""
        features = {}
        
        # Base features
        features['elem_reduction_pct'] = reduction_pct
        features['orig_el'] = part_data['orig_el']
        features['orig_p95'] = part_data.get('orig_p95', part_data['orig_mean'] * 1.2)  # Estimate if missing
        features['orig_mean'] = part_data['orig_mean']
        features['orig_std'] = part_data['orig_std']
        
        # Volume and mass features (original values)
        features['orig_volume_m3'] = part_data.get('orig_volume_m3', 0.001)
        features['orig_mass_kg'] = part_data.get('orig_mass_kg', 1.0)
        
        # Pitch and density
        features['pitch_m'] = part_data.get('pitch_m', 0.001)
        features['density_kg_m3'] = part_data.get('density_kg_m3', 2700)
        
        # Eroded values (simulated based on reduction)
        features['ero_el'] = int(part_data['orig_el'] * (1 - reduction_pct/100))
        features['ero_volume_m3'] = features['orig_volume_m3'] * (1 - reduction_pct/100)
        features['ero_mass_kg'] = features['orig_mass_kg'] * (1 - reduction_pct/100)
        
        return features
    
    def predict_stress_change(self, features):
        """Predict stress change for given features"""
        try:
            # Create feature array in correct order
            feature_array = np.array([[features[col] for col in self.feature_columns]])
            
            # Scale features
            features_scaled = self.scaler.transform(feature_array)
            
            # Predict stress change
            predicted_delta = self.model.predict(features_scaled)[0]
            
            return predicted_delta
            
        except Exception as e:
            print(f"  Warning: Prediction failed: {e}")
            return None
    
    def apply_decision_rule(self, predicted_delta_p95_pct, safety_margin=10.0, max_stress_increase=20.0):
        """Apply decision rule for thinning acceptance"""
        if predicted_delta_p95_pct is None:
            return False, "Prediction failed"
        
        # Rule: predicted_delta_p95_pct + safety_margin <= max_stress_increase
        if predicted_delta_p95_pct + safety_margin <= max_stress_increase:
            return True, "Accepted"
        else:
            return False, f"Stress increase too high: {predicted_delta_p95_pct + safety_margin:.1f}% > {max_stress_increase}%"
    
    def recommend_thinning(self, part_data, reduction_steps):
        """Recommend optimal thinning for a single part"""
        best_reduction = 0.0
        best_prediction = 0.0
        best_decision = "No thinning recommended"
        
        for reduction_pct in reduction_steps:
            # Create features for this reduction
            features = self.create_feature_row(part_data, reduction_pct)
            
            # Predict stress change
            predicted_delta = self.predict_stress_change(features)
            
            if predicted_delta is not None:
                # Apply decision rule
                is_accepted, reason = self.apply_decision_rule(predicted_delta)
                
                if is_accepted:
                    best_reduction = reduction_pct
                    best_prediction = predicted_delta
                    best_decision = reason
                else:
                    # Stop at first rejection
                    break
        
        return best_reduction, best_prediction, best_decision
    
    def process_parts(self, df, reduction_steps):
        """Process all parts and generate recommendations"""
        print(f"\n=== PROCESSING {len(df)} PARTS ===")
        
        recommendations = []
        rejected_count = 0
        processed_count = 0
        
        for idx, (_, part_data) in enumerate(df.iterrows()):
            # Skip rows with NaN values
            if part_data.isnull().any():
                print(f"  Skipping part {part_data.get('part_id', idx)}: NaN values")
                continue
            
            processed_count += 1
            
            # Get recommendation
            recommended_reduction, predicted_delta, decision_reason = self.recommend_thinning(
                part_data, reduction_steps
            )
            
            # Calculate expected savings
            orig_mass = part_data.get('orig_mass_kg', 0)
            orig_volume = part_data.get('orig_volume_m3', 0)
            
            expected_mass_saving = orig_mass * (recommended_reduction / 100) if orig_mass > 0 else 0
            expected_volume_saving = orig_volume * (recommended_reduction / 100) if orig_volume > 0 else 0
            
            # Track rejections
            if recommended_reduction == 0:
                rejected_count += 1
            
            # Create recommendation record
            recommendation = {
                'model_id': part_data['model_id'],
                'part_id': part_data['part_id'],
                'recommended_reduction_pct': recommended_reduction,
                'predicted_delta_p95_pct': predicted_delta,
                'expected_mass_saving_kg': expected_mass_saving,
                'expected_volume_saving_m3': expected_volume_saving,
                'decision_reason': decision_reason
            }
            
            recommendations.append(recommendation)
            
            # Progress logging
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{len(df)} parts...")
        
        print(f"✅ Processed {processed_count} parts successfully")
        print(f"❌ Rejected {rejected_count} parts (no safe thinning possible)")
        
        return recommendations
    
    def print_summary_stats(self, recommendations):
        """Print summary statistics"""
        print(f"\n=== SUMMARY STATISTICS ===")
        
        if not recommendations:
            print("No recommendations generated")
            return
        
        df_rec = pd.DataFrame(recommendations)
        
        # Basic stats
        total_parts = len(df_rec)
        rejected_parts = (df_rec['recommended_reduction_pct'] == 0).sum()
        accepted_parts = total_parts - rejected_parts
        
        print(f"Total parts: {total_parts}")
        print(f"Accepted for thinning: {accepted_parts} ({accepted_parts/total_parts*100:.1f}%)")
        print(f"Rejected: {rejected_parts} ({rejected_parts/total_parts*100:.1f}%)")
        
        if accepted_parts > 0:
            # Reduction statistics
            reductions = df_rec[df_rec['recommended_reduction_pct'] > 0]['recommended_reduction_pct']
            print(f"\nReduction Statistics (accepted parts):")
            print(f"  Mean reduction: {reductions.mean():.1f}%")
            print(f"  Median reduction: {reductions.median():.1f}%")
            print(f"  Min reduction: {reductions.min():.1f}%")
            print(f"  Max reduction: {reductions.max():.1f}%")
            
            # Stress change statistics
            stress_changes = df_rec[df_rec['recommended_reduction_pct'] > 0]['predicted_delta_p95_pct']
            print(f"\nPredicted Stress Changes:")
            print(f"  Mean stress increase: {stress_changes.mean():.1f}%")
            print(f"  Median stress increase: {stress_changes.median():.1f}%")
            print(f"  Max stress increase: {stress_changes.max():.1f}%")
            
            # Mass/volume savings
            if 'expected_mass_saving_kg' in df_rec.columns:
                mass_savings = df_rec[df_rec['recommended_reduction_pct'] > 0]['expected_mass_saving_kg']
                total_mass_saving = mass_savings.sum()
                print(f"\nMaterial Savings:")
                print(f"  Total mass saving: {total_mass_saving:.2f} kg")
                print(f"  Mean mass saving per part: {mass_savings.mean():.3f} kg")
        
        # Decision reasons
        print(f"\nDecision Reasons:")
        reason_counts = df_rec['decision_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} parts")

def main():
    """Main recommendation pipeline"""
    print("=== FIXED THINNING RECOMMENDATION SYSTEM ===")
    
    # Initialize recommender
    recommender = ThinningRecommender()
    
    if not recommender.model or not recommender.scaler:
        print("❌ Failed to load models. Exiting.")
        return
    
    # Get feature schema
    if not recommender.get_feature_schema():
        print("❌ Failed to load feature schema. Exiting.")
        return
    
    # Load original parts (you'll need to provide this CSV)
    original_parts_path = input("Enter path to original parts CSV (or press Enter for 'original_parts.csv'): ").strip()
    if not original_parts_path:
        original_parts_path = "original_parts.csv"
    
    df = recommender.load_original_parts(original_parts_path)
    if df is None:
        print("❌ Failed to load original parts. Exiting.")
        return
    
    # Define reduction steps (0% to 40% in 2.5% steps)
    reduction_steps = np.arange(0, 42.5, 2.5)  # 0, 2.5, 5.0, ..., 40.0
    print(f"\nReduction steps: {reduction_steps}")
    
    # Process all parts
    recommendations = recommender.process_parts(df, reduction_steps)
    
    # Save recommendations
    if recommendations:
        output_path = "thinning_recommendations_fixed.csv"
        df_rec = pd.DataFrame(recommendations)
        df_rec.to_csv(output_path, index=False)
        print(f"\n✅ Recommendations saved to: {output_path}")
        
        # Print summary
        recommender.print_summary_stats(recommendations)
    else:
        print("❌ No recommendations generated")

if __name__ == "__main__":
    main()

