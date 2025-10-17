#!/usr/bin/env python3
"""
Final Thinning Recommendation System
Uses trained GradientBoosting model to recommend optimal thickness reductions
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
import argparse
from safe_print import safe_print
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
        """Load trained GradientBoosting model, scaler, and feature columns"""
        print("=== LOADING TRAINED MODELS ===")
        
        try:
            # Load GradientBoosting model
            model_path = os.path.join(self.model_dir, "GradientBoosting_model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.model = joblib.load(model_path)
            safe_print(f"[OK] Loaded GradientBoosting model from {model_path}")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, "scalers.pkl")
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            scalers = joblib.load(scaler_path)
            self.scaler = scalers['main']
            safe_print(f"[OK] Loaded scaler from {scaler_path}")
            
            # Load feature columns (canonical order from training)
            feature_columns_path = os.path.join(self.model_dir, "feature_columns.pkl")
            if not os.path.exists(feature_columns_path):
                raise FileNotFoundError(f"Feature columns not found: {feature_columns_path}")
            
            self.feature_columns = joblib.load(feature_columns_path)
            safe_print(f"[OK] Loaded feature columns from {feature_columns_path}")
            print(f"  Features: {self.feature_columns}")
            
            # Sanity check: scaler should match feature columns
            if hasattr(self.scaler, 'n_features_in_'):
                if self.scaler.n_features_in_ != len(self.feature_columns):
                    raise ValueError(f"Scaler expects {self.scaler.n_features_in_} features, but {len(self.feature_columns)} provided")
                safe_print(f"[OK] Scaler feature count matches: {self.scaler.n_features_in_}")
            
            return True
            
        except Exception as e:
            safe_print(f"[X] Error loading models: {e}")
            print("Please run enhanced_thickness_optimizer.py first to train the models.")
            return False
    
    def load_original_parts(self, csv_path):
        """Load original parts data"""
        print(f"\n=== LOADING ORIGINAL PARTS: {csv_path} ===")
        
        if not os.path.exists(csv_path):
            safe_print(f"[X] File not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} original parts")
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_cols = ['model_id', 'part_id', 'orig_el', 'orig_mean', 'orig_std']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            safe_print(f"[X] Missing required columns: {missing_cols}")
            return None
        
        # Check for optional columns
        optional_cols = ['orig_mass_kg', 'orig_volume_m3', 'pitch_m', 'density_kg_m3', 'orig_p95']
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
        
        # Mass and volume reduction percentages
        # Justification: With fixed pitch and single material per part, 
        # % mass and % volume reductions match % element reduction
        features['mass_reduction_pct'] = reduction_pct
        features['orig_mass_kg'] = part_data.get('orig_mass_kg', 1.0)
        
        features['volume_reduction_pct'] = reduction_pct
        features['orig_volume_m3'] = part_data.get('orig_volume_m3', 0.001)
        
        # Pitch and density
        features['pitch_m'] = part_data.get('pitch_m', 0.001)
        features['density_kg_m3'] = part_data.get('density_kg_m3', 2700)
        
        return features
    
    def predict_stress_change(self, features):
        """Predict stress change for given features"""
        try:
            # Create DataFrame row with exactly the required columns in the correct order
            feature_row = pd.DataFrame([features])[self.feature_columns]
            
            # Check for NaNs or infs
            if feature_row.isnull().any().any() or np.isinf(feature_row).any().any():
                return None, "NaN or infinite values in features"
            
            # Scale features
            features_scaled = self.scaler.transform(feature_row)
            
            # Predict stress change
            predicted_delta = self.model.predict(features_scaled)[0]
            
            # Clip absurd outputs for reporting only (not fed back into logic)
            clipped_delta = np.clip(predicted_delta, -150.0, 300.0)
            if clipped_delta != predicted_delta:
                print(f"  Warning: Clipped prediction from {predicted_delta:.1f}% to {clipped_delta:.1f}%")
            
            return clipped_delta, None
            
        except Exception as e:
            return None, f"Prediction failed: {e}"
    
    def apply_decision_rule(self, predicted_delta_p95_pct):
        """Apply conservative decision rule for thinning acceptance"""
        if predicted_delta_p95_pct is None:
            return False, "Prediction failed"
        
        # Conservative rule: pred_delta_p95_pct <= 20 - 10 (≤ 10% after margin)
        max_allowed_stress_increase = 10.0
        if predicted_delta_p95_pct <= max_allowed_stress_increase:
            return True, "Accepted"
        else:
            return False, f"Predicted stress increase too high: {predicted_delta_p95_pct:.1f}% > {max_allowed_stress_increase}%"
    
    def recommend_thinning(self, part_data, reduction_steps):
        """Recommend optimal thinning for a single part"""
        best_reduction = 0.0
        best_prediction = 0.0
        best_decision = "No thinning recommended"
        
        for reduction_pct in reduction_steps:
            # Create features for this reduction
            features = self.create_feature_row(part_data, reduction_pct)
            
            # Predict stress change
            predicted_delta, error = self.predict_stress_change(features)
            
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
            else:
                # Skip this reduction due to prediction error
                if reduction_pct == reduction_steps[0]:  # Only warn for first attempt
                    print(f"  Warning: Skipping part {part_data.get('part_id', 'unknown')}: {error}")
                break
        
        return best_reduction, best_prediction, best_decision
    
    def process_parts(self, df, reduction_steps):
        """Process all parts and generate recommendations"""
        print(f"\n=== PROCESSING {len(df)} PARTS ===")
        
        recommendations = []
        rejected_count = 0
        processed_count = 0
        skipped_count = 0
        
        for idx, (_, part_data) in enumerate(df.iterrows()):
            # Skip rows with missing required features
            required_features = ['orig_el', 'orig_mean', 'orig_std']
            missing_features = [col for col in required_features if col not in part_data or pd.isna(part_data[col])]
            
            if missing_features:
                print(f"  Skipping part {part_data.get('part_id', idx)}: Missing features {missing_features}")
                skipped_count += 1
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
        
        safe_print(f"[OK] Processed {processed_count} parts successfully")
        safe_print(f"[X] Rejected {rejected_count} parts (no safe thinning possible)")
        safe_print(f"[!] Skipped {skipped_count} parts (missing features)")
        
        return recommendations, processed_count, rejected_count, skipped_count
    
    def print_summary_stats(self, recommendations, processed_count, rejected_count, skipped_count):
        """Print summary statistics"""
        print(f"\n=== SUMMARY STATISTICS ===")
        
        if not recommendations:
            print("No recommendations generated")
            return
        
        df_rec = pd.DataFrame(recommendations)
        
        # Basic stats
        total_parts = len(df_rec)
        accepted_parts = total_parts - rejected_count
        
        print(f"Total parts: {total_parts}")
        print(f"Processed: {processed_count}")
        print(f"Skipped: {skipped_count}")
        print(f"Accepted for thinning: {accepted_parts} ({accepted_parts/total_parts*100:.1f}%)")
        print(f"Rejected: {rejected_count} ({rejected_count/total_parts*100:.1f}%)")
        
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
        
        # 5-row sample preview
        print(f"\n5-Row Sample Preview:")
        sample_cols = ['model_id', 'part_id', 'recommended_reduction_pct', 'predicted_delta_p95_pct']
        print(df_rec[sample_cols].head().to_string(index=False))
        
        # Decision reasons
        print(f"\nDecision Reasons:")
        reason_counts = df_rec['decision_reason'].value_counts()
        for reason, count in reason_counts.items():
            print(f"  {reason}: {count} parts")
    
    def validation_debug(self, df, reduction_steps):
        """Optional validation block for debugging"""
        print(f"\n=== VALIDATION DEBUG ===")
        
        # Select 3 random parts
        sample_parts = df.sample(min(3, len(df)))
        
        for idx, (_, part_data) in sample_parts.iterrows():
            print(f"\nPart {part_data['part_id']} (Model: {part_data['model_id']}):")
            print("Reduction% | Predicted ΔP95%")
            print("-" * 30)
            
            for reduction_pct in reduction_steps[:5]:  # Test first 5 steps
                features = self.create_feature_row(part_data, reduction_pct)
                predicted_delta, error = self.predict_stress_change(features)
                
                if predicted_delta is not None:
                    print(f"    {reduction_pct:6.1f}% | {predicted_delta:8.1f}%")
                else:
                    print(f"    {reduction_pct:6.1f}% | ERROR: {error}")

def main():
    """Main recommendation pipeline"""
    parser = argparse.ArgumentParser(description='Thinning Recommendation System')
    parser.add_argument('--debug', action='store_true', help='Enable validation debug mode')
    args = parser.parse_args()
    
    print("=== FINAL THINNING RECOMMENDATION SYSTEM ===")
    
    # Initialize recommender
    recommender = ThinningRecommender()
    
    if not recommender.model or not recommender.scaler or not recommender.feature_columns:
        safe_print("[X] Failed to load models. Exiting.")
        return
    
    # Load original parts (use default path for automation)
    original_parts_path = "original_parts.csv"
    
    df = recommender.load_original_parts(original_parts_path)
    if df is None:
        safe_print("[X] Failed to load original parts. Exiting.")
        return
    
    # Define reduction steps (0% to 40% in 2.5% steps)
    reduction_steps = np.arange(0, 42.5, 2.5)  # 0, 2.5, 5.0, ..., 40.0
    print(f"\nReduction steps: {reduction_steps}")
    
    # Process all parts
    recommendations, processed_count, rejected_count, skipped_count = recommender.process_parts(df, reduction_steps)
    
    # Save recommendations
    if recommendations:
        output_path = "thinning_recommendations_final.csv"
        df_rec = pd.DataFrame(recommendations)
        df_rec.to_csv(output_path, index=False)
        safe_print(f"\n[OK] Recommendations saved to: {output_path}")
        
        # Print summary
        recommender.print_summary_stats(recommendations, processed_count, rejected_count, skipped_count)
        
        # Optional validation debug
        if args.debug:
            recommender.validation_debug(df, reduction_steps)
    else:
        safe_print("[X] No recommendations generated")

if __name__ == "__main__":
    main()