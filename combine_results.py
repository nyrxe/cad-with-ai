#!/usr/bin/env python3
"""
Combine FEA Results from Original and Eroded Models
Creates unified CSV files comparing thick vs thin model results.
"""

import os
import pandas as pd
import glob

def combine_fea_results(base_dir="voxel_out"):
    """
    Combine FEA results from original and eroded models into unified CSV files.
    
    Args:
        base_dir (str): Base directory containing model results
    """
    print("=== COMBINING FEA RESULTS ===")
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    # Find all model directories
    model_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    if not model_dirs:
        print("No model directories found")
        return
    
    print(f"Found {len(model_dirs)} models: {model_dirs}")
    
    # Combine part stress summaries
    all_summaries = []
    all_detailed = []
    
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        
        # Process original model results
        original_summary = os.path.join(model_path, "fea_analysis", "part_stress_summary_original.csv")
        original_detailed = os.path.join(model_path, "fea_analysis", "part_detailed_results_original.csv")
        
        if os.path.exists(original_summary):
            df_summary = pd.read_csv(original_summary)
            df_summary['Model'] = model_dir
            all_summaries.append(df_summary)
            print(f"Added original results for {model_dir}")
        
        # Skip detailed results for AI - only need part summaries
        # if os.path.exists(original_detailed):
        #     df_detailed = pd.read_csv(original_detailed)
        #     df_detailed['Model'] = model_dir
        #     all_detailed.append(df_detailed)
        
        # Process eroded model results
        eroded_summary = os.path.join(model_path, "fea_analysis_eroded", "part_stress_summary_eroded.csv")
        eroded_detailed = os.path.join(model_path, "fea_analysis_eroded", "part_detailed_results_eroded.csv")
        
        if os.path.exists(eroded_summary):
            df_summary = pd.read_csv(eroded_summary)
            df_summary['Model'] = model_dir
            all_summaries.append(df_summary)
            print(f"Added eroded results for {model_dir}")
        
        # Skip detailed results for AI - only need part summaries
        # if os.path.exists(eroded_detailed):
        #     df_detailed = pd.read_csv(eroded_detailed)
        #     df_detailed['Model'] = model_dir
        #     all_detailed.append(df_detailed)
    
    # Combine and save results
    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        summary_path = os.path.join(base_dir, "combined_stress_summary.csv")
        combined_summary.to_csv(summary_path, index=False)
        print(f"\nCombined summary saved to: {summary_path}")
        print(f"Total records: {len(combined_summary)}")
        
        # Show summary statistics
        print(f"\nSummary by model type:")
        type_counts = combined_summary['Model_Type'].value_counts()
        for model_type, count in type_counts.items():
            print(f"  {model_type}: {count} parts")
    
    # Skip detailed results for AI - only need part summaries
    # if all_detailed:
    #     combined_detailed = pd.concat(all_detailed, ignore_index=True)
    #     detailed_path = os.path.join(base_dir, "combined_detailed_results.csv")
    #     combined_detailed.to_csv(detailed_path, index=False)
    #     print(f"Combined detailed results saved to: {detailed_path}")
    #     print(f"Total records: {len(combined_detailed)}")
    
    # Create comparison analysis
    if all_summaries:
        create_comparison_analysis(combined_summary, base_dir)
    
    print(f"\n✅ Results combination complete!")

def create_comparison_analysis(df, base_dir):
    """Create comparison analysis between original and eroded models"""
    print(f"\n=== CREATING COMPARISON ANALYSIS ===")
    
    # Group by model and part to compare original vs eroded
    comparison_data = []
    
    for model in df['Model'].unique():
        model_data = df[df['Model'] == model]
        
        for part in model_data['Part'].unique():
            part_data = model_data[model_data['Part'] == part]
            
            if len(part_data) == 2:  # Both original and eroded
                original = part_data[part_data['Model_Type'] == 'original'].iloc[0]
                eroded = part_data[part_data['Model_Type'] == 'eroded'].iloc[0]
                
                comparison = {
                    'Model': model,
                    'Part': part,
                    'Original_Max_Stress_MPa': original['vonMises_max_Pa'] / 1e6,
                    'Eroded_Max_Stress_MPa': eroded['vonMises_max_Pa'] / 1e6,
                    'Stress_Change_Percent': ((eroded['vonMises_max_Pa'] - original['vonMises_max_Pa']) / original['vonMises_max_Pa']) * 100,
                    'Original_Mean_Stress_MPa': original['vonMises_mean_Pa'] / 1e6,
                    'Eroded_Mean_Stress_MPa': eroded['vonMises_mean_Pa'] / 1e6,
                    'Original_Elements': original['ElementCount'],
                    'Eroded_Elements': eroded['ElementCount'],
                    'Element_Reduction_Percent': ((original['ElementCount'] - eroded['ElementCount']) / original['ElementCount']) * 100
                }
                comparison_data.append(comparison)
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(base_dir, "thick_vs_thin_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        print(f"Comparison analysis saved to: {comparison_path}")
        
        # Show key statistics
        print(f"\nComparison Statistics:")
        print(f"Models compared: {len(comparison_df['Model'].unique())}")
        print(f"Parts compared: {len(comparison_df)}")
        
        avg_stress_change = comparison_df['Stress_Change_Percent'].mean()
        avg_element_reduction = comparison_df['Element_Reduction_Percent'].mean()
        
        print(f"Average stress change: {avg_stress_change:.1f}%")
        print(f"Average element reduction: {avg_element_reduction:.1f}%")
        
        # Show models with highest stress changes
        print(f"\nTop 5 stress increases (thick → thin):")
        top_increases = comparison_df.nlargest(5, 'Stress_Change_Percent')[['Model', 'Part', 'Stress_Change_Percent']]
        for _, row in top_increases.iterrows():
            print(f"  {row['Model']} - {row['Part']}: +{row['Stress_Change_Percent']:.1f}%")

def main():
    """Main function"""
    import sys
    
    base_dir = "voxel_out"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    combine_fea_results(base_dir)

if __name__ == "__main__":
    main()
