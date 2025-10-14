#!/usr/bin/env python3
"""
Run FEA analysis only on thinned models
"""

import os
import sys
sys.path.append('.')

from voxel_to_fea_complete import process_single_model

def main():
    """Run FEA analysis only on thinned models"""
    print("=== RUNNING FEA ANALYSIS ON THINNED MODELS ONLY ===")
    
    voxel_out_dir = "voxel_out"
    if not os.path.exists(voxel_out_dir):
        print(f"Voxel output directory not found: {voxel_out_dir}")
        return
    
    # Find all voxel data files
    voxel_dirs = [d for d in os.listdir(voxel_out_dir) 
                  if os.path.isdir(os.path.join(voxel_out_dir, d))]
    
    if not voxel_dirs:
        print("No voxel data directories found in voxel_out/")
        return
    
    print(f"Found {len(voxel_dirs)} voxel datasets")
    
    # Process only thinned models
    successful_thinned = 0
    
    for voxel_dir in voxel_dirs:
        voxel_path = os.path.join(voxel_out_dir, voxel_dir)
        
        # Check if thinned version exists
        thinned_npz = os.path.join(voxel_path, "voxels_filled_indices_colors_thinned.npz")
        if os.path.exists(thinned_npz):
            print(f"\n{'='*60}")
            print(f"PROCESSING THINNED MODEL: {voxel_dir}")
            print(f"{'='*60}")
            if process_single_model(voxel_path, "eroded"):
                successful_thinned += 1
        else:
            print(f"No thinned version found for {voxel_dir}")
    
    print(f"\n{'='*60}")
    print(f"THINNED FEA ANALYSIS COMPLETE")
    print(f"Successfully processed thinned models: {successful_thinned}/{len(voxel_dirs)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
