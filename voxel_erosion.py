#!/usr/bin/env python3
"""
Voxel Erosion Tool
Creates thinner versions of voxel models by eroding each part by 1 layer
while preserving material/color labels for FEA analysis comparison.
"""

import numpy as np
import os
from scipy.ndimage import binary_erosion
from collections import defaultdict

def erode_voxel_parts(npz_path, output_path=None, erosion_layers=1, min_thickness=3, max_reduction=0.2):
    """
    Create a thinner version of voxel model by eroding each part by specified layers.
    
    Args:
        npz_path (str): Path to input voxels_filled_indices_colors.npz file
        output_path (str): Path for output .npz file (default: adds '_eroded' suffix)
        erosion_layers (int): Number of layers to erode (default: 1)
        min_thickness (int): Minimum thickness to preserve (default: 2)
    
    Returns:
        dict: Statistics about the erosion process
    """
    print(f"\n=== VOXEL EROSION ANALYSIS ===")
    print(f"Input file: {npz_path}")
    
    # Load voxel data
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Voxel data not found: {npz_path}")
    
    data = np.load(npz_path)
    indices = data["indices"].astype(np.int64)   # (N,3) voxel coordinates
    colors = data["colors"].astype(np.uint8)     # (N,4) RGBA colors
    pitch_arr = data["pitch"]
    pitch = float(pitch_arr.item() if pitch_arr.size == 1 else np.ravel(pitch_arr)[0])
    transform = data["transform"].astype(np.float64)  # (4,4) transform matrix
    
    print(f"Original voxels: {len(indices)}")
    print(f"Pitch: {pitch:.6f}")
    
    # Get unique colors (parts/materials)
    unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
    num_parts = len(unique_colors)
    print(f"Unique parts: {num_parts}")
    
    # Create 3D grid for erosion operations
    imin, jmin, kmin = indices.min(axis=0)
    imax, jmax, kmax = indices.max(axis=0)
    
    Nx = int(imax - imin + 1)
    Ny = int(jmax - jmin + 1) 
    Nz = int(kmax - kmin + 1)
    
    print(f"Grid dimensions: {Nx} x {Ny} x {Nz}")
    
    # Create 3D occupancy grid
    grid_shape = (Nx, Ny, Nz)
    occupancy_grid = np.zeros(grid_shape, dtype=bool)
    
    # Fill grid with voxel positions
    for idx in indices:
        i, j, k = idx - [imin, jmin, kmin]
        occupancy_grid[i, j, k] = True
    
    # Process each part separately
    eroded_indices = []
    eroded_colors = []
    part_stats = {}
    
    for part_idx in range(num_parts):
        part_color = unique_colors[part_idx]
        part_mask = (color_indices == part_idx)
        part_voxels = indices[part_mask]
        
        if len(part_voxels) == 0:
            continue
            
        print(f"\nProcessing part {part_idx}: {len(part_voxels)} voxels")
        
        # Create 3D mask for this part
        part_grid = np.zeros(grid_shape, dtype=bool)
        for voxel in part_voxels:
            i, j, k = voxel - [imin, jmin, kmin]
            part_grid[i, j, k] = True
        
        # Check if part is thick enough for erosion
        original_count = np.sum(part_grid)
        
        # Apply erosion with safety limits
        eroded_grid = part_grid.copy()
        original_count = np.sum(part_grid)
        
        for layer in range(erosion_layers):
            temp_eroded = binary_erosion(eroded_grid)
            if np.sum(temp_eroded) == 0:
                print(f"  Warning: Part {part_idx} would disappear after {layer+1} erosion layers")
                break
            
            # Check if reduction would be too aggressive
            temp_count = np.sum(temp_eroded)
            reduction = (original_count - temp_count) / original_count
            
            if reduction > max_reduction:
                print(f"  Part {part_idx} reduction would be {reduction:.1%} (limit: {max_reduction:.1%}), stopping erosion")
                break
            
            eroded_grid = temp_eroded
        
        # Safety check: if part becomes too thin, keep original
        eroded_count = np.sum(eroded_grid)
        if eroded_count < min_thickness:
            print(f"  Part {part_idx} too thin after erosion, keeping original")
            eroded_grid = part_grid.copy()
            eroded_count = original_count
        
        # Convert back to voxel coordinates
        eroded_voxels = []
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    if eroded_grid[i, j, k]:
                        voxel_coords = [i + imin, j + jmin, k + kmin]
                        eroded_voxels.append(voxel_coords)
        
        eroded_voxels = np.array(eroded_voxels, dtype=np.int64)
        
        # Add to results
        if len(eroded_voxels) > 0:
            eroded_indices.extend(eroded_voxels)
            eroded_colors.extend([part_color] * len(eroded_voxels))
            
            # Store statistics
            part_stats[part_idx] = {
                'original_voxels': original_count,
                'eroded_voxels': len(eroded_voxels),
                'reduction_percent': (1 - len(eroded_voxels) / original_count) * 100,
                'color': part_color
            }
            
            print(f"  Original: {original_count} voxels")
            print(f"  Eroded: {len(eroded_voxels)} voxels")
            print(f"  Reduction: {part_stats[part_idx]['reduction_percent']:.1f}%")
    
    # Convert to numpy arrays
    if len(eroded_indices) == 0:
        print("Warning: No voxels remaining after erosion!")
        return None
    
    eroded_indices = np.array(eroded_indices, dtype=np.int64)
    eroded_colors = np.array(eroded_colors, dtype=np.uint8)
    
    print(f"\nErosion complete:")
    print(f"Original voxels: {len(indices)}")
    print(f"Eroded voxels: {len(eroded_indices)}")
    print(f"Total reduction: {(1 - len(eroded_indices) / len(indices)) * 100:.1f}%")
    
    # Save eroded data
    if output_path is None:
        base_path = npz_path.replace('.npz', '')
        output_path = f"{base_path}_eroded.npz"
    
    np.savez_compressed(
        output_path,
        indices=eroded_indices,
        colors=eroded_colors,
        pitch=np.array([pitch], np.float32),
        transform=transform.astype(np.float32)
    )
    
    print(f"Eroded data saved to: {output_path}")
    
    return {
        'output_path': output_path,
        'original_voxels': len(indices),
        'eroded_voxels': len(eroded_indices),
        'reduction_percent': (1 - len(eroded_indices) / len(indices)) * 100,
        'part_stats': part_stats
    }

def process_all_models(base_dir="voxel_out", erosion_layers=1, min_thickness=3, max_reduction=0.2):
    """
    Process all voxel models in the directory to create eroded versions.
    
    Args:
        base_dir (str): Base directory containing voxel models
        erosion_layers (int): Number of layers to erode
        min_thickness (int): Minimum thickness to preserve
    """
    print("=== BATCH VOXEL EROSION ===")
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return
    
    model_dirs = [d for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d))]
    
    if not model_dirs:
        print("No model directories found")
        return
    
    print(f"Found {len(model_dirs)} models: {model_dirs}")
    
    results = {}
    for model_dir in model_dirs:
        model_path = os.path.join(base_dir, model_dir)
        npz_path = os.path.join(model_path, "voxels_filled_indices_colors.npz")
        
        if not os.path.exists(npz_path):
            print(f"Skipping {model_dir}: No voxel data found")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing: {model_dir}")
        print(f"{'='*50}")
        
        try:
            result = erode_voxel_parts(npz_path, erosion_layers=erosion_layers, 
                                     min_thickness=min_thickness, max_reduction=max_reduction)
            if result:
                results[model_dir] = result
        except Exception as e:
            print(f"Error processing {model_dir}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print("BATCH EROSION COMPLETE")
    print(f"{'='*50}")
    print(f"Successfully processed: {len(results)}/{len(model_dirs)} models")
    
    for model, stats in results.items():
        print(f"\n{model}:")
        print(f"  Original: {stats['original_voxels']} voxels")
        print(f"  Eroded: {stats['eroded_voxels']} voxels")
        print(f"  Reduction: {stats['reduction_percent']:.1f}%")
    
    return results

def main():
    """Main function for command-line usage"""
    import sys
    
    erosion_layers = 1
    min_thickness = 3
    max_reduction = 0.2
    
    if len(sys.argv) > 1:
        try:
            erosion_layers = int(sys.argv[1])
        except ValueError:
            print(f"Invalid erosion layers '{sys.argv[1]}', using default: {erosion_layers}")
    
    if len(sys.argv) > 2:
        try:
            min_thickness = int(sys.argv[2])
        except ValueError:
            print(f"Invalid min thickness '{sys.argv[2]}', using default: {min_thickness}")
    
    if len(sys.argv) > 3:
        try:
            max_reduction = float(sys.argv[3])
        except ValueError:
            print(f"Invalid max reduction '{sys.argv[3]}', using default: {max_reduction}")
    
    print(f"Erosion parameters: {erosion_layers} layers, min thickness: {min_thickness}, max reduction: {max_reduction:.1%}")
    
    # Process all models
    results = process_all_models(erosion_layers=erosion_layers, min_thickness=min_thickness, max_reduction=max_reduction)
    
    if results:
        print(f"\nEroded models ready for FEA analysis!")
        print("Run: python voxel_to_fea_complete.py")
        print("This will analyze both original and eroded versions.")

if __name__ == "__main__":
    main()
