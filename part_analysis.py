#!/usr/bin/env python3
"""
Part-based stress analysis from voxel colors
Groups elements by color/material and calculates part statistics
"""

import os
import numpy as np
import csv
import math
from collections import defaultdict

def von_mises(sxx, syy, szz, sxy, syz, sxz):
    return math.sqrt(0.5*((sxx-syy)**2 + (syy-szz)**2 + (szz-sxx)**2) + 3*(sxy**2 + syz**2 + sxz**2))

def load_voxel_data(npz_path):
    """Load voxel indices and colors"""
    data = np.load(npz_path)
    indices = data["indices"]  # (N,3)
    colors = data["colors"]    # (N,4) RGBA
    return indices, colors

def extract_stress_data(dat_path):
    """Extract stress data from CalculiX results"""
    if not os.path.exists(dat_path):
        print(f"Results file not found: {dat_path}")
        return []
    
    with open(dat_path, "r") as f:
        lines = f.readlines()
    
    # Find stress section
    stress_start = None
    for i, line in enumerate(lines):
        if "stresses (elem, integ.pnt.,sxx,syy,szz,sxy,sxz,syz)" in line:
            stress_start = i + 1
            break
    
    if stress_start is None:
        print("No stress section found")
        return []
    
    # Extract stress data
    stress_data = []
    for i in range(stress_start, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) >= 8:
            try:
                elem_id = int(parts[0])
                sxx = float(parts[2])
                syy = float(parts[3]) 
                szz = float(parts[4])
                sxy = float(parts[5])
                sxz = float(parts[6])
                syz = float(parts[7])
                
                vm = von_mises(sxx, syy, szz, sxy, syz, sxz)
                stress_data.append([elem_id, sxx, syy, szz, sxy, syz, sxz, vm])
            except (ValueError, IndexError):
                continue
    
    return stress_data

def analyze_parts():
    """Analyze stress by part/color"""
    model_dir = "voxel_out/example"
    npz_path = os.path.join(model_dir, "voxels_filled_indices_colors.npz")
    dat_path = os.path.join(model_dir, "fea_analysis/model.dat")
    
    print("=== PART-BASED STRESS ANALYSIS ===")
    
    # Load voxel data
    if not os.path.exists(npz_path):
        print(f"Voxel data not found: {npz_path}")
        return
    
    indices, colors = load_voxel_data(npz_path)
    print(f"Loaded {len(indices)} voxels with colors")
    
    # Get unique colors (parts)
    unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
    num_parts = len(unique_colors)
    print(f"Found {num_parts} unique parts (colors)")
    
    # Show parts
    for i, color in enumerate(unique_colors):
        part_name = f"PART_{i:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
        count = np.sum(color_indices == i)
        print(f"  {part_name}: {count} voxels")
    
    # Load stress data
    stress_data = extract_stress_data(dat_path)
    if not stress_data:
        print("No stress data found")
        return
    
    print(f"Loaded {len(stress_data)} stress results")
    
    # Group elements by part
    part_stresses = defaultdict(list)
    for elem_data in stress_data:
        elem_id = elem_data[0]
        vm_stress = elem_data[7]  # von Mises stress
        
        # Find which part this element belongs to
        if elem_id <= len(color_indices):
            part_idx = color_indices[elem_id - 1]  # elem_id is 1-based
            part_stresses[part_idx].append(vm_stress)
    
    # Calculate part statistics
    part_summary = []
    for part_idx in range(num_parts):
        if part_idx in part_stresses:
            stresses = np.array(part_stresses[part_idx])
            color = unique_colors[part_idx]
            part_name = f"PART_{part_idx:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
            
            summary = {
                'Part': part_name,
                'Color_RGBA': f"({color[0]},{color[1]},{color[2]},{color[3]})",
                'ElementCount': len(stresses),
                'vonMises_max_Pa': np.max(stresses),
                'vonMises_mean_Pa': np.mean(stresses),
                'vonMises_min_Pa': np.min(stresses),
                'vonMises_p95_Pa': np.percentile(stresses, 95),
                'vonMises_std_Pa': np.std(stresses)
            }
            part_summary.append(summary)
            
            print(f"\n{part_name}:")
            print(f"  Elements: {len(stresses)}")
            print(f"  Max stress: {np.max(stresses):.2e} Pa ({np.max(stresses)/1e6:.1f} MPa)")
            print(f"  Mean stress: {np.mean(stresses):.2e} Pa ({np.mean(stresses)/1e6:.1f} MPa)")
            print(f"  95th percentile: {np.percentile(stresses, 95):.2e} Pa ({np.percentile(stresses, 95)/1e6:.1f} MPa)")
    
    # Save part summary CSV
    csv_path = os.path.join(model_dir, "fea_analysis/part_stress_summary.csv")
    with open(csv_path, "w", newline="") as f:
        if part_summary:
            writer = csv.DictWriter(f, fieldnames=part_summary[0].keys())
            writer.writeheader()
            writer.writerows(part_summary)
    
    print(f"\nPart summary saved to: {csv_path}")
    
    # Save detailed part data
    detailed_path = os.path.join(model_dir, "fea_analysis/part_detailed_results.csv")
    with open(detailed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Part", "ElementID", "vonMises_Pa", "Sxx_Pa", "Syy_Pa", "Szz_Pa", "Sxy_Pa", "Syz_Pa", "Szx_Pa"])
        
        for elem_data in stress_data:
            elem_id = elem_data[0]
            if elem_id <= len(color_indices):
                part_idx = color_indices[elem_id - 1]
                color = unique_colors[part_idx]
                part_name = f"PART_{part_idx:03d}_R{color[0]}G{color[1]}B{color[2]}A{color[3]}"
                
                writer.writerow([part_name, elem_id] + elem_data[1:])
    
    print(f"Detailed results saved to: {detailed_path}")

if __name__ == "__main__":
    analyze_parts()
