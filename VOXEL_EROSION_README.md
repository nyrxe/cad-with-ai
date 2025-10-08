# Voxel Erosion and Thick vs Thin FEA Analysis

This workflow allows you to create thinner versions of your voxel models and compare FEA results between thick and thin versions.

## **What This Does**

1. **Creates thinner models** by eroding each part/material by 1 layer
2. **Preserves material labels** so you can still distinguish parts
3. **Runs FEA analysis** on both original and eroded models
4. **Combines results** in unified CSV files for comparison
5. **Safeguards** against parts disappearing completely

##  **Workflow Steps**

### Step 1: Create Voxel Models (32 resolution)
```bash
python voxelize_local.py
```

### Step 2: Create Eroded Versions
```bash
python voxel_erosion.py
```

### Step 3: Run FEA Analysis on Both
```bash
python voxel_to_fea_complete.py
```

### Step 4: Combine Results for Comparison
```bash
python combine_results.py
```



**Custom parameters:** **if you want**
```bash
python voxel_erosion.py 2 3
# 2 = erosion layers
# 3 = minimum thickness to preserve
```

**What it does:**
- Loads `voxels_filled_indices_colors.npz` from each model
- Erodes each part/material by specified layers
- Saves as `voxels_filled_indices_colors_eroded.npz`
- Preserves color labels for material distinction
- Prevents parts from disappearing completely

### FEA Analysis (`voxel_to_fea_complete.py`)

**Enhanced to handle both versions:**
- Processes original models → `fea_analysis/`
- Processes eroded models → `fea_analysis_eroded/`
- Adds `Model_Type` column to CSV results
- Keeps results separate but comparable

### Results Combination (`combine_results.py`)

**Creates unified comparison files:**
- `combined_stress_summary.csv` - All part summaries
- `combined_detailed_results.csv` - All element results  
- `thick_vs_thin_comparison.csv` - Direct comparison analysis



##  **Comparison Analysis**

The `thick_vs_thin_comparison.csv` file contains:

 `Model`  Model name (thin1, thin2, ...) 
 `Part`  Part/material name 
 `Original_Max_Stress_MPa`  Max stress in thick model
 `Eroded_Max_Stress_MPa`  Max stress in thin model 
 `Stress_Change_Percent` % change in max stress 
 `Element_Reduction_Percent`  % reduction in elements 

##  **Safeguards**

### Erosion Safety
- **Minimum thickness check**: Prevents parts from disappearing
- **Part preservation**: If erosion would remove entire part, keeps original
- **Color label preservation**: Materials remain distinguishable

### FEA Safety
- **Large model detection**: Warns before processing large models
- **User confirmation**: Asks before proceeding with time-consuming analysis
- **Error handling**: Graceful failure with detailed error messages

##  **Parameters**

### Erosion Parameters
- **Erosion layers**: Number of voxel layers to remove (default: 1)
- **Minimum thickness**: Minimum voxels to preserve (default: 2)

### FEA Parameters
- **Resolution**: 32 (optimal balance of speed/accuracy)
- **Material properties**: Aluminum (E=70 GPa, ν=0.33)
- **Loading**: 1 MPa pressure on +X face
- **Boundary conditions**: Fixed on -X face



##  **Troubleshooting**

### No eroded models found
```bash
# Make sure you ran erosion first
python voxel_erosion.py
```

### Erosion removes entire parts
```bash
# Reduce erosion layers or increase minimum thickness
python voxel_erosion.py 1 5
```

### FEA analysis fails
```bash
# Check CalculiX installation
python install_calculix.py
```

