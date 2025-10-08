# Voxel Erosion and Thick vs Thin FEA Analysis

This workflow allows you to create thinner versions of your voxel models and compare FEA results between thick and thin versions.

## 🎯 **What This Does**

1. **Creates thinner models** by eroding each part/material by 1 layer
2. **Preserves material labels** so you can still distinguish parts
3. **Runs FEA analysis** on both original and eroded models
4. **Combines results** in unified CSV files for comparison
5. **Safeguards** against parts disappearing completely

## 📋 **Workflow Steps**

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

## 🔧 **Detailed Usage**

### Voxel Erosion Tool (`voxel_erosion.py`)

**Basic usage:**
```bash
python voxel_erosion.py
```

**Custom parameters:**
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

## 📊 **Output Files**

### Individual Model Results
```
voxel_out/
├── thin1/
│   ├── fea_analysis/
│   │   ├── part_stress_summary_original.csv
│   │   └── part_detailed_results_original.csv
│   └── fea_analysis_eroded/
│       ├── part_stress_summary_eroded.csv
│       └── part_detailed_results_eroded.csv
```

### Combined Results
```
voxel_out/
├── combined_stress_summary.csv
├── combined_detailed_results.csv
└── thick_vs_thin_comparison.csv
```

## 📈 **Comparison Analysis**

The `thick_vs_thin_comparison.csv` file contains:

| Column | Description |
|--------|-------------|
| `Model` | Model name (thin1, thin2, etc.) |
| `Part` | Part/material name |
| `Original_Max_Stress_MPa` | Max stress in thick model |
| `Eroded_Max_Stress_MPa` | Max stress in thin model |
| `Stress_Change_Percent` | % change in max stress |
| `Element_Reduction_Percent` | % reduction in elements |

## 🛡️ **Safeguards**

### Erosion Safety
- **Minimum thickness check**: Prevents parts from disappearing
- **Part preservation**: If erosion would remove entire part, keeps original
- **Color label preservation**: Materials remain distinguishable

### FEA Safety
- **Large model detection**: Warns before processing large models
- **User confirmation**: Asks before proceeding with time-consuming analysis
- **Error handling**: Graceful failure with detailed error messages

## ⚙️ **Parameters**

### Erosion Parameters
- **Erosion layers**: Number of voxel layers to remove (default: 1)
- **Minimum thickness**: Minimum voxels to preserve (default: 2)

### FEA Parameters
- **Resolution**: 32 (optimal balance of speed/accuracy)
- **Material properties**: Aluminum (E=70 GPa, ν=0.33)
- **Loading**: 1 MPa pressure on +X face
- **Boundary conditions**: Fixed on -X face

## 🎯 **Use Cases**

### Design Optimization
- Compare stress in thick vs thin designs
- Identify critical regions that need more material
- Optimize material usage while maintaining safety

### Manufacturing Analysis
- Simulate material removal processes
- Analyze effects of machining/erosion
- Study wear and degradation

### Research Applications
- Study stress concentration effects
- Analyze geometric sensitivity
- Validate design assumptions

## 📝 **Example Results**

```
Comparison Statistics:
Models compared: 6
Parts compared: 12
Average stress change: +15.3%
Average element reduction: -25.7%

Top 5 stress increases (thick → thin):
  thin1 - PART_000: +45.2%
  thin2 - PART_001: +32.1%
  thin3 - PART_000: +28.7%
```

## 🔍 **Troubleshooting**

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

## 💡 **Tips**

1. **Start with 1 erosion layer** for initial analysis
2. **Use 32 resolution** for optimal speed/accuracy balance
3. **Check comparison results** to understand stress changes
4. **Use safeguards** to prevent parts from disappearing
5. **Combine results** for comprehensive analysis

This workflow gives you powerful insights into how material thickness affects stress distribution in your designs!
