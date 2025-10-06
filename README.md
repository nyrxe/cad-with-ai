# CAD with AI - Voxelizer part

An PLY file model to voxel pipeline with color propagation for CAD models. Process multiple files with organized batch processing.

##  Quick Start

### 1. Setup
```bash
# Clone the repository
git clone https://github.com/nyrxe/cad-with-ai-.git
cd cad-with-ai-

# Create virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your PLY Files
```bash
# Create input directory (if not exists)
mkdir input

# Copy your PLY files to the input folder
# You can add multiple PLY files for batch processing
```

### 3. Run Voxelization
```bash
# Process all PLY files in input/ folder
python voxelize_local.py
```

### 4. Run Complete FEA Analysis (Optional)
```bash
# Install CalculiX first (see below)
python install_calculix.py  # Check CalculiX installation

# Run complete FEA analysis with part-based results
python voxel_to_fea_complete.py
```

## 📁 Output Structure

Each PLY file gets its own organized output folder:

```
voxel_out/
<<<<<<< HEAD
├── model1/                    # Folder for model1.ply
│   ├── voxels_filled_colored_boxes.ply    # 🎨 Colored voxel mesh
│   ├── voxels_filled_colored_points.ply   # 🎨 Colored point cloud
│   ├── vox_filled_boxes.ply              # Uncolored filled voxels
│   ├── vox_surface_boxes.ply             # Surface voxels only
│   ├── vox_occupancy_sparse.npz          # Raw voxel data
│   ├── voxels_filled_indices_colors.npz  # FEA-ready voxel data
│   └── fea_analysis/                     # FEA results (if run)
│       ├── model.inp                     # CalculiX input file
│       ├── model.dat                     # CalculiX results
│       ├── part_stress_summary.csv      # Part stress statistics (MAIN RESULT)
│       └── part_detailed_results.csv    # Detailed element data by part
└── model2/                    # Folder for model2.ply
    └── ... (same structure)
=======
    model1/                    # Folder for model1.ply
     -- voxels_filled_colored_boxes.ply    # Colored voxel mesh
     -- voxels_filled_colored_points.ply   # Colored point cloud
     -- vox_filled_boxes.ply              # Uncolored filled voxels
     -- vox_surface_boxes.ply             # Surface voxels only
     -- vox_occupancy_sparse.npz          # Raw voxel data
   model2/                    # Folder for model2.ply
     ... 
>>>>>>> 8d8348650594437d0e69842239a703fa63ae0352
```

##  Features

<<<<<<< HEAD
- **✅ Batch Processing** - Process multiple PLY files at once
- **✅ Color Propagation** - Preserves original mesh colors in voxels
- **✅ Organized Output** - Each file gets its own folder
- **✅ Multiple Formats** - Box meshes, point clouds, raw data
- **✅ Surface & Filled** - Both surface and interior voxels
- **✅ Memory Efficient** - Chunked processing for large models
- **✅ FEA Analysis** - Convert voxels to CalculiX FEA models
- **✅ Stress Analysis** - Run finite element stress calculations
- **✅ Part-based Results** - Analyze results by material/color regions
=======
- **Batch Processing** - Process multiple PLY files at once
- **Color Propagation** - Preserves original mesh colors in voxels
- **Organized Output** - Each file gets its own folder
- **Multiple Formats** - Box meshes, point clouds, raw data
- **Surface & Filled** - Both surface and interior voxels
- **Memory Efficient** - Chunked processing for large models
>>>>>>> 8d8348650594437d0e69842239a703fa63ae0352

## Requirements

- Python 3.8+
- trimesh
- numpy
- scipy
- rtree
- pandas
- CalculiX (for FEA analysis)

## 🔧 Usage Examples

### Single File Processing
```bash
# Add one PLY file to input/ folder
echo "model.ply" > input/model.ply
python voxelize_local.py
```

### Batch Processing
```bash
# Add multiple PLY files to input/ folder
cp *.ply input/
python voxelize_local.py
```

##  Output Files Explained


`voxels_filled_colored_boxes.ply`  **Main result** - Colored voxel mesh (view in MeshLab) 
`voxels_filled_colored_points.ply`  Colored point cloud of voxel centers |
`vox_filled_boxes.ply`  Uncolored filled voxel mesh |
`vox_surface_boxes.ply` Surface voxels only |
`vox_occupancy_sparse.npz`  Raw voxel data with positions and transform |

## Viewing Results

Open `voxels_filled_colored_boxes.ply` in:
- **MeshLab** (recommended)..


##  Technical Details

- **Resolution**: 128 voxels along longest axis (configurable)
- **Color Propagation**: Uses proximity queries to map face colors to voxels
- **Memory Safe**: Processes large models in chunks
- **Watertight Support**: Works with both open and closed meshes

<<<<<<< HEAD
## 🔧 CalculiX Installation

For FEA analysis, you need CalculiX installed:

### Windows
1. Download from: https://www.calculix.de/
2. Extract and add to PATH
3. Or place `ccx.exe` in your project directory

### Linux
```bash
sudo apt-get install calculix-ccx
```

### macOS
```bash
brew install calculix
```

### Verify Installation
```bash
python install_calculix.py
```

## 🐛 Troubleshooting
=======
##  Troubleshooting
>>>>>>> 8d8348650594437d0e69842239a703fa63ae0352

**No PLY files found**
- Make sure files are in `input/` folder
- Check file extensions are `.ply`

**Out of memory?**
- Reduce target resolution in code
- Process files one at a time

**Colors not showing**
- Ensure your PLY files have face colors
- Check mesh has proper color data

<<<<<<< HEAD
**CalculiX not found?**
- Run `python install_calculix.py` for installation help
- Make sure CalculiX is in your PATH
- Check CalculiX installation with `ccx --version`

## 📝 License
=======
##License
>>>>>>> 8d8348650594437d0e69842239a703fa63ae0352

This project is open source.
