# CAD with AI - Advanced Thickness Optimization

An intelligent PLY-to-voxel pipeline with AI-powered thickness optimization for CAD models. Features advanced thinning algorithms, FEA analysis, mass/volume calculations, and machine learning models for optimal material usage.

## 🚀 Quick Start

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

### 3. Complete AI Pipeline
```bash
# Step 1: Voxelize PLY files
python voxelize_local.py

# Step 2: Create thinned versions (advanced erosion)
python voxel_thinning_advanced.py

# Step 3: Run FEA analysis on both original and thinned models
python voxel_to_fea_complete.py

# Alternative: Run FEA only on thinned models (faster)
python run_thinned_fea.py

# Step 4: Add mass and volume calculations
python add_mass_volume.py

# Step 5: Build pairwise comparison dataset
python build_pairwise.py

# Step 6: Train AI models for thickness optimization
python enhanced_thickness_optimizer.py
```

## 🤖 AI Features

### **Advanced Thickness Optimization**
- **Smart Thinning** - Interface protection and connectivity preservation
- **Mass/Volume Analysis** - Calculate material savings and density
- **AI Predictions** - ML models predict stress changes from thickness reduction
- **Safety Margins** - Automatic safety factor calculations
- **Material Optimization** - Suggest optimal thickness for target stress levels

### **Machine Learning Models**
- **RandomForest** - Ensemble learning for robust predictions
- **GradientBoosting** - Advanced boosting algorithms
- **Ridge/Linear** - Linear models for interpretability
- **Cross-validation** - Comprehensive model evaluation
- **Feature Importance** - Understand which factors matter most

### **Intelligent Data Processing**
- **Robust Error Handling** - Tolerance systems for floating point precision
- **Data Cleaning** - Automatic removal of infinite/NaN values
- **Outlier Detection** - Smart clipping for visualization
- **Pairwise Analysis** - Compare original vs thinned models

## 🔧 Core Features

### **Voxelization Pipeline**
- **Batch Processing** - Process multiple PLY files at once
- **Color Propagation** - Preserves original mesh colors in voxels
- **Memory Efficient** - Chunked processing for large models
- **Multiple Formats** - Box meshes, point clouds, raw data
- **Surface & Filled** - Both surface and interior voxels

### **FEA Analysis**
- **CalculiX Integration** - Professional finite element analysis
- **Stress Analysis** - von Mises stress calculations
- **Part-based Results** - Analyze results by material/color regions
- **Automated Workflow** - Complete FEA pipeline automation

### **Advanced Thinning**
- **Interface Protection** - Preserve material boundaries
- **Connectivity Checks** - Maintain structural integrity
- **Stress-aware Mode** - Protect high-stress regions
- **Safety Validation** - Comprehensive safety checks

## 📊 Output Files

### **Voxelization Results**
- `voxels_filled_colored_boxes.ply` - **Main result** - Colored voxel mesh
- `voxels_filled_colored_points.ply` - Colored point cloud
- `vox_occupancy_sparse.npz` - Raw voxel data with metadata

### **AI Analysis Results**
- `dataset_with_mass_volume.csv` - Complete dataset with mass/volume
- `pairwise_finite.csv` - Clean pairwise comparison data
- `pairwise_eda_clipped.csv` - Visualization-ready dataset
- `ai_models/` - Trained ML models and scalers

### **Visualizations**
- `mass_volume_analysis.png` - Mass/volume distribution analysis
- `pairwise_thinning_analysis.png` - Thinning vs stress relationships
- `enhanced_thickness_optimization_analysis.png` - AI model performance

## 🛠️ Requirements

### **Core Dependencies**
- Python 3.8+
- trimesh - 3D mesh processing
- numpy, scipy - Numerical computing
- pandas - Data analysis
- rtree - Spatial indexing

### **AI/ML Dependencies**
- matplotlib, seaborn - Visualization
- scikit-learn - Machine learning
- joblib - Model persistence

### **FEA Analysis**
- CalculiX - Finite element analysis
- (Optional) MeshLab - 3D visualization

## 🔧 Usage Examples

### **Complete AI Pipeline**
```bash
# Run the full AI thickness optimization pipeline
python voxelize_local.py && \
python voxel_thinning_advanced.py && \
python voxel_to_fea_complete.py && \
python add_mass_volume.py && \
python build_pairwise.py && \
python enhanced_thickness_optimizer.py
```

### **Individual Components**
```bash
# Just voxelization
python voxelize_local.py

# Just thinning analysis
python voxel_thinning_advanced.py

# FEA analysis on both original and thinned models
python voxel_to_fea_complete.py

# FEA analysis only on thinned models (faster)
python run_thinned_fea.py

# Just mass/volume analysis
python add_mass_volume.py

# Build pairwise dataset
python build_pairwise.py

# Train AI models
python enhanced_thickness_optimizer.py
```

### **AI Model Training**
```bash
# Train thickness optimization models
python enhanced_thickness_optimizer.py

# Use trained models for predictions
python thickness_recommender.py
```

## 📈 Technical Details

### **Voxelization**
- **Resolution**: 32-128 voxels along longest axis (configurable)
- **Color Propagation**: Proximity queries map face colors to voxels
- **Memory Safe**: Chunked processing for large models
- **Watertight Support**: Works with both open and closed meshes

### **AI Thickness Optimization**
- **Feature Engineering**: Element count, stress metrics, mass/volume
- **Model Selection**: Multiple algorithms with cross-validation
- **Performance Metrics**: R², RMSE, MAE with statistical significance
- **Safety Integration**: Yield stress and safety factor calculations

### **Advanced Thinning**
- **Interface Protection**: 1-voxel halo around material boundaries
- **Stress Awareness**: Protect high-stress regions automatically
- **Connectivity Preservation**: Maintain structural integrity
- **Validation**: Comprehensive safety and quality checks

## 🔧 CalculiX Installation

For FEA analysis, install CalculiX:

### **Windows**
1. Download from: https://www.calculix.de/
2. Extract and add to PATH
3. Or place `ccx.exe` in your project directory

### **Linux**
```bash
sudo apt-get install calculix-ccx
```

### **macOS**
```bash
brew install calculix
```

### **Verify Installation**
```bash
python install_calculix.py
```

## 📊 Performance

### **Typical Processing Times**
- **Voxelization**: 1-5 minutes per model
- **Thinning**: 2-10 minutes per model
- **FEA Analysis**: 5-30 minutes per model
- **AI Training**: 1-5 minutes for full dataset

### **Memory Requirements**
- **Small models** (< 10k elements): 2-4 GB RAM
- **Medium models** (10k-100k elements): 4-8 GB RAM
- **Large models** (> 100k elements): 8+ GB RAM

## 🐛 Troubleshooting

### **Common Issues**
- **CalculiX not found**: Check PATH or place ccx.exe in project directory
- **Memory errors**: Reduce voxel resolution or process models individually
- **Missing dependencies**: Run `pip install -r requirements.txt`
- **Visualization errors**: Install matplotlib with `pip install matplotlib`

### **Performance Tips**
- Use SSD storage for faster I/O
- Increase RAM for large models
- Process models in smaller batches
- Use lower voxel resolution for initial testing

## 📚 Documentation

- **Code Documentation**: Comprehensive docstrings in all scripts
- **API Reference**: Detailed function documentation
- **Examples**: Multiple usage examples and tutorials
- **Troubleshooting**: Common issues and solutions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. See LICENSE file for details.

## 🙏 Acknowledgments

- **Trimesh** - 3D mesh processing
- **CalculiX** - Finite element analysis
- **Scikit-learn** - Machine learning algorithms
- **Matplotlib/Seaborn** - Data visualization