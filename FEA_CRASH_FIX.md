# FEA Analysis Crash Fix - Troubleshooting Guide

## 🐛 Problem: Frontend Crashes During FEA Analysis

The frontend crashes when running FEA analysis. This document explains the fixes applied and how to troubleshoot.

---

## ✅ Fixes Applied

### 1. **Improved Error Handling in `voxel_to_fea_complete.py`**

**Changes:**
- ✅ Better CalculiX detection (tries multiple command names)
- ✅ Dynamic timeout based on model size
- ✅ Memory requirement estimation
- ✅ Detailed error messages
- ✅ Graceful failure (doesn't crash)

**Timeout Settings:**
- Small models (<10k nodes): 5 minutes
- Medium models (10k-20k): 10 minutes  
- Large models (20k-50k): 15 minutes
- Very large (>50k): 30 minutes

### 2. **Improved Frontend Error Handling**

**Changes:**
- ✅ Longer timeout for FEA scripts (1 hour)
- ✅ Better error messages in log
- ✅ Continues pipeline even if FEA fails
- ✅ Prevents crashes from propagating

---

## 🔍 Common Causes of Crashes

### **Cause 1: CalculiX Not Installed**

**Symptoms:**
- Error: "CalculiX not found"
- Script fails immediately

**Solution:**
```bash
# Windows: Download from https://www.calculix.de/
# Extract and add to PATH, or place ccx.exe in project directory

# Linux:
sudo apt-get install calculix-ccx

# macOS:
brew install calculix
```

**Verify Installation:**
```bash
ccx --version
```

---

### **Cause 2: Model Too Large (Memory)**

**Symptoms:**
- Crash during mesh generation
- "Out of memory" errors
- System becomes unresponsive

**Solution:**
Reduce voxel resolution when voxelizing:

```bash
# Default resolution is 32
# Try lower resolution for large models:
python voxelize_local.py 16  # Lower resolution = fewer voxels
```

**Memory Estimates:**
- 10k elements ≈ 100-200 MB RAM
- 50k elements ≈ 500 MB - 1 GB RAM
- 100k+ elements ≈ 2+ GB RAM

---

### **Cause 3: Model Too Large (Timeout)**

**Symptoms:**
- FEA runs for a long time then times out
- "Timeout" error messages

**Solution:**
1. **Reduce voxel resolution** (most effective):
   ```bash
   python voxelize_local.py 16  # Instead of default 32
   ```

2. **Process smaller models** or split into parts

3. **Increase timeout** (if needed, edit code):
   - Edit `voxel_to_fea_complete.py`
   - Increase timeout values in `run_calculix()`

---

### **Cause 4: CalculiX Crashes**

**Symptoms:**
- CalculiX starts then crashes
- No output files created
- Error codes in log

**Common Issues:**
- **Invalid mesh**: Check that voxelization completed successfully
- **Numerical instability**: Model may have geometry issues
- **File permissions**: Check write permissions in output directory

**Solution:**
```bash
# Check CalculiX input file
cat voxel_out/{model_name}/fea_analysis/model.inp | head -50

# Try running CalculiX manually
cd voxel_out/{model_name}/fea_analysis
ccx model
```

---

### **Cause 5: Frontend Thread Issues**

**Symptoms:**
- GUI freezes during FEA
- Application becomes unresponsive
- Windows shows "Not Responding"

**Solution:**
- ✅ **Fixed**: FEA now runs in separate thread with timeout
- ✅ **Fixed**: GUI stays responsive
- ✅ **Fixed**: Progress bar shows activity

---

## 🛠️ Troubleshooting Steps

### **Step 1: Check CalculiX Installation**

```bash
# Test CalculiX
ccx --version

# If not found, check PATH or install
```

### **Step 2: Check Model Size**

Look at the log output:
```
Model size: X nodes, Y elements
```

**If >50k nodes/elements:**
- Reduce voxel resolution
- Process smaller models
- Consider splitting model

### **Step 3: Check Available Memory**

**Windows:**
```powershell
Get-ComputerInfo | Select-Object TotalPhysicalMemory
```

**Linux:**
```bash
free -h
```

**Ensure:** At least 4 GB free RAM for large models

### **Step 4: Test FEA Manually**

```bash
# Navigate to FEA directory
cd voxel_out/{your_model}/fea_analysis

# Check input file exists
dir model.inp

# Run CalculiX manually
ccx model

# Check for errors
type model.dat | findstr /i "error"
```

### **Step 5: Check Log Output**

In the frontend log, look for:
- ✅ "CalculiX found" - Good!
- ❌ "CalculiX not found" - Install CalculiX
- ❌ "timed out" - Model too large
- ❌ "out of memory" - Reduce resolution
- ❌ "failed with return code" - Check CalculiX output

---

## 🎯 Quick Fixes

### **Fix 1: Reduce Voxel Resolution**

**Before running pipeline:**
```python
# Edit voxelize_local.py or pass resolution as argument
python voxelize_local.py 16  # Lower = faster, less accurate
```

**Recommended resolutions:**
- **Small models** (< 100mm): 32-64
- **Medium models** (100-500mm): 16-32
- **Large models** (> 500mm): 8-16

### **Fix 2: Skip FEA (Use Existing Data)**

If you have existing FEA results:
- The pipeline will continue even if FEA fails
- You can use previously generated CSV files
- AI recommendations may be less accurate without fresh FEA

### **Fix 3: Process One Model at a Time**

Instead of batch processing:
- Process models individually
- Check each one completes successfully
- Identify problematic models

---

## 📊 Expected Processing Times

**Voxelization:**
- Small model: 1-2 minutes
- Medium model: 2-5 minutes
- Large model: 5-10 minutes

**FEA Analysis:**
- Small model (<10k elements): 2-5 minutes
- Medium model (10k-50k): 5-15 minutes
- Large model (50k+): 15-30+ minutes

**If FEA takes >30 minutes:**
- Model is likely too large
- Reduce voxel resolution
- Or skip FEA and use existing data

---

## 🔧 Advanced Troubleshooting

### **Check CalculiX Input File**

```bash
# View input file
notepad voxel_out\{model}\fea_analysis\model.inp

# Check for issues:
# - Node count reasonable?
# - Element count reasonable?
# - Material properties set?
# - Boundary conditions defined?
```

### **Check CalculiX Output**

```bash
# Check for errors in output
type voxel_out\{model}\fea_analysis\model.dat | findstr /i "error warning"

# Check completion
type voxel_out\{model}\fea_analysis\model.sta
```

### **Monitor System Resources**

**Windows Task Manager:**
- Watch CPU usage (should be high during FEA)
- Watch Memory usage (should increase)
- Watch Disk usage (temporary files)

**If memory maxes out:**
- Reduce voxel resolution
- Close other applications
- Process smaller models

---

## ✅ Verification

After fixes, verify:

1. **CalculiX works:**
   ```bash
   ccx --version
   ```

2. **Small test model works:**
   - Use test_cube.ply or test_sphere.ply
   - Should complete in <5 minutes

3. **Frontend doesn't crash:**
   - GUI stays responsive
   - Progress bar shows activity
   - Log shows detailed messages

---

## 🆘 Still Having Issues?

### **Check These:**

1. ✅ CalculiX installed and in PATH
2. ✅ Sufficient RAM available (4+ GB free)
3. ✅ Voxel resolution appropriate for model size
4. ✅ Model voxelization completed successfully
5. ✅ Write permissions in output directory

### **Get More Info:**

Enable detailed logging:
```python
# In voxel_to_fea_complete.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Alternative: Skip FEA**

If FEA continues to fail:
1. Use existing FEA results from previous runs
2. Or skip FEA entirely (AI won't work, but other features will)
3. Focus on voxelization and thinning only

---

## 📝 Summary

**Main fixes:**
- ✅ Better error handling (no crashes)
- ✅ Dynamic timeouts (handles large models)
- ✅ Better error messages (easier debugging)
- ✅ Graceful failure (pipeline continues)

**Most common issue:**
- CalculiX not installed → Install CalculiX
- Model too large → Reduce voxel resolution

**Quick fix:**
```bash
# Reduce resolution and try again
python voxelize_local.py 16
```

The frontend should now handle FEA failures gracefully without crashing! 🎉

