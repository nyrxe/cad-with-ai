# FEA Crash Debugging - CalculiX 2.22 Installed

## ✅ Good News: CalculiX is Installed!

Your CalculiX 2.22 installation is **correct** - the version output is normal. The crash is likely due to something else.

---

## 🔍 Most Likely Causes (Since CalculiX Works)

### **1. Model Too Large (Most Common)**

**Symptoms:**
- Crash during mesh generation
- Crash during CalculiX solve
- "Out of memory" errors
- System becomes unresponsive

**Check Model Size:**
```python
import numpy as np
data = np.load('voxel_out/{your_model}/voxels_filled_indices_colors.npz')
indices = data['indices']
print(f"Voxels: {len(indices):,}")
```

**If >50,000 voxels:**
- Reduce resolution: `python voxelize_local.py 16`
- Or skip FEA for this model

---

### **2. CalculiX Process Crash (Not Installation)**

**Symptoms:**
- CalculiX starts but crashes
- No output files created
- Process exits with error code

**Test Manually:**
```bash
# Navigate to FEA directory
cd voxel_out\{your_model}\fea_analysis

# Check input file exists
dir model.inp

# Run CalculiX manually
ccx model

# Check for errors
type model.dat | findstr /i "error"
```

**Common CalculiX Errors:**
- **"singular matrix"** → Geometry issue
- **"memory"** → Model too large
- **"cannot open"** → File path issue
- **"element"** → Mesh generation issue

---

### **3. Python Subprocess Issue**

**Symptoms:**
- Frontend crashes but CalculiX works manually
- No error messages in log
- GUI freezes or closes

**This is what we're fixing** - the subprocess wrapper may be crashing.

---

### **4. Memory Exhaustion**

**Symptoms:**
- System slows down before crash
- Windows shows "low memory" warning
- Task Manager shows high memory usage

**Check Available RAM:**
- Need at least 4 GB free for large models
- Close other applications
- Reduce voxel resolution

---

## 🛠️ Diagnostic Steps

### **Step 1: Test CalculiX Manually**

```bash
# Find a model that was voxelized
cd voxel_out\{your_model}\fea_analysis

# Check if input file exists
dir model.inp

# Run CalculiX directly (bypass Python)
ccx model

# Watch for errors
```

**If CalculiX crashes here:**
- Problem is with CalculiX/model, not Python
- Check model size
- Check CalculiX output for errors

**If CalculiX works here:**
- Problem is with Python subprocess wrapper
- Use the fixes we applied

---

### **Step 2: Check Model Size**

```python
# Run this in Python
import numpy as np
import os

model_dir = "voxel_out/{your_model}"  # Replace with your model
npz_path = os.path.join(model_dir, "voxels_filled_indices_colors.npz")

data = np.load(npz_path)
indices = data['indices']
pitch = float(data['pitch'].item())

num_voxels = len(indices)
imin, jmin, kmin = indices.min(axis=0)
imax, jmax, kmax = indices.max(axis=0)

nx = imax - imin + 2
ny = jmax - jmin + 2
nz = kmax - kmin + 2
estimated_nodes = nx * ny * nz
estimated_elements = num_voxels

print(f"Voxels: {num_voxels:,}")
print(f"Estimated nodes: {estimated_nodes:,}")
print(f"Estimated elements: {estimated_elements:,}")

if estimated_elements > 100000:
    print("⚠️  VERY LARGE MODEL - likely to crash!")
    print("   Reduce resolution: python voxelize_local.py 16")
elif estimated_elements > 50000:
    print("⚠️  Large model - may take 15-30 minutes")
```

---

### **Step 3: Test with Small Model**

**Use test models:**
```bash
# These should work quickly
python voxelize_local.py 16  # Use lower resolution
python voxel_to_fea_complete.py
```

**If small models work but yours doesn't:**
- Your model is too large
- Reduce resolution

**If small models also crash:**
- Different issue (CalculiX config, Python issue, etc.)

---

## 🔧 Solutions Based on Issue

### **If Model Too Large:**

**Option A: Reduce Resolution**
```bash
# Re-voxelize with lower resolution
python voxelize_local.py 16  # Instead of 32
python voxel_to_fea_complete.py
```

**Option B: Skip FEA Temporarily**
- Edit `simple_ply_processor.py`
- Comment out FEA call
- Continue with rest of pipeline

**Option C: Process in Parts**
- Split PLY file into smaller parts
- Process each separately

---

### **If CalculiX Crashes:**

**Check CalculiX Output:**
```bash
cd voxel_out\{model}\fea_analysis
type model.dat | findstr /i "error warning"
```

**Common Fixes:**
- **Singular matrix** → Check mesh quality
- **Memory error** → Reduce resolution
- **File error** → Check permissions

---

### **If Python Subprocess Crashes:**

**The fixes we applied should help:**
- ✅ Better error handling
- ✅ Timeout protection
- ✅ Memory error catching
- ✅ Exception wrapping

**If still crashing:**
- Run FEA manually (outside frontend)
- Use `test_fea_safely.py` to diagnose

---

## 🎯 Quick Test

**Run this to test everything:**
```bash
# 1. Test CalculiX
ccx --version  # ✅ You already did this - works!

# 2. Test with small model
python test_fea_safely.py test_cube

# 3. Check your model size
python -c "import numpy as np; import os; data=np.load('voxel_out/{your_model}/voxels_filled_indices_colors.npz'); print(f'Voxels: {len(data[\"indices\"]):,}')"
```

---

## 📊 What to Look For

**In the frontend log, check:**

1. **Before crash:**
   - "Model size: X nodes, Y elements" → If >50k, too large
   - "Starting CalculiX analysis" → Gets this far
   - "CalculiX found" → Detection works

2. **Error messages:**
   - "Out of memory" → Reduce resolution
   - "Timeout" → Model too large
   - "Failed with return code" → Check CalculiX output
   - No message → Python crash (use test script)

3. **Timing:**
   - Crashes immediately → Setup issue
   - Crashes after time → Timeout/Memory
   - Crashes during solve → CalculiX issue

---

## 💡 Most Likely Fix

**Since CalculiX 2.22 works, the issue is probably:**

1. **Model too large** (80% chance)
   - Fix: `python voxelize_local.py 16`

2. **Memory issue** (15% chance)
   - Fix: Close apps, reduce resolution

3. **Subprocess wrapper** (5% chance)
   - Fix: Use manual FEA or test script

---

## 🆘 Next Steps

1. **Check model size** (run the Python code above)
2. **Test with small model** (`python test_fea_safely.py test_cube`)
3. **Run FEA manually** (bypass frontend to isolate issue)
4. **Share results** - I can help diagnose further

**CalculiX 2.22 is fine - the crash is from something else!** 🎯


