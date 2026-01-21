# Setup Guide - CAD with AI Frontend

##  Prerequisites

Before running the frontend, you need to set up the Python environment and install dependencies.

---

##  Step-by-Step Setup

### **Step 1: Create Virtual Environment**

Open PowerShell or Command Prompt in the project directory and run:

```bash
# Create virtual environment
python -m venv .venv
```

**What this does**: Creates a `.venv` folder with an isolated Python environment

---

### **Step 2: Activate Virtual Environment**

**Windows (PowerShell)**:
```powershell
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```cmd
.venv\Scripts\activate.bat
```

**macOS/Linux**:
```bash
source .venv/bin/activate
```

**What this does**: Activates the virtual environment (you should see `(.venv)` in your prompt)

**Note**: If you get an execution policy error on Windows PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### **Step 3: Install Dependencies**

With the virtual environment activated, run:

```bash
pip install -r requirements.txt
```

**What this installs**:
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `trimesh` - 3D mesh processing
- `rtree` - Spatial indexing
- `pandas` - Data analysis
- `matplotlib` - Visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning
- `joblib` - Model persistence

**Expected output**: Should see packages being installed successfully

---

### **Step 4: Verify Installation**

Test that numpy is installed correctly:

```bash
python -c "import numpy; print('NumPy version:', numpy.__version__)"
```

Should output something like: `NumPy version: 1.24.3`

---

### **Step 5: Run the Frontend**

Now you can run the GUI:

```bash
python simple_ply_processor.py
```

**Note**: You can run this command with or without the virtual environment activated, but the frontend will automatically use `.venv\Scripts\python.exe` if it exists.

---

## 🔍 Quick Setup Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created (`.venv` folder exists)
- [ ] Virtual environment activated (see `(.venv)` in prompt)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] NumPy verified (`python -c "import numpy"` works)
- [ ] Frontend runs (`python simple_ply_processor.py`)

---

##  Troubleshooting

### **Issue: "No module named 'numpy'"**

**Solution**: Make sure you:
1. Created the virtual environment: `python -m venv .venv`
2. Activated it: `.venv\Scripts\activate`
3. Installed dependencies: `pip install -r requirements.txt`

### **Issue: "Python not found"**

**Solution**: 
- Make sure Python is installed and in PATH
- Try `python3` instead of `python` on some systems
- Check installation: `python --version`

### **Issue: "Execution Policy" error (Windows PowerShell)**

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Issue: Frontend can't find virtual environment**

**Solution**: 
- Make sure `.venv` folder exists in the project root
- The frontend looks for `.venv\Scripts\python.exe` (Windows)
- If using system Python, install numpy globally: `pip install numpy`

---

##  Complete Setup Commands (Copy-Paste Ready)

**Windows PowerShell**:
```powershell
# Navigate to project directory
cd C:\Users\bayir\OneDrive\Masaüstü\cad-with-ai-

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Verify numpy
python -c "import numpy; print('OK')"

# Run frontend
python simple_ply_processor.py
```

**Windows Command Prompt**:
```cmd
cd C:\Users\bayir\OneDrive\Masaüstü\cad-with-ai-
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python -c "import numpy; print('OK')"
python simple_ply_processor.py
```

**macOS/Linux**:
```bash
cd /path/to/cad-with-ai-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import numpy; print('OK')"
python simple_ply_processor.py
```

---

##  Verification Steps

After setup, the frontend should:

1. **Start without errors** - Window opens successfully
2. **Show environment check** - Log shows "OK Virtual environment detected"
3. **Load files** - Can select PLY files
4. **Run pipeline** - Can execute voxelization and AI analysis

If you see "OK Virtual environment detected" in the log, you're all set! 🎉

---

## 🔄 Re-running Setup

If you need to reinstall dependencies:

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # macOS/Linux

# Upgrade pip first
pip install --upgrade pip

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

---

##  Additional Notes

- **Virtual environment is optional but recommended** - Keeps dependencies isolated
- **Frontend auto-detects venv** - Will use `.venv\Scripts\python.exe` if it exists
- **System Python works too** - But you need to install numpy globally
- **Tkinter comes with Python** - No need to install separately (usually)

---

##  Next Steps

Once setup is complete:

1. **Add PLY files** to the `input/` folder (or use drag & drop in GUI)
2. **Run the frontend**: `python simple_ply_processor.py`
3. **Select a PLY file** in the GUI
4. **Click "Run AI Pipeline"** to start processing
5. **Select parts** and click "Run AI on Selected Parts" for recommendations

Happy optimizing! 🚀

