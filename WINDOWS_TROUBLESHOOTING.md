# Windows Troubleshooting Guide

This guide helps Windows users resolve common installation and runtime issues.

---

## 🔴 Error: "DLL initialization routine failed" (PyTorch)

**Error Message**:
```
OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed.
Error loading "...\torch\lib\c10.dll"
```

### Root Causes:
1. Python version too new (3.13, 3.14)
2. Missing Visual C++ Redistributables
3. Corrupted PyTorch installation
4. Incompatible CUDA/GPU drivers

---

## ✅ Solution 1: Downgrade Python (RECOMMENDED)

PyTorch currently supports **Python 3.8 - 3.12** only.

### Steps:

1. **Uninstall Python 3.14**
   - Go to Settings → Apps → Python 3.14 → Uninstall

2. **Download Python 3.11.x**
   - Visit: https://www.python.org/downloads/
   - Download: Python 3.11.8 (or latest 3.11.x)
   - **Important**: Check "Add Python to PATH" during installation

3. **Verify Installation**
   ```bash
   python --version
   # Should show: Python 3.11.x
   ```

4. **Recreate Virtual Environment**
   ```bash
   cd AI-fast-Project
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Test**
   ```bash
   python -c "import torch; print('PyTorch OK!')"
   ```

---

## ✅ Solution 2: Install Visual C++ Redistributables

### Download and Install:

1. **Download VC++ Redistributable**
   - Visit: https://aka.ms/vs/17/release/vc_redist.x64.exe
   - Or search: "Microsoft Visual C++ Redistributable latest"

2. **Install**
   - Run the downloaded .exe file
   - Accept license and install
   - Restart your computer

3. **Verify**
   - Check in: Control Panel → Programs → Installed Programs
   - Should see: "Microsoft Visual C++ 2015-2022 Redistributable"

4. **Test PyTorch Again**
   ```bash
   python -c "import torch; print('PyTorch OK!')"
   ```

---

## ✅ Solution 3: Reinstall PyTorch (CPU Version)

If you don't have an NVIDIA GPU or just want to test:

### Steps:

1. **Activate Virtual Environment**
   ```bash
   venv\Scripts\activate
   ```

2. **Uninstall Current PyTorch**
   ```bash
   pip uninstall torch torchvision torchaudio -y
   ```

3. **Install CPU-Only PyTorch**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

4. **Test**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__} OK!')"
   ```

---

## ✅ Solution 4: Reinstall PyTorch (GPU Version)

If you have an NVIDIA GPU:

### Prerequisites:

1. **Check GPU**
   ```bash
   # Open Command Prompt and run:
   nvidia-smi
   ```
   - If command not found, you don't have NVIDIA GPU or drivers

2. **Install CUDA Toolkit** (if needed)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose: Windows → x86_64 → Your Windows version
   - Install CUDA 11.8 or 12.1

### Installation:

```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio -y

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Test:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## ✅ Solution 5: Use Demo Mode (No PyTorch)

If nothing else works, use the demo version:

```bash
streamlit run app_demo_only.py
```

**Features**:
- ✅ Shows the full UI
- ✅ Mock predictions for demonstration
- ✅ No PyTorch required
- ✅ Works on any Python version
- ❌ Not the actual AI model

**Use Case**: Perfect for presentation if you just need to show the interface.

---

## 🔧 Solution 6: Complete Clean Reinstall

Nuclear option - start fresh:

### Steps:

1. **Delete Everything**
   ```bash
   # Deactivate venv
   deactivate

   # Delete venv folder
   rmdir /s venv

   # Delete pip cache
   pip cache purge
   ```

2. **Verify Python Version**
   ```bash
   python --version
   # Must be 3.8 - 3.12
   ```

3. **Create Fresh Environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   python -m pip install --upgrade pip
   ```

4. **Install Dependencies One by One**
   ```bash
   # Install PyTorch first
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # Test PyTorch
   python -c "import torch; print('PyTorch OK')"

   # Install rest
   pip install transformers datasets accelerate peft bitsandbytes
   pip install pandas numpy scikit-learn
   pip install streamlit
   pip install sentencepiece protobuf
   ```

5. **Test App**
   ```bash
   streamlit run app_demo_only.py
   ```

---

## 🐛 Other Common Windows Issues

### Issue: "streamlit: command not found"

**Solution**:
```bash
# Make sure venv is activated
venv\Scripts\activate

# Verify streamlit is installed
pip list | findstr streamlit

# If not installed:
pip install streamlit

# Run with python -m
python -m streamlit run app.py
```

### Issue: "Permission denied"

**Solution**:
1. Run Command Prompt as Administrator
2. Or disable antivirus temporarily
3. Or exclude Python folder from antivirus

### Issue: Long path errors

**Solution**:
1. Enable long paths in Windows:
   - Run as Admin: `regedit`
   - Navigate to: `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`
2. Or move project to shorter path: `C:\Projects\AI-fast-Project`

### Issue: "No module named 'torch'"

**Solution**:
```bash
# Check which Python is running
python --version
which python  # On Git Bash
where python  # On CMD

# Make sure venv is activated
venv\Scripts\activate

# Reinstall torch
pip install torch
```

---

## 📊 Quick Diagnostic

Run this to diagnose your setup:

```bash
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print(f'PyTorch Error: {e}')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('Transformers: Not installed')

try:
    import streamlit
    print(f'Streamlit: {streamlit.__version__}')
except ImportError:
    print('Streamlit: Not installed')
"
```

**Expected Output** (minimum working):
```
Python: 3.11.x
PyTorch: 2.x.x
CUDA available: True/False
Transformers: 4.x.x
Streamlit: 1.x.x
```

---

## 🎯 Recommended Setup for Windows

### Ideal Configuration:

```
✅ Python 3.11.8 (from python.org)
✅ Visual C++ Redistributable 2015-2022
✅ PyTorch 2.1+ (CPU or CUDA 11.8)
✅ Virtual environment (venv)
✅ Updated pip (23.0+)
```

### Installation Commands:

```bash
# 1. Create project directory
mkdir C:\Projects
cd C:\Projects
git clone <your-repo> AI-fast-Project
cd AI-fast-Project

# 2. Create venv
python -m venv venv
venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch (CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Install requirements
pip install -r requirements.txt

# 6. Test
python -c "import torch; print('OK')"
streamlit run app_demo_only.py
```

---

## 🆘 Still Not Working?

### Option 1: Use Google Colab

Upload your project to Google Colab:
- Free GPU access
- All libraries pre-installed
- No Windows issues

### Option 2: Use WSL2 (Windows Subsystem for Linux)

Install Ubuntu on Windows:
```bash
wsl --install
# Restart computer
# Open Ubuntu terminal
# Follow Linux installation instructions
```

### Option 3: Use Docker

```bash
# Install Docker Desktop for Windows
# Run project in container
docker run -it -p 8501:8501 -v ${PWD}:/app python:3.11
cd /app
pip install -r requirements.txt
streamlit run app.py
```

### Option 4: Use Demo Mode

```bash
# Just use the demo version
streamlit run app_demo_only.py
```

---

## 📞 Getting Help

If all else fails:

1. **Check Python version**: Must be 3.8-3.12
2. **Install VC++ Redistributables**
3. **Use demo mode for presentation**: `app_demo_only.py`
4. **Ask for help with**:
   - Your Python version
   - Error messages (full text)
   - Output of diagnostic script above

---

## ✅ Success Checklist

Before running the app:

- [ ] Python 3.11 or 3.12 installed
- [ ] Visual C++ Redistributable installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed and imports successfully
- [ ] All requirements.txt packages installed
- [ ] Diagnostic script shows no errors

Then:
```bash
streamlit run app.py  # Full version
# OR
streamlit run app_demo_only.py  # Demo version
```

Good luck! 🚀
