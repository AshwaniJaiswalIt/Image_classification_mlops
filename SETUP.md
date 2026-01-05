# ⚠️ SETUP FIRST - Run This on New Laptop

## Step-by-Step Setup

### 1. Create Virtual Environment
```powershell
python -m venv venv
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install ALL Dependencies (REQUIRED!)
```powershell
pip install -r requirements.txt
```

**This installs:**
- pytest (for testing)
- flake8 (for linting)  
- jupyter (for notebooks)
- All ML libraries

### 4. Verify Installation
```powershell
pytest --version
# Should show: pytest 7.4.3

flake8 --version
# Should show: 6.1.0

jupyter --version
# Should show version info
```

---

## ❌ Common Error

**Error:** `pytest : The term 'pytest' is not recognized`

**Cause:** You didn't run `pip install -r requirements.txt`

**Solution:**
```powershell
# Make sure venv is activated (you see (venv) in prompt)
pip install -r requirements.txt
```

---

## ✅ After Setup

Now you can run:
- `pytest -v` - Run tests
- `jupyter notebook assignment1.ipynb` - Run notebook
- `flake8 src tests` - Run linting

See README.md for all commands.
