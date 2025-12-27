# 🚀 COMMANDS TO RUN

## Quick Start Guide for Heart Disease Code

---

## 📍 Step 1: Navigate to the Structured Folder

```powershell
cd "c:\Users\priyansh.agrawal\OneDrive - Wabtec Corporation\FT Work\DnA Team\3-3-25 Backup\work\assignment1_mlops\heart_disease_code"
```

---

## 📦 Step 2: Create Virtual Environment (Optional but Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 📥 Step 3: Install Dependencies

```powershell
pip install -r requirements.txt
```

**This installs:**
- pandas, numpy, scikit-learn (for ML)
- matplotlib, seaborn (for visualization)
- mlflow (for experiment tracking)
- pytest, pytest-cov (for testing)
- flake8, pylint (for linting)

---

## ✅ Step 4: Run Unit Tests

### Run All Tests
```powershell
pytest
```

### Run with Verbose Output
```powershell
pytest -v
```

### Run with Coverage Report
```powershell
pytest --cov=src --cov-report=html
```

### Run Specific Test File
```powershell
pytest tests/test_preprocessing.py -v
pytest tests/test_model.py -v
```

### Run Specific Test
```powershell
pytest tests/test_preprocessing.py::test_clean_missing_values -v
```

---

## 🔍 Step 5: Check Code Quality (Linting)

### Run Flake8
```powershell
flake8 src tests
```

### Run Flake8 (Ignoring Whitespace Warnings)
```powershell
flake8 src tests --extend-ignore=W293,W291
```

### Run Pylint
```powershell
pylint src tests
```

---

## 📊 Step 6: View Coverage Report

### Generate HTML Coverage Report
```powershell
pytest --cov=src --cov-report=html
```

### Open Coverage Report in Browser
```powershell
start htmlcov/index.html
```

---

## 🧪 Step 7: Import and Use Modules (Optional)

### Create a Test Script
Create a file `test_import.py`:

```python
from src.preprocessing import load_data, clean_missing_values
from src.training import train_logistic_regression, evaluate_model

# Example usage
print("✅ Modules imported successfully!")
print("Available preprocessing functions:", dir(src.preprocessing))
print("Available training functions:", dir(src.training))
```

### Run It
```powershell
python test_import.py
```

---

## 🔧 Step 8: Run All Quality Checks Together

### Single Command for Everything
```powershell
# Run linting, tests, and coverage
flake8 src tests --extend-ignore=W293,W291; pytest --cov=src --cov-report=html --cov-report=term
```

---

## 📝 Step 9: Use in Jupyter Notebook (Optional)

### Option A: Use Original Notebook (Unchanged)
```powershell
# Just open the original notebook in parent directory
jupyter notebook ../assignment1.ipynb
```

### Option B: Import Modules in New Notebook
```python
import sys
sys.path.append('.')  # Add current directory to path

from src.preprocessing import load_data, clean_missing_values
from src.training import train_logistic_regression, evaluate_model

# Use the functions
df = load_data('../heart_disease_dataset/processed.cleveland.data')
df_clean = clean_missing_values(df)
```

---

## 🚀 Step 10: CI/CD Pipeline (GitHub Actions)

### If You Have a GitHub Repo:

1. **Initialize Git:**
```powershell
git init
git add .
git commit -m "Add MLOps pipeline with tests and CI/CD"
```

2. **Push to GitHub:**
```powershell
git remote add origin https://github.com/YOUR_USERNAME/heart-disease-mlops.git
git branch -M main
git push -u origin main
```

3. **GitHub Actions Will Automatically:**
   - ✅ Run linting (flake8, pylint)
   - ✅ Run all tests
   - ✅ Generate coverage reports
   - ✅ Upload artifacts

---

## 🧹 Step 11: Clean Up (Optional)

### Remove Test Cache
```powershell
Remove-Item -Recurse -Force .pytest_cache
Remove-Item -Recurse -Force __pycache__
Remove-Item -Recurse -Force src\__pycache__
Remove-Item -Recurse -Force tests\__pycache__
```

### Remove Coverage Files
```powershell
Remove-Item -Force .coverage
Remove-Item -Recurse -Force htmlcov
```

---

## 📋 Common Commands Summary

| Task | Command |
|------|---------|
| **Run all tests** | `pytest` |
| **Tests with coverage** | `pytest --cov=src --cov-report=html` |
| **Verbose tests** | `pytest -v` |
| **Single test file** | `pytest tests/test_preprocessing.py` |
| **Lint code** | `flake8 src tests --extend-ignore=W293,W291` |
| **View coverage** | `start htmlcov/index.html` |
| **Clean cache** | `Remove-Item -Recurse -Force .pytest_cache` |

---

## 🎯 Expected Outputs

### Successful Test Run:
```
======================== test session starts =========================
collected 12 items

tests/test_model.py ......                                       [ 50%]
tests/test_preprocessing.py ......                               [100%]

=================== 12 passed in 2.99s ===================
```

### Successful Linting:
```
(No output means success - code is clean!)
```

### Coverage Report:
```
Name                   Stmts   Miss  Cover
------------------------------------------
src/__init__.py            1      0   100%
src/preprocessing.py      36      3    92%
src/training.py           29      0   100%
------------------------------------------
TOTAL                     66      3    95%
```

---

## ⚠️ Troubleshooting

### If Tests Fail:
```powershell
# Run with detailed error output
pytest -v --tb=long
```

### If Imports Don't Work:
```powershell
# Make sure you're in the correct directory
pwd

# Should show: ...\heart_disease_code
```

### If Coverage is Missing:
```powershell
# Install coverage tool
pip install pytest-cov
```

---

## 🎓 Learning Path

### Beginner:
1. Run `pytest` to see tests pass ✅
2. Run `pytest --cov=src` to see coverage ✅
3. Look at test files to understand structure ✅

### Intermediate:
1. Modify a function in `src/preprocessing.py`
2. Run tests to see if anything breaks
3. Add a new test for your modification

### Advanced:
1. Write a new preprocessing function
2. Write tests for it
3. Achieve 100% coverage
4. Push to GitHub and watch CI/CD run

---

## 📞 Quick Help

### Check Python Version:
```powershell
python --version
# Should be 3.8 or higher
```

### Check Installed Packages:
```powershell
pip list
```

### Check Current Directory:
```powershell
pwd
```

### List Files:
```powershell
Get-ChildItem
```

---

## ✅ Verification Checklist

Run these commands in order to verify everything works:

```powershell
# 1. Navigate to folder
cd heart_disease_code

# 2. Activate environment (if using)
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests
pytest -v

# 5. Check coverage
pytest --cov=src --cov-report=term

# 6. Lint code
flake8 src tests --extend-ignore=W293,W291

# ✅ If all pass, you're ready!
```

---

## 🎉 Success Criteria

You're good to go if you see:
- ✅ 12/12 tests passing
- ✅ 95%+ code coverage
- ✅ No linting errors
- ✅ All modules importable

**Congratulations! Your MLOps pipeline is production-ready! 🚀**
