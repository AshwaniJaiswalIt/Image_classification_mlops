#  SETUP FIRST - Run This on New Laptop

## Step-by-Step Setup

### 1. Create Virtual Environment
`powershell
python -m venv venv
`

### 2. Activate Virtual Environment
`powershell
.\venv\Scripts\Activate.ps1
`

### 3. Install ALL Dependencies (REQUIRED!)
`powershell
pip install -r requirements.txt
`

**This installs:**
- pytest (for testing)
- flake8 (for linting)  
- tensorflow, pillow, opencv, dvc
- All other ML libraries (pandas, numpy, etc.)

### 4. Verify Installation
`powershell
pytest --version
# Should show: pytest 7.4.3

flake8 --version
# Should show: 6.1.0

python -c "import matplotlib; print('matplotlib OK')"
# Should show: matplotlib OK
`

### 5. Pull dataset using DVC
The cats vs dogs images are tracked with DVC and are not committed to Git.
After installing dvc (already added to requirements), run:

`powershell
cd Image_classification_mlops
pip install dvc
# fetch the data
dvc pull cats_dogs_dataset
`

The folder cats_dogs_dataset/ will be populated with two subdirectories (cats/ and dogs/).

### 6. Generate Model Files (Required for Docker API)
Run the training pipeline to create the Keras model used by the API:

`powershell
python train_pipeline.py
`

Once training completes, copy the generated model into the API directory:

`powershell
Copy-Item cat_dog_model.h5 api/models/
`

---

##  Common Errors

**Error:** pytest : The term 'pytest' is not recognized  
**Cause:** You didn't run pip install -r requirements.txt

**Error:** No module named 'matplotlib' in notebook  
**Cause:** Notebook using wrong kernel. Follow step 5 above to select venv kernel.

**Error:** COPY models/ models/ fails in Docker build  
**Cause:** Model files don't exist. Run pipeline to generate them (step 6).

**Solution:**
`powershell
# Make sure venv is activated (you see (venv) in prompt)
pip install -r requirements.txt
`

---

##  After Setup

Now you can run:
- pytest -v - Run tests
- python train_pipeline.py - Train the model
- lake8 src tests - Run linting

See README.md for full command list.
