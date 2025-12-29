# Diabetes Analysis Project

## Setup and Usage Instructions

### Prerequisites
- Python 3.11 (as specified in environment.yml)
- Conda (recommended) or pip

### Installation

#### Using Conda (Recommended)
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate diabetes_analysis project

# Install XGBoost and imbalanced-learn (not included in environment.yml)
pip install xgboost imbalanced-learn
```

#### Using pip (Alternative)
If you prefer pip, install the required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

### Project Structure

```
Diabetes_analysis/
├── README.md              # Setup and usage instructions
├── PROPOSAL.md            # Your project proposal
├── environment.yml        # Conda dependencies
├── main.py               # Entry point - THIS MUST WORK
├── src/                  # Source code modules
│   ├── __init__.py
│   ├── data_loader.py    # Data loading and preprocessing
│   ├── models.py         # Model definitions
│   └── evaluation.py     # Evaluation and visualization
├── data/
│   └── raw/              # Original data
│       └── diabetes_analysis.csv
├── results/              # Output figures and metrics
└── notebooks/            # Jupyter notebooks (optional, for exploration)
```

### Running the Project

1. **Ensure you're in the project root directory:**
   ```bash
   cd Diabetes_analysis
   ```

2. **Verify the data file exists:**
   ```bash
   ls data/raw/diabetes_analysis.csv
   ```

3. **Run the main script:**
   ```bash
   python main.py
   ```

### Data

The dataset (`diabetes_analysis.csv`) should be located in `data/raw/` directory.

**Dataset Features:**
- age, gender, pulse_rate, systolic_bp, diastolic_bp, glucose
- height, weight, bmi
- family_diabetes, hypertensive, family_hypertension
- cardiovascular_disease, stroke

### Models

The project compares four machine learning models:
1. **Naive Bayes** - Baseline probabilistic model
2. **Logistic Regression** - Linear model with StandardScaler pipeline
3. **Random Forest** - Ensemble of decision trees
4. **XGBoost** - Gradient boosting model

### Results

All results will be saved in the `results/` directory:
- Model comparison tables (CSV)
- Confusion matrices (PNG)
- ROC curves comparison (PNG)
- Precision-Recall curves (PNG)
- Feature importance plots (PNG)
- Classification reports (TXT)

### Troubleshooting

**Issue: XGBoost not found**
```bash
pip install xgboost
```

**Issue: SMOTE/imbalanced-learn not found**
```bash
pip install imbalanced-learn
```

**Issue: Data file not found**
- Ensure you're running `python main.py` from the project root directory
- Verify `data/raw/diabetes_analysis.csv` exists
- Check the error message for the expected path

**Issue: Import errors**
- Make sure the conda environment is activated: `conda activate diabetes_analysis project`
- Verify all dependencies are installed: `conda list`

### Notes

- All models use `random_state=42` for reproducibility
- The code automatically handles data preprocessing and encoding
- Results are saved automatically to the `results/` directory
- The project uses 80/20 train/test split with stratification
