# Diabetes Prediction : Model Comparison

## Research Question :
Which model best predicts diabetes : Logistic Regression, Random Forest or XGBoost ?

# Setup :
Make Sure you are in the right directory : 

```bash
cd capstone_project/diabetes_analysis
``

### Using Conda :
            
### Create environment from environment.yml

```bash
conda env create -f environment.yml
```

### Activate the environment

```bash
conda activate diabetes_analysis
```

### Install XGBoost and imbalanced-learn (not included in environment.yml)

```bash
pip install xgboost imbalanced-learn
```

### Using pip :

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost 
```

## Usage

To run the code : 

```bash
python main.py
```

Expected output:

Comparison between four machine learning models; Gaussian Naïve Bayes (as a baseline), Logistic Regression, Random Forest and XGBoost models. The program trains, tests, compares the models and presents which one is the best performing for the applied dataset.

## Project Structure :

```bash
diabetes_analysis/
    ├── data/
    │   └── raw/               # Original data (or instructions to download)
    ├── results/              # Output figures and metrics
    ├── src/                  # Source code modules
    │   ├── __init__.py
    │   ├── data_loader.py     # Data loading and preprocessing
    │   ├── evaluation.py      # Evaluation and visualization
    │   └── models.py          # Model definitions
    ├── AI_USAGE.md           # AI usage
    ├── environment.yml       # Conda dependencies
    ├── main.py               # Entry point - THIS MUST WORK
    ├── PROPOSAL.md           # Your project proposal
    └── README.md             # Setup and usage instructions
```

## Results :

- All the results and visualizations are saved in the results file of the project.

```bash
======================================================================
COMPREHENSIVE MODEL COMPARISON
======================================================================

[1] Performance Metrics Comparison:
----------------------------------------------------------------------
                     accuracy  precision  recall  f1_score  roc_auc
Naive Bayes            0.8450     0.2363  0.6324    0.3440   0.8253
Logistic Regression    0.8450     0.2363  0.6324    0.3440   0.8259
Random Forest          0.9112     0.3333  0.3824    0.3562   0.8434
XGBoost                0.9216     0.3469  0.2500    0.2906   0.8526

[2] Best Model by Metric:
----------------------------------------------------------------------
Metric               Best Model                Score          
----------------------------------------------------------------------
Accuracy             XGBoost                   0.9216
Precision            XGBoost                   0.3469
Recall               Naive Bayes               0.6324
F1-Score             Random Forest             0.3562
ROC-AUC              XGBoost                   0.8526

[3] Model Selection for Imbalanced Dataset:
----------------------------------------------------------------------
  Note: For imbalanced datasets, we prioritize:
    • Recall: Ability to identify diabetes cases (minimize false negatives)
    • F1-Score: Balance between precision and recall

  Best F1-Score: Random Forest (0.3562)
  Best Recall:    Naive Bayes (0.6324)

[4] Overall Best Model (Selected by F1-Score):
----------------------------------------------------------------------
  🏆 WINNER: Random Forest
  Best F1-Score: 0.3562
  (F1-Score balances precision and recall, ideal for imbalanced datasets)

  Complete Performance:
    Accuracy:  0.9112
    Precision: 0.3333
    Recall:    0.3824 ⭐ (Key metric for minority class)
    F1-Score:  0.3562 ⭐ (Key metric for imbalanced data)
    ROC-AUC:   0.8434

[5] Model Rankings (Prioritizing Recall & F1-Score):
----------------------------------------------------------------------
Rank     Model                     F1-Score     Recall       ROC-AUC     
----------------------------------------------------------------------
🏆        Random Forest             0.3562    0.3824    0.8434
2.       Logistic Regression       0.3440    0.6324    0.8259
3.       Naive Bayes               0.3440    0.6324    0.8253
4.       XGBoost                   0.2906    0.2500    0.8526

======================================================================
                    ANALYSIS COMPLETE!
======================================================================

⏱️  Total execution time: 33.63 seconds (0.56 minutes)
📁 All results saved to: /files/capstone_project/diabetes_analysis/results

======================================================================
               FINAL RECOMMENDATION
======================================================================

🏆 Best Model for Diabetes Prediction: Random Forest

   Performance Metrics:
     • Accuracy:  0.9112
     • Precision: 0.3333
     • Recall:    0.3824
     • F1-Score:  0.3562
     • ROC-AUC:   0.8434
```

## Requirements :

- Python 3.11
- scikit-learn, pandas, matplotlib, seaborn