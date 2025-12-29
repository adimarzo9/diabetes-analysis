"""
This module defines the machine learning models used in this project.
It creates and returns all four machine learning models to compare:
1. Gaussian Naïve Bayes (as a baseline)
2. Logistic Regression (with StandardScaler pipeline)
3. Random Forest
4. XGBoost

All models use random_state=42 for reproducibility.
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Imports xgboost with helpful error message if not available
try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "XGBoost is not installed. Please install it using: pip install xgboost\n"
        "Or add it to your conda environment: conda install -c conda-forge xgboost"
    )


def get_models():
    """
    Creates and returns a dictionary of all four machine learning models.
    
    Models included:
    - Gaussian Naive Bayes: Simple probabilistic baseline model
    - Logistic Regression: Linear model with StandardScaler in pipeline (hyperparameter)
    - Random Forest: Ensemble of decision trees with default hyperparameters
    - XGBoost: Gradient boosting model with default hyperparameters
    
    Returns:
    --------
    models : dict
        Dictionary with model names as keys and model objects as values
    """
    print("\n" + "="*70)
    print("INITIALIZING MACHINE LEARNING MODELS")
    print("="*70)
    
    models = {
        # 1. Gaussian Naive Bayes - Baseline model
        'Naive Bayes': GaussianNB(),
        
        # 2. Logistic Regression with StandardScaler in a pipeline
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=42, 
                max_iter=1000,
                solver='lbfgs'  # Good default solver
            ))
        ]),
        
        # 3. Random Forest Model
        # Using default hyperparameters
        'Random Forest': RandomForestClassifier(
            n_estimators=100,       # Number of trees in the forest
            max_depth=10,           # Maximum depth of trees
            min_samples_split=5,    # Minimum samples to split a node
            min_samples_leaf=2,     # Minimum samples in a leaf
            random_state=42,
            n_jobs=-1               # Use all available CPU cores
        ),
        
        # 4. XGBoost Model
        # Using default hyperparameters
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,       # Number of boosting rounds
            max_depth=6,            # Maximum tree depth
            learning_rate=0.1,      # Step size shrinkage
            subsample=0.8,          # Subsample ratio of training instances
            colsample_bytree=0.8,   # Subsample ratio of columns
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            n_jobs=-1               # Use all available CPU cores
        )
    }
    
    # Prints models'information
    print("\n[1] Model Summary:")
    print("-" * 70)
    print(f"{'Model Name':<25} {'Type':<30} {'Description'}")
    print("-" * 70)
    print(f"{'Naive Bayes':<25} {'Probabilistic':<30} {'Baseline model - simple and fast'}")
    print(f"{'Logistic Regression':<25} {'Linear':<30} {'With StandardScaler pipeline'}")
    print(f"{'Random Forest':<25} {'Ensemble':<30} {'100 trees, max_depth=10'}")
    print(f"{'XGBoost':<25} {'Gradient Boosting':<30} {'100 estimators, max_depth=6'}")
    
    print("\n[2] Model Configuration:")
    print("-" * 70)
    for name, model in models.items():
        print(f"\n  {name}:")
        if isinstance(model, Pipeline):
            print(f"    - Pipeline with {len(model.named_steps)} steps")
            for step_name, step_model in model.named_steps.items():
                print(f"      Step '{step_name}': {type(step_model).__name__}")
        else:
            print(f"    - Type: {type(model).__name__}")
            # Prints key hyperparameters
            if hasattr(model, 'n_estimators'):
                print(f"    - n_estimators: {model.n_estimators}")
            if hasattr(model, 'max_depth'):
                print(f"    - max_depth: {model.max_depth}")
            if hasattr(model, 'learning_rate'):
                print(f"    - learning_rate: {model.learning_rate}")
    
    print("\n[3] Reproducibility:")
    print("-" * 70)
    print("  ✓ All models use random_state=42 for reproducibility")
    print("  ✓ Results will be consistent across runs")
    
    print("\n" + "="*70)
    
    return models