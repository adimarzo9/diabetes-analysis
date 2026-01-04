"""
This script orchestrates the entire machine learning pipeline. 
It performs the following steps to be able to run the entire code.
1. Load and explore the dataset
2. Preprocesses the data
3. Trains four different models (Gaussian Naïve Bayes, Logistic Regression, Random Forest, XGBoost)
4. Evaluates all models with cross-validation
5. Compares models and identifies the best performing model
6. Saves all results and visualizations in the results file

The project predicts whether a patient has diabetes (0 = no, 1 = yes) using
demographic and clinical features.
"""

import os
import sys
import time
try:
    import pandas as pd
except ImportError as e:
    raise RuntimeError(
        "pandas is required by main.py. Install it with: python -m pip install pandas"
    ) from e

# Adds src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import load_data, explore_data, preprocess_data, prepare_data, apply_smote
from src.models import get_models
from src.evaluation import evaluate_model, compare_models

import warnings
warnings.filterwarnings("ignore")

def main():
    """
    Main function to run the complete diabetes analysis pipeline.
    Executes all steps: data loading, preprocessing, model training,
    evaluation, and comparison.
    """
    # Starts recording the execution time
    start_time = time.time()
    
    # Prints the project header
    print("\n" + "="*70)
    print(" " * 15 + "DIABETES PREDICTION PROJECT")
    print(" " * 10 + "Machine Learning Model Comparison")
    print("="*70)
    print("\nThis project compares four machine learning models to predict")
    print("whether a patient has diabetes based on clinical features.")
    print("\nModels: Naive Bayes, Logistic Regression, Random Forest, XGBoost")
    print("="*70)
    
    # Creates results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n📁 Results will be saved to: {results_dir}\n")
    
    # ========================================================================
    # STEP 1: Load the data
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 1: DATA LOADING")
    print("█"*70)
    data = load_data()
    
    # ========================================================================
    # STEP 2: Exploratory Data Analysis (EDA)
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("█"*70)
    explore_data(data)
    
    # ========================================================================
    # STEP 3: Preprocess the data
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 3: DATA PREPROCESSING")
    print("█"*70)
    data_processed = preprocess_data(data)
    
    # ========================================================================
    # STEP 4: Prepare data for modeling (split into X, y and train/test)
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 4: DATA PREPARATION FOR MODELING")
    print("█"*70)
    X_train, X_test, y_train, y_test = prepare_data(data_processed, 
                                                     test_size=0.2, 
                                                     random_state=42)
    
    # ========================================================================
    # STEP 4.5: Apply SMOTE to balance training set (ONLY training set!)
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 4.5: APPLYING SMOTE TO BALANCE TRAINING SET")
    print("█"*70)
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state=42)
    
    # Converts back to DataFrame/Series for compatibility (preserves column names)
    if isinstance(X_train, pd.DataFrame):
        X_train_resampled = pd.DataFrame(
            X_train_resampled, 
            columns=X_train.columns,
            index=range(len(X_train_resampled))
        )
    y_train_resampled = pd.Series(y_train_resampled, name=y_train.name)
    
    # Uses resampled data for training
    X_train = X_train_resampled
    y_train = y_train_resampled
    
    # ========================================================================
    # STEP 5: Initialize models
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 5: MODEL INITIALIZATION")
    print("█"*70)
    models = get_models()
    
    # ========================================================================
    # STEP 6: Train and evaluate each model
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 6: MODEL TRAINING AND EVALUATION")
    print("█"*70)
    print("\nTraining and evaluating all models...")
    print("This may take a few minutes depending on dataset size.\n")
    
    results_dict = {}
    trained_models = {}
    models_predictions = {}  # Store predictions for visualizations
    
    # Evaluates each model
    for idx, (model_name, model) in enumerate(models.items(), 1):
        print(f"\n[{idx}/{len(models)}] Processing: {model_name}")
        
        metrics, trained_model, y_pred_proba, cv_results = evaluate_model(
            model, model_name,
            X_train, y_train,
            X_test, y_test,
            results_dir,
            perform_cv=True  # Perform cross-validation
        )
        
        # Stores results
        results_dict[model_name] = metrics
        trained_models[model_name] = trained_model
        models_predictions[model_name] = y_pred_proba
    
    # ========================================================================
    # STEP 7: Compare all models and generate final report
    # ========================================================================
    print("\n" + "█"*70)
    print("STEP 7: MODEL COMPARISON AND FINAL REPORT")
    print("█"*70)
    compare_models(results_dict, models_predictions, trained_models, 
                   X_train, y_test, results_dir)
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print(" " * 20 + "ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\n⏱️  Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"📁 All results saved to: {results_dir}")
    
    print("\n📊 Generated Files:")
    print("-" * 70)
    print("\n  [Comparison Files]")
    print("    ✓ model_comparison.csv - Numerical comparison table")
    print("    ✓ model_comparison.png - Bar chart comparison")
    print("    ✓ metrics_heatmap.png - Heatmap visualization")
    print("    ✓ metrics_radar_chart.png - Radar chart comparison")
    print("    ✓ roc_curves_comparison.png - ROC curves for all models")
    print("    ✓ precision_recall_curves_comparison.png - PR curves")
    
    print("\n  [Model-Specific Files]")
    for model_name in models.keys():
        print(f"\n    {model_name}:")
        print(f"      ✓ confusion_matrix_{model_name.replace(' ', '_')}.png")
        print(f"      ✓ classification_report_{model_name.replace(' ', '_')}.txt")
        if model_name in ['Random Forest', 'XGBoost']:
            print(f"      ✓ feature_importance_{model_name.replace(' ', '_')}.png")
    
    # Finds and displays the best performing model
    comparison_df = pd.DataFrame(results_dict).T
    best_model = comparison_df['f1_score'].idxmax()
    best_metrics = comparison_df.loc[best_model]
    
    print("\n" + "="*70)
    print(" " * 15 + "FINAL RECOMMENDATION")
    print("="*70)
    print(f"\n🏆 Best Model for Diabetes Prediction: {best_model}")
    print(f"\n   Performance Metrics:")
    print(f"     • Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"     • Precision: {best_metrics['precision']:.4f}")
    print(f"     • Recall:    {best_metrics['recall']:.4f}")
    print(f"     • F1-Score:  {best_metrics['f1_score']:.4f}")
    print(f"     • ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    
    print("\n" + "="*70)
    print("Thank you for using the Diabetes Prediction System!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()