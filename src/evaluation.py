"""
Evaluation and visualization module.
This module handles model training, prediction, evaluation, cross-validation,
hyperparameter and comprehensive model comparison.
It includes structured terminal outputs and visualization functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def train_model(model, X_train, y_train, model_name="Model"):
    """
    Train a machine learning model with progress output.
    
    Parameters:
    -----------
    model : object
        Model to train (must have .fit() method)
    X_train : pandas.DataFrame or numpy.array
        Training features
    y_train : pandas.Series or numpy.array
        Training target
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    trained_model : object
        Trained model
    """
    print(f"  Training {model_name}...", end=" ", flush=True)
    model.fit(X_train, y_train)
    print("✓ Complete")
    return model


def make_predictions(model, X_test):
    """
    Make predictions using a trained model.
    Returns both class predictions and probability predictions.
    
    Parameters:
    -----------
    model : object
        Trained model (must have .predict() and .predict_proba() methods)
    X_test : pandas.DataFrame or numpy.array
        Test features
    
    Returns:
    --------
    y_pred : numpy.array
        Predicted class labels (0 or 1)
    y_pred_proba : numpy.array
        Predicted probabilities for positive class (diabetes)
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    return y_pred, y_pred_proba


def compute_metrics(y_true, y_pred, y_pred_proba):
    """
    Compute comprehensive evaluation metrics for a model.
    
    Parameters:
    -----------
    y_true : numpy.array or pandas.Series
        True labels
    y_pred : numpy.array
        Predicted labels
    y_pred_proba : numpy.array
        Predicted probabilities
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }
    
    return metrics


def get_confusion_matrix(y_true, y_pred):
    """
    Generate confusion matrix.
    
    Parameters:
    -----------
    y_true : numpy.array or pandas.Series
        True labels
    y_pred : numpy.array
        Predicted labels
    
    Returns:
    --------
    cm : numpy.array
        Confusion matrix as 2x2 array
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm


def plot_confusion_matrix(cm, model_name, save_path=None):
    """
    Plot and save confusion matrix with annotations.
    
    Parameters:
    -----------
    cm : numpy.array
        Confusion matrix
    model_name : str
        Name of the model (for title)
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add text annotations with percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = (cm[i, j] / total) * 100
            plt.text(j + 0.5, i + 0.7, f'({pct:.1f}%)', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ Saved confusion matrix: {os.path.basename(save_path)}")
    else:
        plt.show()
    
    plt.close()


def cross_validate_model(model, X_train, y_train, model_name, cv=5):
    """
    Perform cross-validation on a model and return mean scores.
    
    Parameters:
    -----------
    model : object
        Model to cross-validate
    X_train : pandas.DataFrame or numpy.array
        Training features
    y_train : pandas.Series or numpy.array
        Training target
    model_name : str
        Name of the model
    cv : int, default=5
        Number of cross-validation folds
    
    Returns:
    --------
    cv_results : dict
        Dictionary containing cross-validation results
    """
    print(f"\n  Cross-validating {model_name} ({cv}-fold)...")
    
    # Use stratified K-fold for classification
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Compute cross-validation scores for different metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = {}
    
    for metric_name, scoring in scoring_metrics.items():
        scores = cross_val_score(model, X_train, y_train, 
                                cv=cv_strategy, scoring=scoring, n_jobs=-1)
        cv_results[metric_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return cv_results


def print_cv_results(cv_results, model_name):
    """
    Print cross-validation results in a structured format.
    
    Parameters:
    -----------
    cv_results : dict
        Dictionary containing cross-validation results
    model_name : str
        Name of the model
    """
    print(f"\n    Cross-Validation Results for {model_name}:")
    print("    " + "-" * 60)
    print(f"    {'Metric':<15} {'Mean Score':<15} {'Std Dev':<15} {'Range'}")
    print("    " + "-" * 60)
    
    for metric_name, results in cv_results.items():
        mean = results['mean']
        std = results['std']
        min_score = mean - std
        max_score = mean + std
        print(f"    {metric_name.capitalize():<15} {mean:>6.4f}        {std:>6.4f}        [{min_score:.4f}, {max_score:.4f}]")
    
    print("    " + "-" * 60)


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, results_dir, perform_cv=True):
    """
    Complete evaluation pipeline for a single model.
    Trains, predicts, computes metrics, performs cross-validation, and saves results.
    
    Parameters:
    -----------
    model : object
        Model to evaluate
    model_name : str
        Name of the model
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    X_test : pandas.DataFrame
        Test features
    y_test : pandas.Series
        Test target
    results_dir : str
        Directory to save results
    perform_cv : bool, default=True
        Whether to perform cross-validation
    
    Returns:
    --------
    metrics : dict
        Dictionary containing all evaluation metrics
    trained_model : object
        Trained model
    y_pred_proba : numpy.array
        Predicted probabilities
    cv_results : dict
        Cross-validation results (if performed)
    """
    print("\n" + "="*70)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("="*70)
    
    # Train the model
    print("\n[1] Model Training:")
    print("-" * 70)
    trained_model = train_model(model, X_train, y_train, model_name)
    
    # Make predictions
    print("\n[2] Making Predictions:")
    print("-" * 70)
    print("  Generating predictions on test set...", end=" ", flush=True)
    y_pred, y_pred_proba = make_predictions(trained_model, X_test)
    print("✓ Complete")
    print(f"  Predicted {len(y_pred)} samples")
    print(f"  Class distribution: {np.bincount(y_pred)}")
    
    # Compute metrics
    print("\n[3] Computing Evaluation Metrics:")
    print("-" * 70)
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    
    # Print metrics in structured format
    print(f"\n  Test Set Performance Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Metric':<20} {'Score':<15} {'Interpretation'}")
    print("  " + "-" * 60)
    print(f"  {'Accuracy':<20} {metrics['accuracy']:>6.4f}        {'Proportion of correct predictions'}")
    print(f"  {'Precision':<20} {metrics['precision']:>6.4f}        {'True positives / (TP + FP)'}")
    print(f"  {'Recall':<20} {metrics['recall']:>6.4f}        {'True positives / (TP + FN)'}")
    print(f"  {'F1-Score':<20} {metrics['f1_score']:>6.4f}        {'Harmonic mean of precision & recall'}")
    print(f"  {'ROC-AUC':<20} {metrics['roc_auc']:>6.4f}        {'Area under ROC curve'}")
    print("  " + "-" * 60)
    
    # Generate confusion matrix
    print("\n[4] Confusion Matrix Analysis:")
    print("-" * 70)
    cm = get_confusion_matrix(y_test, y_pred)
    
    # Print confusion matrix
    print("\n  Confusion Matrix:")
    print("  " + "-" * 40)
    print(f"  {'':<15} {'Predicted No':<15} {'Predicted Yes':<15}")
    print("  " + "-" * 40)
    print(f"  {'Actual No':<15} {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"  {'Actual Yes':<15} {cm[1,0]:<15} {cm[1,1]:<15}")
    print("  " + "-" * 40)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Detailed Breakdown:")
    print(f"    True Negatives (TN):  {tn:>6} - Correctly predicted no diabetes")
    print(f"    False Positives (FP): {fp:>6} - Incorrectly predicted diabetes")
    print(f"    False Negatives (FN): {fn:>6} - Missed diabetes cases")
    print(f"    True Positives (TP):  {tp:>6} - Correctly predicted diabetes")
    
    # Save confusion matrix plot
    cm_path = os.path.join(results_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png')
    plot_confusion_matrix(cm, model_name, cm_path)
    
    # Save classification report
    print("\n[5] Classification Report:")
    print("-" * 70)
    report = classification_report(y_test, y_pred, 
                                   target_names=['No Diabetes', 'Diabetes'],
                                   output_dict=True)
    
    # Print classification report
    print("\n  Per-Class Metrics:")
    print("  " + "-" * 60)
    print(f"  {'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support'}")
    print("  " + "-" * 60)
    for class_name in ['No Diabetes', 'Diabetes']:
        if class_name in report:
            metrics_dict = report[class_name]
            print(f"  {class_name:<15} {metrics_dict['precision']:>6.4f}      {metrics_dict['recall']:>6.4f}      {metrics_dict['f1-score']:>6.4f}      {int(metrics_dict['support']):>6}")
    print("  " + "-" * 60)
    print(f"  {'Accuracy':<15} {'':<12} {'':<12} {report['accuracy']:>6.4f}      {len(y_test):>6}")
    print(f"  {'Macro Avg':<15} {report['macro avg']['precision']:>6.4f}      {report['macro avg']['recall']:>6.4f}      {report['macro avg']['f1-score']:>6.4f}      {int(report['macro avg']['support']):>6}")
    print(f"  {'Weighted Avg':<15} {report['weighted avg']['precision']:>6.4f}      {report['weighted avg']['recall']:>6.4f}      {report['weighted avg']['f1-score']:>6.4f}      {int(report['weighted avg']['support']):>6}")
    
    # Save classification report to file
    report_path = os.path.join(results_dir, f'classification_report_{model_name.replace(" ", "_")}.txt')
    with open(report_path, 'w') as f:
        f.write(f"Classification Report - {model_name}\n")
        f.write("="*70 + "\n\n")
        f.write(classification_report(y_test, y_pred, 
                                      target_names=['No Diabetes', 'Diabetes']))
    print(f"\n    ✓ Saved classification report: {os.path.basename(report_path)}")
    
    # Perform cross-validation if requested
    cv_results = None
    if perform_cv:
        print("\n[6] Cross-Validation:")
        print("-" * 70)
        cv_results = cross_validate_model(trained_model, X_train, y_train, model_name, cv=5)
        print_cv_results(cv_results, model_name)
    
    print("\n" + "="*70)
    
    return metrics, trained_model, y_pred_proba, cv_results


def plot_roc_curves(models_data, y_test, results_dir):
    """
    Plot ROC curves for all models on the same graph for comparison.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and y_pred_proba arrays as values
    y_test : pandas.Series or numpy.array
        True test labels
    results_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Plot diagonal line (random classifier baseline)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)', 
             linewidth=2, alpha=0.7)
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot ROC curve for each model
    for idx, (model_name, y_pred_proba) in enumerate(models_data.items()):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', 
                linewidth=2.5, color=colors[idx % len(colors)])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    plt.title('ROC Curves Comparison - All Models', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    
    plot_path = os.path.join(results_dir, 'roc_curves_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved ROC curves comparison: {os.path.basename(plot_path)}")
    plt.close()


def plot_precision_recall_curves(models_data, y_test, results_dir):
    """
    Plot Precision-Recall curves for all models on the same graph.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and y_pred_proba arrays as values
    y_test : pandas.Series or numpy.array
        True test labels
    results_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Define colors for each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot Precision-Recall curve for each model
    for idx, (model_name, y_pred_proba) in enumerate(models_data.items()):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.4f})', 
                linewidth=2.5, color=colors[idx % len(colors)])
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
    plt.title('Precision-Recall Curves Comparison - All Models', 
             fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower left", fontsize=10, framealpha=0.9)
    plt.grid(alpha=0.3, linestyle='--')
    
    plot_path = os.path.join(results_dir, 'precision_recall_curves_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved Precision-Recall curves: {os.path.basename(plot_path)}")
    plt.close()


def plot_feature_importance(model, model_name, feature_names, results_dir, top_n=15):
    """
    Plot feature importance for tree-based models (Random Forest, XGBoost).
    
    Parameters:
    -----------
    model : object
        Trained model (must have feature_importances_ attribute)
    model_name : str
        Name of the model
    feature_names : list
        List of feature names
    results_dir : str
        Directory to save the plot
    top_n : int, default=15
        Number of top features to display
    """
    # Check if model has feature_importances_ attribute
    importances = None
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps'):
        # For pipeline models, check the last step
        for step_name, step_model in model.named_steps.items():
            if hasattr(step_model, 'feature_importances_'):
                importances = step_model.feature_importances_
                break
    
    if importances is None:
        return  # No feature importance available
    
    # Get feature importance
    feature_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(range(len(feature_imp)), feature_imp['importance'], 
                    color='steelblue', edgecolor='navy', linewidth=1.5)
    plt.yticks(range(len(feature_imp)), feature_imp['feature'])
    plt.xlabel('Feature Importance', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importance - {model_name}', 
             fontsize=14, fontweight='bold', pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(feature_imp.iterrows()):
        plt.text(row['importance'], i, f' {row["importance"]:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, f'feature_importance_{model_name.replace(" ", "_")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved feature importance: {os.path.basename(plot_path)}")
    plt.close()


def plot_metrics_heatmap(comparison_df, results_dir):
    """
    Create a heatmap visualization of all metrics for all models.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison metrics
    results_dir : str
        Directory to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Create heatmap
    sns.heatmap(comparison_df.T, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, 
                square=False, vmin=0, vmax=1)
    plt.title('Model Performance Metrics Heatmap', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel('Metrics', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plot_path = os.path.join(results_dir, 'metrics_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved metrics heatmap: {os.path.basename(plot_path)}")
    plt.close()


def plot_metrics_radar(comparison_df, results_dir):
    """
    Create a radar chart comparing all models across all metrics.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison metrics
    results_dir : str
        Directory to save the plot
    """
    # Number of metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    num_metrics = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each model
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (model_name, row) in enumerate(comparison_df.iterrows()):
        values = [row[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model_name, 
               color=colors[idx % len(colors)], markersize=8)
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    plot_path = os.path.join(results_dir, 'metrics_radar_chart.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved radar chart: {os.path.basename(plot_path)}")
    plt.close()


def plot_comparison(comparison_df, results_dir):
    """
    Create a bar plot comparing all models across all metrics.
    
    Parameters:
    -----------
    comparison_df : pandas.DataFrame
        DataFrame with model comparison metrics
    results_dir : str
        Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for plotting
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    x = np.arange(len(comparison_df.index))
    width = 0.15
    
    # Define colors for each metric
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, metric in enumerate(metrics):
        offset = (i - 2) * width
        ax.bar(x + offset, comparison_df[metric], width, 
              label=metric.replace('_', ' ').title(), color=colors[i])
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison - All Metrics', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    plot_path = os.path.join(results_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison bar chart: {os.path.basename(plot_path)}")
    plt.close()


def compare_models(results_dict, models_predictions, trained_models, X_train, y_test, results_dir):
    """
    Compare all models and create a comprehensive summary with visualizations.
    Prints structured comparison table and identifies the best model.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys and metrics dictionaries as values
    models_predictions : dict
        Dictionary with model names as keys and y_pred_proba arrays as values
    trained_models : dict
        Dictionary with model names as keys and trained model objects as values
    X_train : pandas.DataFrame
        Training features (for feature importance)
    y_test : pandas.Series
        Test target (for ROC and PR curves)
    results_dir : str
        Directory to save results
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*70)
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame(results_dict).T
    comparison_df = comparison_df.round(4)
    
    # Print comparison table
    print("\n[1] Performance Metrics Comparison:")
    print("-" * 70)
    print(comparison_df.to_string())
    
    # Find best model for each metric
    print("\n[2] Best Model by Metric:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Best Model':<25} {'Score':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<20} {comparison_df['accuracy'].idxmax():<25} {comparison_df['accuracy'].max():>6.4f}")
    print(f"{'Precision':<20} {comparison_df['precision'].idxmax():<25} {comparison_df['precision'].max():>6.4f}")
    print(f"{'Recall':<20} {comparison_df['recall'].idxmax():<25} {comparison_df['recall'].max():>6.4f}")
    print(f"{'F1-Score':<20} {comparison_df['f1_score'].idxmax():<25} {comparison_df['f1_score'].max():>6.4f}")
    print(f"{'ROC-AUC':<20} {comparison_df['roc_auc'].idxmax():<25} {comparison_df['roc_auc'].max():>6.4f}")
    
    # Overall best model - Focus on Recall and F1-Score for imbalanced dataset
    # For imbalanced datasets, we prioritize Recall (to catch more diabetes cases)
    # and F1-Score (to balance precision and recall)
    print("\n[3] Model Selection for Imbalanced Dataset:")
    print("-" * 70)
    print("  Note: For imbalanced datasets, we prioritize:")
    print("    • Recall: Ability to identify diabetes cases (minimize false negatives)")
    print("    • F1-Score: Balance between precision and recall")
    
    # Find best model by F1-score (primary) and Recall (secondary)
    best_model_f1 = comparison_df['f1_score'].idxmax()
    best_model_recall = comparison_df['recall'].idxmax()
    best_f1 = comparison_df.loc[best_model_f1, 'f1_score']
    best_recall = comparison_df.loc[best_model_recall, 'recall']
    
    # Overall best model (by F1-score, as it balances precision and recall)
    best_model = best_model_f1
    
    print(f"\n  Best F1-Score: {best_model_f1} ({best_f1:.4f})")
    print(f"  Best Recall:    {best_model_recall} ({best_recall:.4f})")
    
    print("\n[4] Overall Best Model (Selected by F1-Score):")
    print("-" * 70)
    print(f"  🏆 WINNER: {best_model}")
    print(f"  Best F1-Score: {best_f1:.4f}")
    print(f"  (F1-Score balances precision and recall, ideal for imbalanced datasets)")
    print(f"\n  Complete Performance:")
    print(f"    Accuracy:  {comparison_df.loc[best_model, 'accuracy']:.4f}")
    print(f"    Precision: {comparison_df.loc[best_model, 'precision']:.4f}")
    print(f"    Recall:    {comparison_df.loc[best_model, 'recall']:.4f} ⭐ (Key metric for minority class)")
    print(f"    F1-Score:  {comparison_df.loc[best_model, 'f1_score']:.4f} ⭐ (Key metric for imbalanced data)")
    print(f"    ROC-AUC:   {comparison_df.loc[best_model, 'roc_auc']:.4f}")
    
    # Rank all models - Focus on Recall and F1-Score for imbalanced dataset
    print("\n[5] Model Rankings (Prioritizing Recall & F1-Score):")
    print("-" * 70)
    # Rank by F1-score (primary), then by Recall (secondary), then by ROC-AUC
    ranked = comparison_df.sort_values(['f1_score', 'recall', 'roc_auc'], ascending=False)
    print(f"{'Rank':<8} {'Model':<25} {'F1-Score':<12} {'Recall':<12} {'ROC-AUC':<12}")
    print("-" * 70)
    for rank, (model_name, row) in enumerate(ranked.iterrows(), 1):
        marker = "🏆" if rank == 1 else f"{rank}."
        print(f"{marker:<8} {model_name:<25} {row['f1_score']:>6.4f}    {row['recall']:>6.4f}    {row['roc_auc']:>6.4f}")
    
    # Save comparison table
    comparison_path = os.path.join(results_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path)
    print(f"\n  ✓ Saved comparison table: {os.path.basename(comparison_path)}")
    
    # Create and save all comparison visualizations
    print("\n[6] Generating Visualizations:")
    print("-" * 70)
    plot_comparison(comparison_df, results_dir)
    plot_metrics_heatmap(comparison_df, results_dir)
    plot_metrics_radar(comparison_df, results_dir)
    plot_roc_curves(models_predictions, y_test, results_dir)
    plot_precision_recall_curves(models_predictions, y_test, results_dir)
    
    # Plot feature importance for tree-based models
    feature_names = X_train.columns.tolist()
    for model_name, model in trained_models.items():
        if model_name in ['Random Forest', 'XGBoost']:
            plot_feature_importance(model, model_name, feature_names, results_dir)
    
    print("\n" + "="*70)