"""
This module loads and preprocesses the diabetes dataset used for this project.
It aims to prepare the dataset to make it usable for the different models.
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import SMOTE for handling imbalanced datasets
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn (imblearn) is not installed.")
    print("SMOTE functionality will not be available.")
    print("Install it using: pip install imbalanced-learn")

def load_data(data_path=None):
    """
    Load the diabetes dataset from CSV file.
    
    Parameters:
    -----------
    data_path : str, optional
        Path to the data file. If None, uses the default diabetes dataset path.
    
    Returns:
    --------
    data : pandas.DataFrame
        Loaded dataset
    """
    if data_path is None:
        # Get the project root directory
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        
        # Default path to the diabetes dataset in data/raw/
        data_path = os.path.join(project_root, 'data', 'raw', 'diabetes_analysis.csv')
        
        # Normalize the path to handle any path issues
        data_path = os.path.normpath(data_path)
        
        # If file doesn't exist, try multiple alternative paths
        if not os.path.exists(data_path):
            # Try 1: Relative to current working directory
            alt_path1 = os.path.join(os.getcwd(), 'data', 'raw', 'diabetes_analysis.csv')
            alt_path1 = os.path.normpath(alt_path1)
            if os.path.exists(alt_path1):
                data_path = alt_path1
            else:
                # Try 2: Relative to script location (if running from different directory)
                try:
                    script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                    alt_path2 = os.path.join(script_dir, 'data', 'raw', 'diabetes_analysis.csv')
                    alt_path2 = os.path.normpath(alt_path2)
                    if os.path.exists(alt_path2):
                        data_path = alt_path2
                except:
                    pass  # If sys.argv is not available, skip this attempt
    
    # Load the CSV file
    print("="*70)
    print("LOADING DATASET")
    print("="*70)
    print(f"Loading data from: {data_path}")
    
    # Verifies if the file exists before attempting to load it
    if not os.path.exists(data_path):
        error_msg = f"✗ Error: File not found at {data_path}\n"
        error_msg += f"  Current working directory: {os.getcwd()}\n"
        error_msg += f"  Please ensure the file exists at: data/raw/diabetes_analysis.csv"
        print(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        data = pd.read_csv(data_path)
        print(f"✓ Successfully loaded {len(data)} rows and {len(data.columns)} columns")
        print(f"  Dataset shape: {data.shape[0]} samples × {data.shape[1]} features")
        return data
    except FileNotFoundError:
        print(f"✗ Error: File not found at {data_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        raise


def explore_data(data):
    """
    Perform exploratory data analysis (EDA) on the dataset.
    Prints summary statistics and information about the dataset.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset to explore
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Basic information
    print("\n[1] Dataset Overview:")
    print("-" * 70)
    print(f"  Total samples: {len(data)}")
    print(f"  Total features: {len(data.columns)}")
    print(f"  Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print("\n[2] Column Information:")
    print("-" * 70)
    print(f"{'Column Name':<25} {'Data Type':<15} {'Non-Null Count':<20}")
    print("-" * 70)
    for col in data.columns:
        non_null = data[col].notna().sum()
        dtype = str(data[col].dtype)
        print(f"{col:<25} {dtype:<15} {non_null:<20}")
    
    # Missing values
    print("\n[3] Missing Values Analysis:")
    print("-" * 70)
    missing = data.isnull().sum()
    missing_pct = (missing / len(data)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Missing Percentage': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  ✓ No missing values found in the dataset")
    
    # Distribution of the target variable "diabetic"
    if 'diabetic' in data.columns:
        print("\n[4] Target Variable Distribution:")
        print("-" * 70)
        target_counts = data['diabetic'].value_counts()
        target_pct = data['diabetic'].value_counts(normalize=True) * 100
        print(f"  Class 'No' (0):  {target_counts.get('No', target_counts.get(0, 0)):>6} samples ({target_pct.get('No', target_pct.get(0, 0)):.2f}%)")
        print(f"  Class 'Yes' (1): {target_counts.get('Yes', target_counts.get(1, 0)):>6} samples ({target_pct.get('Yes', target_pct.get(1, 0)):.2f}%)")
        print(f"  Total:           {len(data):>6} samples")
    
    # Statistical summary for numeric columns
    print("\n[5] Statistical Summary (Numeric Features):")
    print("-" * 70)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'diabetic' in numeric_cols:
        numeric_cols.remove('diabetic')
    if numeric_cols:
        summary = data[numeric_cols].describe()
        print(summary.round(2).to_string())
    
    # Categorical columns summary ("gender" and "diabetic" columns)
    print("\n[6] Categorical Features Summary:")
    print("-" * 70)
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            print(f"\n  {col}:")
            value_counts = data[col].value_counts()
            for val, count in value_counts.head(10).items():
                pct = (count / len(data)) * 100
                print(f"    {val}: {count:>6} ({pct:>5.2f}%)")
    else:
        print("  No categorical columns found")
    
    print("\n" + "="*70)


def preprocess_data(data):
    """
    Cleans and preprocesses the dataset.
    Handles missing values, converts categorical variables, and prepares data for modeling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Raw data to preprocess
    
    Returns:
    --------
    processed_data : pandas.DataFrame
        Preprocessed data ready for modeling
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Makes a copy of the data to avoid modifying the original data
    df = data.copy()
    
    print(f"\n[1] Initial Data Shape: {df.shape}")
    
    # Check for missing values
    print("\n[2] Handling Missing Values:")
    print("-" * 70)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"  Found {missing.sum()} missing values across {missing[missing > 0].shape[0]} columns")
        # Fills missing values with the median value for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  ✓ Filled {col} missing values with median: {median_val:.2f}")
        
        # Fill missing values with mode for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  ✓ Filled {col} missing values with mode: {mode_val}")
    else:
        print("  ✓ No missing values found")
    
    # Convert target variable ("diabetic") from Yes/No to 0/1 (binary form)
    print("\n[3] Encoding Target Variable:")
    print("-" * 70)
    if 'diabetic' in df.columns:
        if df['diabetic'].dtype == 'object':
            df['diabetic'] = df['diabetic'].map({'No': 0, 'Yes': 1})
            print("  ✓ Converted 'diabetic' target: 'No' → 0, 'Yes' → 1")
        else:
            print("  ✓ Target variable already numeric")
        
        # Check distribution after conversion
        target_dist = df['diabetic'].value_counts()
        print(f"  Target distribution: {dict(target_dist)}")
    
    # Converts the feature "gender" to binary form (0/1) if it exists
    print("\n[4] Encoding Categorical Features:")
    print("-" * 70)
    if 'gender' in df.columns:
        if df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
            print("  ✓ Encoded 'gender': 'Female' → 0, 'Male' → 1")
        else:
            print("  ✓ 'gender' already encoded")
    
    # Check for any remaining object columns that might need encoding
    remaining_categorical = df.select_dtypes(include=['object']).columns.tolist()
    if 'diabetic' in remaining_categorical:
        remaining_categorical.remove('diabetic')
    
    if remaining_categorical:
        print(f"  Note: {len(remaining_categorical)} categorical columns remain: {remaining_categorical}")
        print("  (These will be handled by the models if needed)")
    
    print(f"\n[5] Final Data Shape: {df.shape}")
    print("="*70)
    
    return df


def prepare_data(data, test_size=0.2, random_state=42):
    """
    Prepare data for machine learning by splitting into features and target,
    and then into training and testing sets using stratified sampling.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Preprocessed dataset
    test_size : float, default=0.2
        Proportion of data to use for testing (20%)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    X_test : pandas.DataFrame
        Testing features
    y_train : pandas.Series
        Training target
    y_test : pandas.Series
        Testing target
    """
    print("\n" + "="*70)
    print("DATA PREPARATION FOR MODELING")
    print("="*70)
    
    # Separate features and target
    print("\n[1] Separating Features and Target:")
    print("-" * 70)
    X = data.drop('diabetic', axis=1)
    y = data['diabetic']
    
    print(f"  Features (X): {X.shape[1]} columns")
    print(f"  Target (y):   {y.shape[0]} samples")
    print(f"  Feature names: {list(X.columns)}")
    
    # Display target ("diabetic") distribution 
    print("\n[2] Target Variable Distribution:")
    print("-" * 70)
    target_counts = y.value_counts().sort_index()
    target_pct = y.value_counts(normalize=True).sort_index() * 100
    for label, count in target_counts.items():
        pct = target_pct[label]
        class_name = "No Diabetes" if label == 0 else "Diabetes"
        print(f"  {class_name} ({label}): {count:>6} samples ({pct:>5.2f}%)")
    
    # Splits the data into training and testing sets with stratification
    print("\n[3] Splitting Data into Train/Test Sets:")
    print("-" * 70)
    print(f"  Train size: {1-test_size:.0%} ({int(len(data) * (1-test_size))} samples)")
    print(f"  Test size:  {test_size:.0%} ({int(len(data) * test_size)} samples)")
    print(f"  Random state: {random_state} (for reproducibility)")
    print(f"  Stratified: Yes (maintains class distribution)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Maintain class distribution in both sets
    )
    
    print("\n[4] Split Results:")
    print("-" * 70)
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Verifies if stratification worked
    print("\n[5] Class Distribution Verification:")
    print("-" * 70)
    train_dist = y_train.value_counts(normalize=True).sort_index() * 100
    test_dist = y_test.value_counts(normalize=True).sort_index() * 100
    
    print(f"{'Class':<15} {'Train Set':<15} {'Test Set':<15} {'Difference':<15}")
    print("-" * 70)
    for label in sorted(y_train.unique()):
        train_pct = train_dist[label]
        test_pct = test_dist[label]
        diff = abs(train_pct - test_pct)
        class_name = "No Diabetes" if label == 0 else "Diabetes"
        print(f"{class_name:<15} {train_pct:>6.2f}%      {test_pct:>6.2f}%      {diff:>6.2f}%")
    
    print("\n" + "="*70)
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to balance the training set.
    SMOTE is applied ONLY to the training set, NOT to the test set.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.array
        Training features
    y_train : pandas.Series or numpy.array
        Training target
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    X_train_resampled : numpy.array
        Resampled training features (balanced)
    y_train_resampled : numpy.array
        Resampled training target (balanced)
    """
    if not SMOTE_AVAILABLE:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install it using: pip install imbalanced-learn"
        )
    
    print("\n" + "="*70)
    print("APPLYING SMOTE FOR IMBALANCED DATASET HANDLING")
    print("="*70)
    
    # Display class distribution BEFORE SMOTE
    print("\n[1] Class Distribution BEFORE SMOTE:")
    print("-" * 70)
    train_counts_before = pd.Series(y_train).value_counts().sort_index()
    train_pct_before = pd.Series(y_train).value_counts(normalize=True).sort_index() * 100
    
    print(f"{'Class':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 70)
    for label in sorted(y_train.unique()):
        count = train_counts_before[label]
        pct = train_pct_before[label]
        class_name = "No Diabetes" if label == 0 else "Diabetes"
        print(f"{class_name:<20} {count:>6}        {pct:>6.2f}%")
    
    print(f"\n  Total samples: {len(y_train)}")
    print(f"  Imbalance ratio: {train_counts_before[0] / train_counts_before[1]:.2f}:1")
    
    # Converts to numpy arrays if they are DataFrames/Series
    X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    # Applies SMOTE
    print("\n[2] Applying SMOTE:")
    print("-" * 70)
    print("  Strategy: Oversample minority class to match majority class")
    print(f"  Random state: {random_state} (for reproducibility)")
    
    smote = SMOTE(random_state=random_state, sampling_strategy='auto')
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_array, y_train_array)
    
    print("  ✓ SMOTE applied successfully")
    
    # Displays the class distribution AFTER SMOTE
    print("\n[3] Class Distribution AFTER SMOTE:")
    print("-" * 70)
    train_counts_after = pd.Series(y_train_resampled).value_counts().sort_index()
    train_pct_after = pd.Series(y_train_resampled).value_counts(normalize=True).sort_index() * 100
    
    print(f"{'Class':<20} {'Count':<15} {'Percentage':<15}")
    print("-" * 70)
    for label in sorted(np.unique(y_train_resampled)):
        count = train_counts_after[label]
        pct = train_pct_after[label]
        class_name = "No Diabetes" if label == 0 else "Diabetes"
        print(f"{class_name:<20} {count:>6}        {pct:>6.2f}%")
    
    print(f"\n  Total samples: {len(y_train_resampled)}")
    print(f"  Samples added: {len(y_train_resampled) - len(y_train)}")
    print(f"  Balance ratio: {train_counts_after[0] / train_counts_after[1]:.2f}:1")
    
    # Verifies balance
    if abs(train_counts_after[0] - train_counts_after[1]) <= 1:
        print("\n  ✓ Dataset is now perfectly balanced!")
    else:
        print(f"\n  ✓ Dataset is now balanced (difference: {abs(train_counts_after[0] - train_counts_after[1])} samples)")
    
    print("\n[4] Important Notes:")
    print("-" * 70)
    print("  ✓ SMOTE was applied ONLY to the training set")
    print("  ✓ Test set remains unchanged (original distribution)")
    print("  ✓ This ensures realistic evaluation on imbalanced test data")
    
    print("\n" + "="*70)
    
    return X_train_resampled, y_train_resampled


def get_scaler():
    """
    Get a StandardScaler instance for optional feature scaling.
    Note: Scaling is typically done inside model pipelines, not globally.
    
    Returns:
    --------
    scaler : StandardScaler
        StandardScaler instance
    """
    return StandardScaler()