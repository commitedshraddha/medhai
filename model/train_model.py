"""
MedhAI+ Machine Learning Model Training Script
===============================================
This script loads the adaptive learning dataset, trains a RandomForestClassifier 
to predict learning paths, and saves the model.

Target Classes:
    0 - Remedial Support (needs foundational work)
    1 - Continue Current Level (on track)
    2 - Advance (ready for harder challenges)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_dataset(filename='adaptive_learning_rich_dataset.csv'):
    """
    Load the adaptive learning dataset from CSV file.
    
    Args:
        filename: Name of the CSV file containing the dataset
        
    Returns:
        DataFrame with student learning metrics
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
    
    print(f"Loading dataset from '{filename}'...")
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total records: {len(df)}")
        print(f"  Features: {list(df.columns)}")
        
        # Display basic statistics
        print("\nDataset Overview:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nTarget Distribution:")
        if 'Learning_Path_Recommendation' in df.columns:
            print(df['Learning_Path_Recommendation'].value_counts())
        
        return df
        
    except FileNotFoundError:
        print(f"✗ Error: Dataset file '{filename}' not found at {data_path}")
        print("Please ensure the dataset file exists in the data/ directory.")
        raise
    except Exception as e:
        print(f"✗ Error loading dataset: {str(e)}")
        raise


def train_model(df):
    """
    Train a RandomForestClassifier on the student data.
    
    Args:
        df: DataFrame containing student features and target
        
    Returns:
        Trained model and test accuracy
        
    Note:
        Feature order for predictions (MUST match this order in app.py):
        [Quiz_Scores, Final_Exam_Score, Engagement_Level, 
         Exercise_Completion_Rate, Time_Spent_On_Learning_Platform]
    """
    print("\n--- Training Machine Learning Model ---")
    
    # Define feature columns - using ALL relevant numeric features for better accuracy
    feature_columns = [
        'Quiz_Scores', 
        'Final_Exam_Score', 
        'Engagement_Level', 
        'Exercise_Completion_Rate', 
        'Time_Spent_On_Learning_Platform',
        'Attendance_Percentage',
        'Vocabulary_Improvement_Score',
        'Grammar_Improvement_Score',
        'Reading_Ability_Score',
        'Listening_Ability_Score',
        'Writing_Ability_Score',
        'Speaking_Ability_Score'
    ]
    
    # Core features for prediction API (first 5)
    core_features = feature_columns[:5]
    
    # Check if all required columns exist
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Dataset missing required columns: {missing_cols}")
    
    # Separate features and target
    X = df[feature_columns].copy()  # Make a copy to avoid SettingWithCopyWarning
    
    # Handle target variable - convert to numeric if it's categorical
    if 'Learning_Path_Recommendation' not in df.columns:
        raise ValueError("Target column 'Learning_Path_Recommendation' not found in dataset")
    
    y = df['Learning_Path_Recommendation']
    
    # Convert categorical learning paths to numeric if needed
    if y.dtype == 'object':
        print("\nConverting categorical learning paths to numeric...")
        
        # Check unique values in target
        unique_paths = y.unique()
        print(f"Unique learning paths found: {unique_paths}")
        
        # Create mapping - handle all possible values
        path_mapping = {
            'Remedial Support': 0,
            'Continue Current Level': 1,
            'Advance': 2,
            'Custom Path': 1  # Map Custom Path to Continue (middle ground)
        }
        
        # Check for unmapped values
        unmapped = [p for p in unique_paths if p not in path_mapping]
        if unmapped:
            print(f"⚠️  Warning: Unmapped paths found: {unmapped}")
            print(f"   These will be mapped to 'Continue Current Level' (1)")
            for path in unmapped:
                path_mapping[path] = 1
        
        y = y.map(path_mapping)
        print(f"Mapping used: {path_mapping}")
        
        # Check for any NaN values after mapping
        if y.isnull().any():
            print(f"✗ Error: {y.isnull().sum()} NaN values found after mapping!")
            print(f"   Problematic values: {df.loc[y.isnull(), 'Learning_Path_Recommendation'].unique()}")
            raise ValueError("Failed to map all learning paths to numeric values")
    
    # Handle Engagement_Level if it's categorical (Low/Medium/High)
    if X['Engagement_Level'].dtype == 'object':
        print("\nConverting Engagement_Level to numeric...")
        engagement_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        X['Engagement_Level'] = X['Engagement_Level'].map(engagement_mapping)
        print(f"Engagement mapping: {engagement_mapping}")
    
    # Remove any rows with missing values
    initial_rows = len(X)
    X = X.dropna()
    y = y[X.index]
    if len(X) < initial_rows:
        print(f"\nRemoved {initial_rows - len(X)} rows with missing values")
    
    print(f"\nFinal dataset size: {len(X)} samples")
    print(f"Total features used: {len(feature_columns)}")
    print(f"Core features (for API): {core_features}")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    
    # Initialize and train RandomForestClassifier with optimized hyperparameters
    model = RandomForestClassifier(
        n_estimators=200,        # More trees for better accuracy
        max_depth=15,            # Deeper trees
        min_samples_split=10,    # Prevent overfitting
        min_samples_leaf=5,      # Minimum samples per leaf
        max_features='sqrt',     # Use sqrt of features at each split
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    print("\nTraining model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✓ Model trained successfully!")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Remedial', 'Continue', 'Advance']))
    
    # Feature importance
    print("\nFeature Importance:")
    for feature, importance in zip(X.columns, model.feature_importances_):
        print(f"  {feature}: {importance:.4f}")
    
    return model, accuracy

def save_model(model, filename='random_forest.pkl'):
    """
    Save the trained model to a pickle file with metadata.
    
    Args:
        model: Trained sklearn model
        filename: Name of the pickle file
    """
    model_path = os.path.join(os.path.dirname(__file__), filename)
    
    # Create a model wrapper with metadata
    model_package = {
        'model': model,
        'core_features': [
            'Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
            'Exercise_Completion_Rate', 'Time_Spent_On_Learning_Platform'
        ],
        'all_features': [
            'Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
            'Exercise_Completion_Rate', 'Time_Spent_On_Learning_Platform',
            'Attendance_Percentage', 'Vocabulary_Improvement_Score',
            'Grammar_Improvement_Score', 'Reading_Ability_Score',
            'Listening_Ability_Score', 'Writing_Ability_Score',
            'Speaking_Ability_Score'
        ],
        'path_names': ['Remedial Support', 'Continue Current Level', 'Advance'],
        'engagement_mapping': {'Low': 1, 'Medium': 2, 'High': 3}
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"\n✓ Model saved as '{filename}'")
    print(f"Location: {model_path}")
    print(f"Package includes: model + metadata")

def test_prediction(model):
    """
    Test the model with sample predictions.
    
    Args:
        model: Trained model
    """
    print("\n--- Testing Predictions ---")
    
    # Test with full feature set (12 features)
    test_cases = [
        {
            'name': 'Low Performer',
            'data': [[45, 50, 1, 55, 2.0, 65, 40, 45, 50, 48, 52, 46]]  # 12 features
        },
        {
            'name': 'Average Performer',
            'data': [[70, 75, 2, 80, 3.5, 85, 70, 72, 75, 73, 71, 74]]  # 12 features
        },
        {
            'name': 'High Performer',
            'data': [[92, 95, 3, 95, 5.0, 98, 90, 92, 94, 93, 91, 95]]  # 12 features
        }
    ]
    
    path_names = ['Remedial Support', 'Continue Current Level', 'Advance']
    
    for case in test_cases:
        prediction = model.predict(case['data'])[0]
        probabilities = model.predict_proba(case['data'])[0]
        
        print(f"\n{case['name']}:")
        print(f"  Quiz={case['data'][0][0]}, Exam={case['data'][0][1]}, "
              f"Engagement={case['data'][0][2]}, Completion={case['data'][0][3]}%, "
              f"Time={case['data'][0][4]}hrs")
        print(f"  Prediction: {path_names[prediction]}")
        print(f"  Confidence: {probabilities[prediction] * 100:.2f}%")
        print(f"  All probabilities: Remedial={probabilities[0]*100:.1f}%, "
              f"Continue={probabilities[1]*100:.1f}%, Advance={probabilities[2]*100:.1f}%")

def main():
    """
    Main function to execute the training pipeline.
    """
    print("=" * 60)
    print("MedhAI+ - Machine Learning Model Training")
    print("=" * 60)
    
    # Load the actual dataset
    df = load_dataset('adaptive_learning_rich_dataset.csv')
    
    # Train model
    model, accuracy = train_model(df)
    
    # Save trained model
    save_model(model)
    
    # Test predictions
    test_prediction(model)
    
    print("\n" + "=" * 60)
    print("Training Complete! Model ready for deployment.")
    print("=" * 60)

if __name__ == "__main__":
    main()
