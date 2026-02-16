"""
MedhAI+ Improved Machine Learning Model Training Script
========================================================
Trains RandomForest with 12 features for better accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import os

np.random.seed(42)

def load_dataset():
    """Load the adaptive learning dataset"""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'adaptive_learning_rich_dataset.csv')
    print(f"Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
    return df

def train_improved_model(df):
    """Train model with 12 features"""
    print("\n=== TRAINING WITH 12 FEATURES ===\n")
    
    # All 12 features
    features = [
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
    
    print(f"Using {len(features)} features:")
    for i, f in enumerate(features, 1):
        print(f"  {i}. {f}")
    
    # Prepare data
    X = df[features].copy()
    y = df['Learning_Path_Recommendation']
    
    # Convert target to numeric
    print(f"\nTarget classes: {y.unique()}")
    path_map = {
        'Remedial Support': 0,
        'Continue Current Level': 1,
        'Advance': 2,
        'Custom Path': 1  # Map to Continue
    }
    y = y.map(path_map)
    print(f"Mapped to: {path_map}")
    
    # Convert Engagement_Level
    if X['Engagement_Level'].dtype == 'object':
        eng_map = {'Low': 1, 'Medium': 2, 'High': 3}
        X['Engagement_Level'] = X['Engagement_Level'].map(eng_map)
        print(f"Engagement mapped: {eng_map}")
    
    # Clean data
    X = X.dropna()
    y = y[X.index]
    print(f"\nFinal dataset: {len(X)} samples × {len(features)} features")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train optimized model
    print("\n Training RandomForest (200 trees, depth=15)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ ACCURACY: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Remedial', 'Continue', 'Advance'],
                                zero_division=0))
    
    print("\nFeature Importance:")
    importances = sorted(zip(features, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)
    for feat, imp in importances:
        print(f"  {feat:40s}: {imp:.4f}")
    
    # Test predictions
    print("\n=== TEST PREDICTIONS ===\n")
    test_cases = [
        ("Low Performer", [45, 50, 1, 55, 2.0, 65, 40, 45, 50, 48, 52, 46]),
        ("Average Performer", [70, 75, 2, 80, 3.5, 85, 70, 72, 75, 73, 71, 74]),
        ("High Performer", [92, 95, 3, 95, 5.0, 98, 90, 92, 94, 93, 91, 95])
    ]
    
    for name, data in test_cases:
        pred = model.predict([data])[0]
        proba = model.predict_proba([data])[0]
        paths = ['Remedial Support', 'Continue Current Level', 'Advance']
        print(f"{name}:")
        print(f"  Prediction: {paths[pred]}")
        print(f"  Confidence: {proba[pred]*100:.1f}%")
        print(f"  All: R={proba[0]*100:.1f}% C={proba[1]*100:.1f}% A={proba[2]*100:.1f}%\n")
    
    # Save model package
    model_package = {
        'model': model,
        'core_features': features[:5],
        'all_features': features,
        'path_names': ['Remedial Support', 'Continue Current Level', 'Advance'],
        'engagement_mapping': {'Low': 1, 'Medium': 2, 'High': 3}
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'random_forest.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print(f"✅ Model saved: {model_path}")
    return model

if __name__ == "__main__":
    print("="*60)
    print("MedhAI+ IMPROVED MODEL TRAINING")
    print("="*60)
    df = load_dataset()
    model = train_improved_model(df)
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
