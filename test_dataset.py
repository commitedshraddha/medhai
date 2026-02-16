"""
Quick test script to verify the dataset structure
"""
import pandas as pd
import os

# Load the dataset
data_path = os.path.join(os.path.dirname(__file__), 'data', 'adaptive_learning_rich_dataset.csv')

print("Loading dataset...")
df = pd.read_csv(data_path)

print(f"\n✓ Dataset loaded successfully!")
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "="*60)
print("DATASET STRUCTURE")
print("="*60)

print("\nColumns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "="*60)
print("FIRST 3 ROWS")
print("="*60)
print(df.head(3).to_string())

print("\n" + "="*60)
print("KEY COLUMN INFO")
print("="*60)

# Check required columns
required_cols = ['Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
                 'Exercise_Completion_Rate', 'Time_Spent_On_Learning_Platform',
                 'Learning_Path_Recommendation']

for col in required_cols:
    if col in df.columns:
        print(f"\n✓ {col}")
        print(f"  Type: {df[col].dtype}")
        print(f"  Sample values: {df[col].head(3).tolist()}")
        if df[col].dtype == 'object':
            print(f"  Unique values: {df[col].unique().tolist()}")
    else:
        print(f"\n✗ {col} - MISSING!")

print("\n" + "="*60)
print("TARGET DISTRIBUTION")
print("="*60)

if 'Learning_Path_Recommendation' in df.columns:
    print(df['Learning_Path_Recommendation'].value_counts())
    print(f"\nTotal: {df['Learning_Path_Recommendation'].count()}")
else:
    print("✗ Learning_Path_Recommendation column not found!")

print("\n" + "="*60)
print("MISSING VALUES CHECK")
print("="*60)
missing = df[required_cols].isnull().sum()
print(missing[missing > 0] if any(missing > 0) else "✓ No missing values in required columns!")

print("\n" + "="*60)
print("READY FOR TRAINING!")
print("="*60)
