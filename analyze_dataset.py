"""
Analyze the dataset to understand why accuracy is low
"""
import pandas as pd
import numpy as np
import os

# Load dataset
data_path = os.path.join(os.path.dirname(__file__), 'data', 'adaptive_learning_rich_dataset.csv')
df = pd.read_csv(data_path)

print("="*70)
print("DATASET ANALYSIS")
print("="*70)

# Check target distribution
print("\nTarget Distribution:")
print(df['Learning_Path_Recommendation'].value_counts())
print(f"\nClass Balance:")
for cls, count in df['Learning_Path_Recommendation'].value_counts().items():
    print(f"  {cls}: {count/len(df)*100:.1f}%")

# Analyze feature correlations with target
print("\n" + "="*70)
print("FEATURE STATISTICS BY TARGET CLASS")
print("="*70)

features = ['Quiz_Scores', 'Final_Exam_Score', 'Exercise_Completion_Rate']

for feature in features:
    print(f"\n{feature}:")
    for target in ['Remedial Support', 'Continue Current Level', 'Advance']:
        subset = df[df['Learning_Path_Recommendation'] == target][feature]
        print(f"  {target:25s}: mean={subset.mean():.1f}, std={subset.std():.1f}, "
              f"min={subset.min():.1f}, max={subset.max():.1f}")

# Check if there's overlap
print("\n" + "="*70)
print("FEATURE OVERLAP ANALYSIS")
print("="*70)

print("\nQuiz_Scores ranges:")
for target in ['Remedial Support', 'Continue Current Level', 'Advance']:
    scores = df[df['Learning_Path_Recommendation'] == target]['Quiz_Scores']
    q1, q3 = scores.quantile(0.25), scores.quantile(0.75)
    print(f"  {target:25s}: 25th={q1:.1f}, 75th={q3:.1f}, IQR={q3-q1:.1f}")

# Check engagement level distribution
print("\n" + "="*70)
print("ENGAGEMENT LEVEL BY TARGET")
print("="*70)
for target in ['Remedial Support', 'Continue Current Level', 'Advance']:
    eng_dist = df[df['Learning_Path_Recommendation'] == target]['Engagement_Level'].value_counts()
    print(f"\n{target}:")
    print(eng_dist)

# Sample some actual records
print("\n" + "="*70)
print("SAMPLE RECORDS")
print("="*70)

cols_to_show = ['Quiz_Scores', 'Final_Exam_Score', 'Engagement_Level', 
                'Exercise_Completion_Rate', 'Learning_Path_Recommendation']

print("\nRemedial Support samples:")
print(df[df['Learning_Path_Recommendation'] == 'Remedial Support'][cols_to_show].head(5))

print("\nAdvance samples:")
print(df[df['Learning_Path_Recommendation'] == 'Advance'][cols_to_show].head(5))

# Check for data quality issues
print("\n" + "="*70)
print("DATA QUALITY CHECK")
print("="*70)

print("\nDuplicate rows:", df.duplicated().sum())
print("Missing values:")
print(df[features + ['Learning_Path_Recommendation']].isnull().sum())

# Correlation check
print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

# Simple rule-based prediction to compare
df_test = df.sample(1000, random_state=42)
correct = 0
for _, row in df_test.iterrows():
    score_avg = (row['Quiz_Scores'] + row['Final_Exam_Score']) / 2
    if score_avg >= 80:
        pred = 'Advance'
    elif score_avg >= 60:
        pred = 'Continue Current Level'
    else:
        pred = 'Remedial Support'
    
    if pred == row['Learning_Path_Recommendation']:
        correct += 1

rule_accuracy = correct / len(df_test) * 100
print(f"\nSimple rule-based accuracy (avg score thresholds): {rule_accuracy:.1f}%")
print("\nIf rule-based accuracy is also low (<50%), the dataset labels may be")
print("inconsistent or based on factors not included in the features.")
