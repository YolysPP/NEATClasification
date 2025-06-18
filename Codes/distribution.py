"""
Class Distribution Analysis for Extracted Feature Files
This script analyzes the class distribution (Benign vs Malignant)
in the extracted feature CSV files from the BreakHis dataset.

Authors:
Dr. Yolanda Pérez Pimentel
Dr. Ismael Osuna Galán
"""

import pandas as pd

# === List of CSV files to analyze ===
csv_files = [
    "features_400x_custom.csv",         # Full dataset
    "features_train_balanced.csv",      # Training subset
    "features_test_balanced.csv"        # Testing subset
]

# === Analyze class distribution in each file ===
for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"\n--- {file} ---")

        # Count instances per class
        counts = df['label'].value_counts().sort_index()
        total = counts.sum()

        print("Class Counts:")
        print(f"  Benign (0):    {counts.get(0, 0)}")
        print(f"  Malignant (1): {counts.get(1, 0)}")

        # Calculate class percentages
        print("Class Percentages:")
        print(f"  Benign (0):    {counts.get(0, 0) / total * 100:.2f}%")
        print(f"  Malignant (1): {counts.get(1, 0) / total * 100:.2f}%")

    except Exception as e:
        print(f"Error processing {file}: {e}")
