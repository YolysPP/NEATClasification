"""
NEAT with PCA Classification - Version 1
This code uses the BreakHis dataset, available on Kaggle:

https://www.kaggle.com/datasets/ambarish/breakhis?resource=download

Authors:
Dr. Yolanda Pérez Pimentel
Dr. Ismael Osuna Galán
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.exposure import equalize_adapthist
from scipy.ndimage import binary_opening, binary_erosion
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull

# === Feature Calculation Utilities ===

def calculate_entropy(image_region):
    """Calculate Shannon entropy of a grayscale region."""
    hist, _ = np.histogram(image_region.ravel(), bins=256, range=(0, 1), density=True)
    return entropy(hist + 1e-10)  # Avoid log(0) with small epsilon

def aspect_ratio(region):
    """Compute aspect ratio of the region."""
    return region.major_axis_length / (region.minor_axis_length + 1e-5)

def hu_moments(region_image):
    """Compute the first 3 Hu moments from the binary region image."""
    moments = cv2.moments(region_image.astype(np.uint8))
    hu = cv2.HuMoments(moments).flatten()
    return hu[:3]

# === Main Feature Extraction Function ===

def extract_features(image_path):
    """Extract multiple features from a histology image."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    # Convert to grayscale and enhance contrast
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = rgb2gray(image_rgb)
    gray_eq = equalize_adapthist(gray)

    # Binarize using Otsu thresholding
    thresh = threshold_otsu(gray_eq)
    binary = gray_eq > thresh

    # Clean binary image using morphological operations
    binary_clean = binary_opening(binary, structure=np.ones((3, 3)))
    binary_clean = binary_erosion(binary_clean, structure=np.ones((2, 2)))

    # Label connected components
    labeled = label(binary_clean)

    # Prepare dictionary to accumulate features
    features = {
        'solidity': [], 'eccentricity': [], 'aspect_ratio': [],
        'extent': [], 'entropy': [], 'hu1': [], 'hu2': [], 'hu3': []
    }

    # Loop through each detected region
    for region in regionprops(labeled):
        if region.area < 100:
            continue  # Ignore small/noisy regions

        try:
            # Compute features
            solidity = region.solidity
            eccentricity = region.eccentricity
            ar = aspect_ratio(region)
            extent = region.extent
            ent = calculate_entropy(gray[region.coords[:, 0], region.coords[:, 1]])
            hu = hu_moments(region.image)

            # Append to lists
            features['solidity'].append(solidity)
            features['eccentricity'].append(eccentricity)
            features['aspect_ratio'].append(ar)
            features['extent'].append(extent)
            features['entropy'].append(ent)
            features['hu1'].append(hu[0])
            features['hu2'].append(hu[1])
            features['hu3'].append(hu[2])
        except:
            continue  # Skip region if feature extraction fails

    # Aggregate features across all regions
    result = {}
    for key, vals in features.items():
        result[f"{key}_mean"] = np.mean(vals) if vals else 0
        result[f"{key}_std"] = np.std(vals) if vals else 0
        result[f"{key}_max"] = np.max(vals) if vals else 0

    # Include region count as a feature
    result["region_count"] = len(features["solidity"])
    return result

# === Image Processing Loop ===

def process_images(root_path, output_csv, train_csv, test_csv):
    """Process all 400X images, extract features, save CSVs."""
    data = []
    unique_paths = set()
    root = Path(root_path)

    # Traverse all subdirectories
    for subdir, _, files in os.walk(root):
        if not subdir.endswith("400X"):
            continue  # Only process 400X images

        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue  # Skip non-image files

            fpath = os.path.join(subdir, fname)
            if fpath in unique_paths:
                continue  # Avoid duplicate files

            unique_paths.add(fpath)

            # Determine label from folder name
            label_value = 1 if "malignant" in fpath.lower() else 0

            # Extract features
            feats = extract_features(fpath)
            if feats is not None:
                feats = {k: feats[k] for k in sorted(feats)}  # Sort features
                row = [label_value] + list(feats.values())
                data.append(row)

    if len(data) == 0:
        print("No features extracted.")
        return

    # Create DataFrame and save full dataset
    columns = ['label'] + sorted(feats.keys())
    df_out = pd.DataFrame(data, columns=columns)
    df_out.to_csv(output_csv, index=False)
    print(f"Full dataset saved: {output_csv} ({len(df_out)} instances)")

    # === Balance classes (undersample majority) ===
    benign_df = df_out[df_out['label'] == 0]
    n_benign = len(benign_df)
    malignant_df = df_out[df_out['label'] == 1].sample(n=n_benign, random_state=42)
    balanced_df = pd.concat([benign_df, malignant_df]).sample(frac=1, random_state=42)

    # === Train-test split (stratified) ===
    train_df, test_df = train_test_split(balanced_df, test_size=0.3, stratify=balanced_df["label"], random_state=42)

    # Save split data
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train file saved: {train_csv}")
    print(f"Test file saved: {test_csv}")

# === Run Extraction ===

process_images(
    root_path="BreakHis_v1/histology_slides/breast",  # Path to BreakHis dataset
    output_csv="features_400x_custom.csv",            # Full extracted dataset
    train_csv="features_train_balanced.csv",          # Balanced train set
    test_csv="features_test_balanced.csv"             # Balanced test set
)
