# NEATClasification

# NEAT with PCA for Breast Cancer Histopathology Classification

This repository contains a complete pipeline for classifying breast tumor histopathology images using **NeuroEvolution of Augmenting Topologies (NEAT)**, combined with **Principal Component Analysis (PCA)** for dimensionality reduction. The project uses the publicly available [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis?resource=download).

## Project Objective

To evaluate the effectiveness of neuroevolutionary models, particularly NEAT, in the binary classification of histopathological images of breast cancer, distinguishing between benign and malignant cases using features extracted from 400X magnification slides.

## Dataset

- **Name:** BreakHis (Breast Cancer Histopathological Database)
- **Source:** [Kaggle - BreakHis](https://www.kaggle.com/datasets/ambarish/breakhis)
- **Magnification used:** 400X
- **Classes:** Benign (0), Malignant (1)

## Methodology

### 1. Feature Extraction

Each image undergoes preprocessing followed by region-based feature extraction:

- **Preprocessing:**
  - RGB to grayscale conversion
  - CLAHE (contrast enhancement)
  - Otsu thresholding
  - Morphological cleaning

- **Extracted features (per region):**
  - Solidity
  - Eccentricity
  - Aspect Ratio
  - Extent
  - Entropy (Shannon)
  - First 3 Hu moments

- **Aggregation (per image):**
  - Mean, standard deviation, and maximum of each feature
  - Region count (number of valid regions per image)

### 2. Dimensionality Reduction

- Features are standardized using `StandardScaler`.
- PCA reduces the feature space to 15 components.

### 3. Classification using NEAT

- NEAT evolves neural networks with topologies adapted to the task.
- Fitness is based on the average recall of both classes, penalizing imbalance.
- Evaluated using:
  - Accuracy
  - F1-score
  - Precision
  - Recall
  - Specificity
  - ROC-AUC

## Outputs

- `features_400x_custom.csv`: Extracted full feature set.
- `features_train_balanced.csv` and `features_test_balanced.csv`: Balanced training and testing sets.
- `confusion_matrix_neat_pca.pdf`: Confusion matrix.
- `roc_curve_neat_pca.pdf`: ROC curve.
- `winner_topology.svg`: Visualization of the evolved NEAT topology.
- `neat_metrics_log.csv`: Metrics log (appended for each execution).

## Installation Instructions

Install the following Python libraries before running the code:

pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install opencv-python
pip install scikit-image
pip install neat-python


## How to Run

1. Download and extract the BreakHis dataset under this path:


   BreakHis_v1/histology_slides/breast/


2. Run the feature extraction and data preparation script:


   python balanced_features_extraction.py


3. Run the NEAT training and evaluation script:


   python Neat_With_PCA.py


4. (Optional) Analyze class distribution in the generated datasets:

 
   python distribution.py


## Authors

* Dr. Yolanda Pérez Pimentel
* Dr. Ismael Osuna Galán
* Dr. Homero Toral Cruz
* Dr. José Antonio León Borges
* Dr. David Ernesto Troncoso Romero
* Dr. Julio César Pacheco Ramírez


## License

This project is for academic and research purposes. Please credit the authors in any derivative work or publication.

## Acknowledgments

* BreakHis dataset: (https://www.kaggle.com/datasets/ambarish/breakhis))
* NEAT algorithm: (http://nn.cs.utexas.edu/?neat)

