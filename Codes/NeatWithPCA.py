"""
NEAT with PCA Classification - Version 1
This code uses the BreakHis dataset, available on Kaggle:

https://www.kaggle.com/datasets/ambarish/breakhis?resource=download

Authors:
Dr. Yolanda Pérez Pimentel  
Dr. Ismael Osuna Galán
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, roc_curve, auc, balanced_accuracy_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import neat
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import visualize

# === 1. Load balanced training and test datasets ===
df_train = pd.read_csv("features_train_balanced.csv")
df_test = pd.read_csv("features_test_balanced.csv")

# Separate features and labels, ignore 'diagnosis' column if present
X_train = df_train.drop(columns=['label', 'diagnosis'], errors='ignore')
y_train = df_train['label']
X_test = df_test.drop(columns=['label', 'diagnosis'], errors='ignore')
y_test = df_test['label']

# === 2. Standardize features and apply PCA ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reduce dimensionality to 15 components
pca = PCA(n_components=15)
X_train_final = pca.fit_transform(X_train_scaled)
X_test_final = pca.transform(X_test_scaled)

# Classification threshold for binary output
THRESHOLD = 0.5

# === 3. Genome evaluation function ===
def eval_genomes(genomes, config):
    """
    Custom fitness function to evaluate genomes.
    It favors balanced recall for both classes and penalizes asymmetry.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        preds = [int(net.activate(x)[0] > THRESHOLD) for x in X_train_final]

        r0 = recall_score(y_train, preds, pos_label=0)
        r1 = recall_score(y_train, preds, pos_label=1)

        # Fitness promotes balanced recall and penalizes difference
        genome.fitness = (r0 + r1)/2 - 0.3 * abs(r0 - r1)

# === 4. Load NEAT configuration ===
config_path = "neat_config_bh_2.txt"
config = neat.Config(
    neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet, neat.DefaultStagnation,
    config_path
)

# === 5. Run NEAT Evolution ===
pop = neat.Population(config)
pop.add_reporter(neat.StdOutReporter(False))  # Suppress stdout logs
stats = neat.StatisticsReporter()
pop.add_reporter(stats)

generations = 50
start_time = time.time()
winner = pop.run(eval_genomes, generations)
time_neat = time.time() - start_time

# === 6. Evaluate best genome on the test set ===
winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
y_pred = [int(winner_net.activate(x)[0] > THRESHOLD) for x in X_test_final]
y_score = [winner_net.activate(x)[0] for x in X_test_final]

# Compute classification metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = conf.ravel()
specificity = tn / (tn + fp)

# === 7. Plot confusion matrix ===
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"], ax=ax)
ax.set_title("Confusion Matrix - NEAT with PCA")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.savefig("confusion_matrix_neat_pca.pdf", format="pdf", bbox_inches="tight")
plt.close()

# === 8. Plot ROC curve ===
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("roc_curve_neat_pca.pdf", format="pdf", bbox_inches="tight")
plt.close()

# === 9. Save winner network visualization and stats ===
visualize.draw_net(config, winner, filename="winner_topology", view=False)
visualize.plot_stats(stats, filename="fitness_stats")
visualize.plot_species(stats, filename="species_evolution")

# === 10. Save metrics to cumulative CSV log ===
summary_csv = "neat_metrics_log.csv"
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data = {
    "datetime": now,
    "generations": generations,
    "population_size": config.pop_size,
    "accuracy": acc,
    "precision": prec,
    "recall": recall,
    "specificity": specificity,
    "f1_score": f1,
    "auc": roc_auc,
    "execution_time_sec": time_neat
}

try:
    df_prev = pd.read_csv(summary_csv)
    df_new = pd.concat([df_prev, pd.DataFrame([data])], ignore_index=True)
except FileNotFoundError:
    df_new = pd.DataFrame([data])

df_new.to_csv(summary_csv, index=False)
print("Summary saved to neat_metrics_log.csv")
