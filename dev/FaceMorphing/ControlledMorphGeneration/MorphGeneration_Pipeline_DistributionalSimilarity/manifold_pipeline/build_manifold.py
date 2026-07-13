# Libraries
import os
import pandas as pd
import json
from sklearn.manifold import TSNE
import numpy as np
import hdbscan


# Save manifold as csv
def save_manifold_dataset(dataset, output_path):
    dataset.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")

# Load manifold csv
def load_manifold_dataset(path):

    dataset = pd.read_csv(path)

    return dataset

# Build manifold
def build_manifold(output_path = "../data/ManifoldAnalysis/manifold_dataset.csv"):
    #Wed 13 May 22:26:55 GMT by MAPA
    
    # Verify if content already exists
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f" File already exists in '{output_path}'. Loading data from cache...")
        dataset = load_manifold_dataset(output_path)
        
        # Rebuild variables for output
        X_tsne = dataset[['tsne_x', 'tsne_y']].to_numpy()
        cluster_labels = dataset['cluster'].to_numpy()
        
        return X_tsne, cluster_labels, dataset

    print("No data found, starting manifold execution...")
    
    print("Loading dataset...") 
    dataset = pd.read_csv('../data/Embeddings_Demographics/ffhq_real_embeddings_and_demographics.csv', converters={'embedding': json.loads})
    X = np.stack(dataset['embedding'].values)

    print("-- Executing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    dataset['tsne_x'] = X_tsne[:,0]
    dataset['tsne_y'] = X_tsne[:,1]

    # --- HDBSCAN
    print("-- Executing HDBSCAN clustering...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20,
        min_samples=10
    )

    cluster_labels = clusterer.fit_predict(X_tsne)

    # Add cluster label for each sample
    dataset['cluster'] = cluster_labels

    # hdbscan cluster asignment confidence 
    dataset['cluster_prob'] = clusterer.probabilities_

    # Save hbdscan manifold
    save_manifold_dataset(
        dataset,
        output_path
    )

    return X_tsne, cluster_labels, dataset