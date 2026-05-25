import pandas as pd
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import hdbscan


#Wed 13 May 22:26:55 GMT by MAPA


def create_morph_pairs(X_subset, df_subset, mode='dense'):
    """
    Takes a subset of images and pairs them up.
    'dense' mode: pairs images with their nearest neighbors in the subset.
    'extreme' mode: pairs images with their furthest neighbors in the subset.
    """
    indices_pool = list(range(len(X_subset)))
    pairs = []
    
    # Calculate distance matrix for the subset
    dist_matrix = cdist(X_subset, X_subset, metric='euclidean')
    
    used_indices = set()
    
    for i in range(len(X_subset)):
        if i in used_indices:
            continue
        
        # Mask already used indices and the current index itself
        mask = np.ones(len(X_subset), dtype=bool)
        mask[list(used_indices)] = False
        mask[i] = False
        
        if not np.any(mask):
            break
            
        available_indices = np.where(mask)[0]
        distances_from_i = dist_matrix[i, available_indices]
        
        if mode == 'dense':
            # Pair with the closest available sample
            best_match_idx = available_indices[np.argmin(distances_from_i)]
        else:
            # Pair with the furthest available sample (Opposite side of cloud)
            best_match_idx = available_indices[np.argmax(distances_from_i)]
            
        used_indices.add(i)
        used_indices.add(best_match_idx)
        
        # Get data for both samples
        s1 = df_subset.iloc[i]
        s2 = df_subset.iloc[best_match_idx]
        
        pairs.append({
            'file_1': s1['file'], 'Age_1': s1['Age'], 'Race_1': s1['Dominant_Race'], 'Gender_1': s1['Dominant_Gender'],
            'file_2': s2['file'], 'Age_2': s2['Age'], 'Race_2': s2['Dominant_Race'], 'Gender_2': s2['Dominant_Gender'],
            'tsne_dist': dist_matrix[i, best_match_idx]
        })
        
    return pd.DataFrame(pairs)

def get_candidates_and_pair(X_tsne, df_original, n_samples=200, mode='dense'):
    """
    Identifies the candidate pool and returns a DataFrame of 100 pairs.
    """
    n_samples = min(n_samples, len(X_tsne))
    
    if mode == 'dense':
        # Find tightest neighborhood
        nn = NearestNeighbors(n_neighbors=n_samples, metric='euclidean')
        nn.fit(X_tsne)
        distances, idxs = nn.kneighbors(X_tsne)
        anchor_idx = distances.mean(axis=1).argmin()
        selected_indices = idxs[anchor_idx]
    else:
        # Find periphery (furthest from centroid)
        centroid = np.mean(X_tsne, axis=0)
        dist_to_center = np.linalg.norm(X_tsne - centroid, axis=1)
        selected_indices = np.argsort(dist_to_center)[-n_samples:]
    
    X_subset = X_tsne[selected_indices]
    df_subset = df_original.iloc[selected_indices].copy()
    
    # Generate paired dataframe
    paired_df = create_morph_pairs(X_subset, df_subset, mode=mode)
    return paired_df, selected_indices

def plot_comparison_regions(X_tsne, dense_idx, extreme_idx):
    plt.figure(figsize=(14, 9))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='lightgrey', s=5, label='Full Dataset', alpha=0.4)
    plt.scatter(X_tsne[dense_idx, 0], X_tsne[dense_idx, 1], c='red', s=25, label='Dense Pool (Similar Pairs)', edgecolors='black')
    plt.scatter(X_tsne[extreme_idx, 0], X_tsne[extreme_idx, 1], c='blue', s=25, label='Extreme Pool (Distant Pairs)', edgecolors='black')
    
    plt.title('t-SNE Pairing Analysis: Similar Clusters vs Peripheral Extremes', fontsize=16)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig("./ControlledMorphGeneration/morphing_pairing_visualization.png", dpi=300)
    plt.show()

def plot_hdbscan_clusters(X_tsne, cluster_labels):
    plt.figure(figsize=(12,10))

    scatter = plt.scatter(
        X_tsne[:,0],
        X_tsne[:,1],
        c=cluster_labels,
        cmap='tab20',
        s=8
    )

    plt.title("HDBSCAN Clusters on t-SNE Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.colorbar(scatter)
    plt.savefig("./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity/hdbscan_clusters_visualization.png", dpi=300)
    plt.show()

def main():
    print("Loading dataset...") 
    dataset = pd.read_csv('../data/Embeddings_Demographics/ffhq_real_embeddings_and_demographics.csv', converters={'embedding': json.loads})
    X = np.stack(dataset['embedding'].values)

    print("-- Executing t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X)

    dataset['tsne_x'] = X_tsne[:,0]
    dataset['tsne_y'] = X_tsne[:,1]

    # HDBSCAN
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

    print(dataset['cluster'].value_counts())

    # Plot clusters
    plot_hdbscan_clusters(X_tsne, cluster_labels)


    # KDE Density
    print("-- Computing local densities...")
    





    # Generate Pairs
    # print("-- Creating Dense Pairs (Similar faces)...")
    # df_dense_pairs, dense_idx = get_candidates_and_pair(X_tsne, dataset, n_samples=200, mode='dense')
    
    # print("-- Creating Extreme Pairs (Opposite faces)...")
    # df_extreme_pairs, extreme_idx = get_candidates_and_pair(X_tsne, dataset, n_samples=200, mode='extreme')

    # Visuals
    # plot_comparison_regions(X_tsne, dense_idx, extreme_idx)

    # # Save to CSV
    # df_dense_pairs.to_csv("../data/ControlledMorphGeneration_MorphPairs/morph_pairs_DENSE_SIMILAR.csv", index=False)
    # df_extreme_pairs.to_csv("../data/ControlledMorphGeneration_MorphPairs/morph_pairs_EXTREME_DISTANT.csv", index=False)
    
    # print("\n" + "="*50)
    # print(f"DONE: Generated 100 pairs for both scenarios.")
    # print(f"Mean t-SNE distance (Dense): {df_dense_pairs['tsne_dist'].mean():.4f}")
    # print(f"Mean t-SNE distance (Extreme): {df_extreme_pairs['tsne_dist'].mean():.4f}")
    # print("="*50)

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )
