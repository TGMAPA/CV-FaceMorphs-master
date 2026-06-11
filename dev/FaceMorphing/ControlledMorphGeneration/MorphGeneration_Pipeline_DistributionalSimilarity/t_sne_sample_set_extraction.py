# Libraries
import pandas as pd
import numpy as np
import json
import datetime

# Plots
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# SKlearn tools
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

# Clustering
import hdbscan

# wasserstein distance
from scipy.stats import wasserstein_distance
import itertools


#Wed 13 May 22:26:55 GMT by MAPA  -- template - missing overwrite for actual pipeline
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

#Wed 13 May 22:26:55 GMT by MAPA    -- template - missing overwrite for actual pipeline
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



#Wed 13 May 22:26:55 GMT by MAPA
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

#Wed 13 May 22:26:55 GMT by MAPA
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

#Sun 24 May 20:47:30 GMT by MAPA
# Plot HDBSCAN clusters with demographic data
def plot_interactive_hdbscan(dataset):

    fig = px.scatter(
        dataset,
        x='tsne_x',
        y='tsne_y',

        color='cluster',

        hover_data=[
            'file',
            'Dominant_Race',
            'Dominant_Gender',
            'Age',
            'cluster_prob'
        ],

        title='t-SNE - HDBSCAN Demographic Manifold',

        opacity=0.8,

        width=1200,
        height=900
    )

    fig.update_traces(
        marker=dict(size=6)
    )

    fig.show()

#Sun 24 May 20:47:30 GMT by MAPA
# Plot HDBSCAN clusters by demographic data
def plot_by_demographic(dataset, demographic_col):

    fig = px.scatter(
        dataset,
        x='tsne_x',
        y='tsne_y',

        color=demographic_col,

        hover_data=[
            'cluster',
            'file',
            'cluster_prob'
        ],

        title=f't-SNE colored by {demographic_col}',

        width=1200,
        height=900
    )

    fig.show()

#Sun 24 May 21:54:40 GMT by MAPA
# Execute Wasserstein distance with HDBSCAN clusters
def compute_cluster_wasserstein(dataset):

    clusters = sorted(dataset['cluster'].unique())

    # Remove outliers
    clusters = [c for c in clusters if c != -1]

    results = []

    for c1, c2 in itertools.combinations(clusters, 2):

        data1 = dataset[dataset['cluster'] == c1]
        data2 = dataset[dataset['cluster'] == c2]

        # Wasserstein in X
        wx = wasserstein_distance(
            data1['tsne_x'],
            data2['tsne_x']
        )

        # Wasserstein in Y
        wy = wasserstein_distance(
            data1['tsne_y'],
            data2['tsne_y']
        )

        # Combined score
        w_total = np.sqrt(wx**2 + wy**2)

        results.append({
            'cluster_a': c1,
            'cluster_b': c2,
            'wasserstein_x': wx,
            'wasserstein_y': wy,
            'wasserstein_total': w_total,

            'size_a': len(data1),
            'size_b': len(data2)
        })

    return pd.DataFrame(results)

#Sun 24 May 21:54:40 GMT by MAPA
# Plot clusters W Distance with a heatmap
def plot_cluster_wasserstein_heatmap(df_w):

    clusters = sorted(
        list(
            set(df_w['cluster_a']).union(
                set(df_w['cluster_b'])
            )
        )
    )

    matrix = pd.DataFrame(
        np.nan,
        index=clusters,
        columns=clusters
    )

    for _, row in df_w.iterrows():

        a = row['cluster_a']
        b = row['cluster_b']
        w = row['wasserstein_total']

        matrix.loc[a,b] = w
        matrix.loc[b,a] = w

    np.fill_diagonal(matrix.values, 0)

    fig = px.imshow(
        matrix,

        text_auto='.2f',

        color_continuous_scale='Viridis',

        title='Cluster-to-Cluster Wasserstein Distance'
    )

    fig.update_layout(
        width=1000,
        height=900
    )

    fig.show()

#Sun 24 May 22:07:40 GMT by MAPA
# SHow clusters summary
def plot_cluster_summary(dataset):

    summaries = []

    for cluster_id in sorted(dataset['cluster'].unique()):

        if cluster_id == -1:
            continue

        subset = dataset[
            dataset['cluster'] == cluster_id
        ]

        # Race Stats
        race_counts = (
            subset['Dominant_Race']
            .value_counts(normalize=True)
        )

        dominant_race = race_counts.idxmax()

        race_purity = race_counts.max()

        # Gender Stats
        gender_counts = (
            subset['Dominant_Gender']
            .value_counts(normalize=True)
        )

        dominant_gender = gender_counts.idxmax()

        gender_purity = gender_counts.max()

        # Age Stats
        mean_age = subset['Age'].mean()

        std_age = subset['Age'].std()

        min_age = subset['Age'].min()

        max_age = subset['Age'].max()

        summaries.append({

            'cluster': cluster_id,

            'cluster_size': len(subset),

            'dominant_race': dominant_race,

            'race_purity': race_purity,

            'dominant_gender': dominant_gender,

            'gender_purity': gender_purity,

            'mean_age': mean_age,

            'std_age': std_age,

            'age_range':
                f"{min_age} - {max_age}",

            'race_distribution':
                "<br>".join([
                    f"{k}: {v:.2f}"
                    for k,v in race_counts.items()
                ]),

            'gender_distribution':
                "<br>".join([
                    f"{k}: {v:.2f}"
                    for k,v in gender_counts.items()
                ])
        })

    df_summary = pd.DataFrame(summaries)

    # Interactive Plot====
    fig = px.scatter(

        df_summary,

        x='cluster',

        y='race_purity',

        size='cluster_size',

        color='mean_age',

        symbol='dominant_gender',

        hover_data={

            'cluster_size': True,

            'dominant_race': True,

            'dominant_gender': True,

            'gender_purity': ':.2f',

            'mean_age': ':.2f',

            'std_age': ':.2f',

            'age_range': True,

            'race_distribution': True,

            'gender_distribution': True
        },

        title='HDBSCAN Cluster Demographic Composition',

        labels={

            'race_purity': 'Race Purity',

            'cluster': 'Cluster ID',

            'mean_age': 'Mean Age'
        },

        width=1900,

        height=850,

        color_continuous_scale='Turbo'
    )

    fig.update_traces(

        marker=dict(

            sizemode='area',

            opacity=0.85,

            line=dict(
                width=1,
                color='black'
            )
        )
    )

    fig.update_layout(

        template='plotly_white'
    )

    fig.show()

    return df_summary


def main():

    #Wed 13 May 22:26:55 GMT by MAPA
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

    #Sun 24 May 20:47:30 GMT by MAPA
    # Plot clusters
    plot_hdbscan_clusters(X_tsne, cluster_labels)

    # Plot interactive clusters viualization
    plot_interactive_hdbscan(dataset)

    # Plot interactive clusters visualization by demographic features
    plot_by_demographic(dataset, demographic_col="Age")
    plot_by_demographic(dataset, demographic_col="Dominant_Race")
    plot_by_demographic(dataset, demographic_col="Dominant_Gender")

    # SHow clusters summary for manual selection
    df_cluster_summary = plot_cluster_summary(dataset)


    #Sun 24 May 20:47:30 GMT by MAPA

    # --- Distributional Similarity Analysis (distributional compatibility between manifold regions W(Ci​,Cj​))
    print("-- Computing cluster Wasserstein distances...")

    df_cluster_w = compute_cluster_wasserstein(dataset)

    # Plot heatmap
    plot_cluster_wasserstein_heatmap(df_cluster_w)

    



if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )
