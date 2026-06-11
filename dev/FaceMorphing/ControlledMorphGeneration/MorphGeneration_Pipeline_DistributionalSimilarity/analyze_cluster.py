# wasserstein distance
from scipy.stats import wasserstein_distance
import itertools

# Libraries
import pandas as pd
import numpy as np
import plotly.express as px


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

    output_path = "./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity/cluster_demographic_summary.html"
    fig.write_html(output_path)
    print(f"Interactive Cluster Summary Plot saved in : {output_path}")

    fig.show()

    return df_summary


# Execute cluster analysis
def analyze_cluster(embeddings_and_demographics_dataset):

    # SHow clusters summary for manual selection
    df_cluster_summary = plot_cluster_summary(embeddings_and_demographics_dataset)

    #Sun 24 May 20:47:30 GMT by MAPA
    # --- Distributional Similarity Analysis Wasserstein distance (distributional compatibility between manifold regions W(Ci​,Cj​))
    df_cluster_w = compute_cluster_wasserstein(embeddings_and_demographics_dataset)

    df_cluster_w.to_csv(
        "../data/ManifoldAnalysis/cluster_wasserstein.csv",
        index=False
    )

    # Plot similarity heatmap
    plot_cluster_wasserstein_heatmap(df_cluster_w)

    return df_cluster_w