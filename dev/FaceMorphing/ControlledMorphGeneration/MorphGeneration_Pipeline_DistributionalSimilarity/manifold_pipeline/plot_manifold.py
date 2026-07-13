import matplotlib.pyplot as plt
import plotly.express as px


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
    plt.savefig("./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity_PercentileThreshold/hdbscan_clusters_visualization.png", dpi=300)
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

    # Save plot as html
    output_path = "./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity_PercentileThreshold/interactive_hdbscan_manifold.html"
    fig.write_html(output_path)
    print(f"Interactive plot saved in : {output_path}")

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

    output_path = f"./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity_PercentileThreshold/hdbscan_by_{demographic_col}.html"
    fig.write_html(output_path)
    print(f"Interactive plot for demographic_feature: ({demographic_col}) saved in: {output_path}")

    fig.show()


# Plot manifold
def plot_manifold(X_tsne, cluster_labels, dataset):
    #Sun 24 May 20:47:30 GMT by MAPA
    # Plot clusters
    plot_hdbscan_clusters(X_tsne, cluster_labels)

    # Plot interactive clusters viualization
    plot_interactive_hdbscan(dataset)

    # Plot interactive clusters visualization by demographic features
    plot_by_demographic(dataset, demographic_col="Age")
    plot_by_demographic(dataset, demographic_col="Dominant_Race")
    plot_by_demographic(dataset, demographic_col="Dominant_Gender")