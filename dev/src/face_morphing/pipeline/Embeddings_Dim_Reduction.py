# Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Modules
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding



# ==============================================
# ---            Plotting Tools              ---
# ==============================================
#Tue 07 April 13:25:35 GMT by MAPA
def plot_embedding(X_transformed, labels, title, filename, show_plots = False):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], hue=labels, palette='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Graph '{title}' saved as: {filename}")

    if show_plots :
        plt.show()

#Tue 07 April 13:25:35 GMT by MAPA
def plot_graph_per_demographic_label(reduced_X, labels, dim_reduction_algorithm, output_plot_dir, show_plots=False):
    # Graph for each demographic label
    for col in labels.columns:
        title = f"{dim_reduction_algorithm} Visualization - Colored by {col}"
        plot_embedding(reduced_X, labels[col], title, output_plot_dir + "/" + title.strip()+".png", show_plots)

            

# ==============================================
# ---Dim Reduction Algorithms Implementation ---
# ==============================================
#Tue 07 April 13:25:35 GMT by MAPA
def run_tsne(X, n_components=2, tSNE_perplexity=30, y=None, init='pca', learning_rate='auto', random_state=42, output_plot_dir=None, plot=False, show_plots=False):
    print("Running t-SNE...")
    tsne = TSNE(n_components=n_components, perplexity=tSNE_perplexity, random_state=random_state, init=init, learning_rate=learning_rate)
    X_tsne = tsne.fit_transform(X)

    if plot:
        plot_graph_per_demographic_label(X_tsne, y, "t-SNE", output_plot_dir, show_plots)

    return X_tsne

#Tue 07 April 13:25:35 GMT by MAPA
def run_pca(X, n_components=2, y=None, output_plot_dir=None, plot=False, show_plots=False, **kwargs):
    print("Running PCA...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    if plot:
        plot_graph_per_demographic_label(X_pca, y, "PCA", output_plot_dir, show_plots)

    return X_pca

#Tue 07 April 13:25:35 GMT by MAPA
def run_isomap(X, n_components=2, n_neighbors=5, y=None, output_plot_dir=None, plot=False, show_plots=False, **kwargs):
    print("Running Isomap...")
    isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    X_isomap = isomap.fit_transform(X)

    if plot:
        plot_graph_per_demographic_label(X_isomap, y, "Isomap", output_plot_dir, show_plots)

    return X_isomap

#Tue 07 April 13:25:35 GMT by MAPA
def run_LLE(X, n_components=2, n_neighbors=5, y=None, output_plot_dir=None, plot=False, show_plots=False, **kwargs):
    print("Running LocallyLinearEmbedding...")
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    X_lle = lle.fit_transform(X)

    if plot:
        plot_graph_per_demographic_label(X_lle, y, "LocallyLinearEmbedding", output_plot_dir, show_plots)

    return X_lle

#Tuesday 06 October 2026 13:04:40 GMT by MAPA
DIM_REDUCTION_ALGORITHMS = {
    "tsne":run_tsne,
    "pca":run_pca,
    "isomap":run_isomap,
    "lle":run_LLE
}



# ==============================================
# ---    Dataset Dim Reduction Pipeline      ---
# ==============================================
#Tue 07 April 13:25:35 GMT by MAPA
def Dataset_Dim_Reduction(dataset, output_plot_dir, n_components=2, n_neighbors=10, tSNE_perplexity=30, exec_heavy_algorithms=False, plot=False, show_plots=False, random_state=42):
    # Split dataset
    # X: dataset_embeddings
    X = np.array(dataset['embedding'].tolist())

    # y: demographic_labels
    y = dataset[['Age', 'Dominant_Race', 'Dominant_Gender']]

    print("Extracted data:")
    print("  |   X.shape: ", X.shape)
    print("  |   Y.head : ", y.shape)

    print("\n Running DimReduction Algorithms:")

    # -- Dimen Reduction
    # PCA
    run_pca(X=X, y=y, n_components=n_components, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)
    
    # t-SNE
    run_tsne(X=X, y=y, n_components=n_components, tSNE_perplexity=tSNE_perplexity, random_state=random_state, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)

    if exec_heavy_algorithms:
        # Isomap
        run_isomap(X=X, y=y, n_components=n_components, n_neighbors=n_neighbors, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)

        # LLE
        run_LLE(X=X, y=y, n_components=n_components, n_neighbors=n_neighbors, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)
                
    print("Embedding Dim Reduction completed...")