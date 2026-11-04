# Libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json, hashlib, os, umap
import pandas as pd

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

#Tue 03 November 18:33:58 GMT by MAPA
def remove_duplicate_embeddings(
    embedding_json_path,
    output_json_path=None,
    duplicates_csv_path=None
):
    """
    Removes records with exactly identical embeddings.

    Keeps the first occurrence of each embedding.
    Does not delete original images.
    """

    with open(embedding_json_path, "r") as f:
        data = json.load(f)

    seen_embeddings = {}
    unique_data = []
    duplicates = []

    for item in data:

        embedding = item.get("embedding")

        if not embedding:
            continue

        embedding_hash = hashlib.sha256(
            json.dumps(
                embedding,
                separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()

        if embedding_hash in seen_embeddings:

            original = seen_embeddings[embedding_hash]

            duplicates.append({
                "duplicate_image": item.get("image_path"),
                "original_image": original.get("image_path"),
                "embedding_hash": embedding_hash
            })

        else:
            seen_embeddings[embedding_hash] = item
            unique_data.append(item)

    if output_json_path is None:
        base, ext = os.path.splitext(embedding_json_path)
        output_json_path = f"{base}_deduplicated{ext}"

    with open(output_json_path, "w") as f:
        json.dump(unique_data, f, indent=4)

    duplicates_df = pd.DataFrame(duplicates)

    if duplicates_csv_path is None:
        duplicates_csv_path = os.path.splitext(
            output_json_path
        )[0] + "_duplicates.csv"

    duplicates_df.to_csv(duplicates_csv_path, index=False)

    print("\n=== EMBEDDING DEDUPLICATION ===")
    print("Original embeddings:", len(data))
    print("Unique embeddings:", len(unique_data))
    print("Duplicates removed:", len(duplicates))
    print("Output JSON:", output_json_path)
    print("Duplicates report:", duplicates_csv_path)

    return output_json_path, unique_data, duplicates_df            

# ==============================================
# ---Dim Reduction Algorithms Implementation ---
# ==============================================
#Tue 07 April 13:25:35 GMT by MAPA
def run_tsne(X, n_components=2, tSNE_perplexity=30, y=None, init='pca', learning_rate='auto', random_state=42, output_plot_dir=None, plot=False, show_plots=False):
    print("Running t-SNE...")
    # Clean nan numbers
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    tsne = TSNE(n_components=n_components, perplexity=tSNE_perplexity, random_state=random_state, init=init, learning_rate=learning_rate, method='barnes_hut', n_jobs=1)
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

#Tue 04 November 09:53:58 GMT by MAPA
def run_umap(X, n_components=2, y=None, random_state=42, output_plot_dir=None, plot=False, show_plots=False):
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state, n_jobs=1)
    X_umap = reducer.fit_transform(X)

    if plot:
        plot_graph_per_demographic_label(X_umap, y, "UMAP", output_plot_dir, show_plots)

    return X_umap

#Tuesday 06 October 2026 13:04:40 GMT by MAPA
DIM_REDUCTION_ALGORITHMS = {
    "tsne":run_tsne,
    "pca":run_pca,
    "isomap":run_isomap,
    "lle":run_LLE,
    "umap":run_umap
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

    try:
        # t-SNE
        run_tsne(X=X, y=y, n_components=n_components, tSNE_perplexity=tSNE_perplexity, random_state=random_state, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)
    except Exception as e:
        print("  t-sne failed with this error: ", e)

    # UMAP
    run_umap(X, n_components=n_components, y=y, random_state=random_state, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)

    if exec_heavy_algorithms:
        # Isomap
        run_isomap(X=X, y=y, n_components=n_components, n_neighbors=n_neighbors, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)

        # LLE
        run_LLE(X=X, y=y, n_components=n_components, n_neighbors=n_neighbors, output_plot_dir=output_plot_dir, plot=plot, show_plots=show_plots)
                
    print("Embedding Dim Reduction completed...")