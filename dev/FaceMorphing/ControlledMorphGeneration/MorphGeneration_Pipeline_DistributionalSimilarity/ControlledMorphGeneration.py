# Libraries
import pandas as pd
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from argparse import Namespace
from scipy.spatial.distance import cdist
from libs import LIB_FaceMorph, LIB_MorphGAN
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# Wed 11 June 2026 by MAPA
def process_cluster_generation(
        cluster_pairs_df,
        cluster_id,
        output_dir_path="./results",
        alpha=0.5
    ):

    # Create output directory if it does not exist
    os.makedirs(
        output_dir_path,
        exist_ok=True
    )

    print(
        f"\nGenerating cluster summary for cluster {cluster_id}"
    )

    generated_samples = []

    # =====================================================
    # Generate all morphs for the current cluster
    # =====================================================
    for idx, row in cluster_pairs_df.iterrows():

        temp_morph_path = os.path.join(
            output_dir_path,
            f"cluster_{cluster_id}_temp_{idx}.png"
        )

        params = Namespace(
            Sb1=row["file_1"],
            Sb2=row["file_2"],
            Morph=temp_morph_path,
            Alpha=alpha
        )

        try:

            LIB_MorphGAN.MorphFace(params)

            generated_samples.append({

                "row": row,

                "morph_path": temp_morph_path

            })

        except Exception as e:

            print(
                f"Error generating morph for pair {idx}: {e}"
            )

    # Validate generated samples
    if len(generated_samples) == 0:

        print(
            f"No morphs generated for cluster {cluster_id}"
        )

        return

    # =====================================================
    # Create summary figure
    # =====================================================
    n_experiments = len(generated_samples)

    fig, axes = plt.subplots(
        3,
        n_experiments,
        figsize=(6 * n_experiments, 16)
    )

    # Fix indexing when there is only one experiment
    if n_experiments == 1:

        axes = np.array(axes).reshape(3, 1)

    # =====================================================
    # Populate figure
    # =====================================================
    for col, sample in enumerate(generated_samples):

        row = sample["row"]

        try:

            img1 = mpimg.imread(
                row["file_1"]
            )

            img2 = mpimg.imread(
                row["file_2"]
            )

            morph_img = mpimg.imread(
                sample["morph_path"]
            )

            # ==========================================
            # Row 1 : Source Image 1
            # ==========================================
            axes[0, col].imshow(img1)

            axes[0, col].set_title(

                f"Source 1\n"
                f"{row['Race_1']} | {row['Gender_1']}\n"
                f"Age: {row['Age_1']}\n"
                f"{os.path.basename(row['file_1'])}",

                fontsize=8

            )

            axes[0, col].axis("off")

            # ==========================================
            # Row 2 : Source Image 2
            # ==========================================
            axes[1, col].imshow(img2)

            axes[1, col].set_title(

                f"Source 2\n"
                f"{row['Race_2']} | {row['Gender_2']}\n"
                f"Age: {row['Age_2']}\n"
                f"{os.path.basename(row['file_2'])}",

                fontsize=8

            )

            axes[1, col].axis("off")

            # ==========================================
            # Row 3 : Morph Result
            # ==========================================
            axes[2, col].imshow(morph_img)

            axes[2, col].set_title(

                f"Morph Result\n"
                f"Type: {row['pair_type']}\n"
                f"t-SNE Dist: {row['tsne_dist']:.4f}",

                fontsize=8

            )

            axes[2, col].axis("off")

        except Exception as e:

            print(
                f"Error creating panel for pair {col}: {e}"
            )

    # =====================================================
    # Add row labels
    # =====================================================
    axes[0, 0].set_ylabel(
        "SOURCE 1",
        fontsize=16,
        fontweight="bold"
    )

    axes[1, 0].set_ylabel(
        "SOURCE 2",
        fontsize=16,
        fontweight="bold"
    )

    axes[2, 0].set_ylabel(
        "MORPH RESULT",
        fontsize=16,
        fontweight="bold"
    )

    # =====================================================
    # Global title
    # =====================================================
    closest_count = len(
        cluster_pairs_df[
            cluster_pairs_df["pair_type"] == "closest"
        ]
    )

    farthest_count = len(
        cluster_pairs_df[
            cluster_pairs_df["pair_type"] == "farthest"
        ]
    )

    fig.suptitle(

        f"Controlled Morph Generation Analysis\n"
        f"Cluster {cluster_id}\n"
        f"Closest Pairs: {closest_count} | "
        f"Farthest Pairs: {farthest_count}\n"
        f"Alpha: {alpha}",

        fontsize=20,
        fontweight="bold"

    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.95]
    )

    # =====================================================
    # Save summary figure
    # =====================================================
    output_path = os.path.join(

        output_dir_path,

        f"cluster_{cluster_id}_summary.png"

    )

    plt.savefig(

        output_path,

        dpi=200,

        bbox_inches="tight"

    )

    plt.close(fig)

    print(
        f"Saved cluster summary: {output_path}"
    )

    # =====================================================
    # Remove temporary morph files
    # =====================================================
    for sample in generated_samples:

        try:

            if os.path.exists(
                sample["morph_path"]
            ):

                os.remove(
                    sample["morph_path"]
                )

        except Exception as e:

            print(
                f"Error removing temp file: {e}"
            )

#Wed 11 June 14:08:13 GMT by MAPA
def load_manifold_dataset(path):

    dataset = pd.read_csv(path)

    print(f"Loaded {len(dataset)} samples")

    return dataset

#Wed 11 June 14:08:13 GMT by MAPA
def get_top_clusters(dataset, n_clusters=5):

    cluster_sizes = (
        dataset[dataset["cluster"] != -1]
        .groupby("cluster")
        .size()
        .sort_values(ascending=False)
    )

    return cluster_sizes.head(n_clusters).index.tolist()

#Wed 11 June 14:08:13 GMT by MAPA
def get_clean_cluster(
        dataset,
        cluster_id,
        min_prob=0.80
    ):

    subset = dataset[
        dataset["cluster"] == cluster_id
    ].copy()

    dominant_race = (
        subset["Dominant_Race"]
        .mode()[0]
    )

    dominant_gender = (
        subset["Dominant_Gender"]
        .mode()[0]
    )

    subset = subset[
        (subset["Dominant_Race"] == dominant_race)
        &
        (subset["Dominant_Gender"] == dominant_gender)
        &
        (subset["cluster_prob"] >= min_prob)
    ]

    return subset

#Tue 30 June 19:02:45 GMT by MAPA
def analyze_neighbor_pairs(cluster_df):
    # Extract 2D t-SNE coordinates
    points = cluster_df[["tsne_x","tsne_y"]].values

    # Fit Nearest Neighbor model using 2 neighbors in order to get only 
    # each point related with it's nearest neighbor
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(points)

    # Compute nearest neighbor distances
    distances, indices = nn.kneighbors(points)

    rows = []

    # Store nearest neighbor information
    for i in range(len(cluster_df)):

        j = indices[i,1]

        rows.append({
            "idx1": i,
            "idx2": j,
            "distance": distances[i,1]
        })

    return pd.DataFrame(rows)

#Tue 30 June 19:02:45 GMT by MAPA
def estimate_min_tsne_distance(cluster_df, percentile):
    # Compute all nearest neighbor distances
    neighbor_pairs = analyze_neighbor_pairs(cluster_df)

    # Estimate minimum valid t-SNE distance based on specified percentile from neighbor_pairs distance distribution
    threshold = np.percentile(neighbor_pairs["distance"], percentile)

    print(f"Automatic minimum t-SNE distance: {threshold:.4f}")

    return threshold, neighbor_pairs

#Tue 30 June 19:02:45 GMT by MAPA
def create_cluster_histogram(pairs, threshold, filepath, percentile):
    # Create histogram figure
    plt.figure(figsize=(10,6))

    # Plot nearest neighbor distance distribution
    plt.hist(
        pairs["distance"],
        bins=40,
        edgecolor="black"
    )

    # Display selected percentile threshold
    plt.axvline(
        threshold,
        color="red",
        linewidth=3,
        label=f"{percentile}th percentile = {threshold:.4f}"
    )

    # Configure plot labels
    plt.xlabel("Nearest Neighbor t-SNE Distance")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Nearest Neighbor Distances. Saved in: {filepath}")
    plt.legend()

    # Save histogram
    plt.savefig(filepath)

    #plt.show()

#Wed 11 June 14:08:13 GMT by MAPA
def generate_cluster_pairs(
        cluster_df,
        n_pairs=5,
        mode="closest",
        min_tsne_distance=0
    ):

    points = cluster_df[
        ["tsne_x", "tsne_y"]
    ].values

    dist_matrix = cdist(
        points,
        points,
        metric="euclidean"
    )

    np.fill_diagonal(
        dist_matrix,
        np.inf if mode=="closest" else -np.inf
    )

    candidate_pairs = []

    for i in range(len(cluster_df)):
        for j in range(i+1, len(cluster_df)):

            d = dist_matrix[i, j]

            # Avoid "identic" samples
            if mode == "closest":

                if d < min_tsne_distance:
                    continue

            # age_diff = abs(
            #     cluster_df.iloc[i]["Age"] -
            #     cluster_df.iloc[j]["Age"]
            # )

            candidate_pairs.append((i,j, d)
            )

    if mode == "closest":
        candidate_pairs.sort(
            key=lambda x:x[2]
        )

    else:
        candidate_pairs.sort(
            key=lambda x:x[2],
            reverse=True
        )

    selected = []
    used = set()

    for i,j,d in candidate_pairs:

        if i in used:
            continue

        if j in used:
            continue

        used.add(i)
        used.add(j)

        selected.append((i,j,d))

        if len(selected) >= n_pairs:
            break

    pairs = []

    for i,j,d in selected:

        s1 = cluster_df.iloc[i]
        s2 = cluster_df.iloc[j]

        pairs.append({

            "cluster": s1["cluster"],

            "file_1": s1["file"],
            "file_2": s2["file"],

            "Age_1": s1["Age"],
            "Age_2": s2["Age"],

            "Race_1": s1["Dominant_Race"],
            "Race_2": s2["Dominant_Race"],

            "Gender_1": s1["Dominant_Gender"],
            "Gender_2": s2["Dominant_Gender"],

            "cluster_prob_1":
                s1["cluster_prob"],

            "cluster_prob_2":
                s2["cluster_prob"],

            "tsne_dist": d,

            "pair_type": mode
        })

    return pd.DataFrame(pairs)

#Wed 11 June 14:08:13 GMT by MAPA
def ControlledMorphGeneration(manifold_dataset_clustered_path = "../data/ManifoldAnalysis/manifold_dataset.csv"):
    print("\n" + "\033[0;34m" + "[Loading manifold clustered dataset...] " + str(start) + "\033[0m")
    # Load clustered manifold dataset
    dataset = load_manifold_dataset(manifold_dataset_clustered_path)

    print("\n" + "\033[0;34m" + "[Extracting top Clusters...] " + str(start) + "\033[0m")
    # Get top populated clusters
    top_clusters = get_top_clusters(dataset, n_clusters=5)

    print("Selected clusters:",top_clusters)

    # Store generated pairs from clusters
    all_pairs = []

    # Controlled Morph generation dir base path
    controlled_morph_generation_dir_base_path = "../data/DistributionalSim_ControlledMorphGeneration_MorphPairs"

    print("\n" + "\033[0;34m" + "[Cleaning Top Clusters...] " + str(start) + "\033[0m")
    # Clean top cluters
    for cluster_id in top_clusters:

        print(f"\nProcessing cluster {cluster_id}")

        # Remove low-confidence samples and demographic inconsistencies
        cluster_df = get_clean_cluster(
            dataset,
            cluster_id,
            min_prob=0.80
        )

        print(f"Clean samples: {len(cluster_df)}")

        # Skip clusters with insufficient samples
        if len(cluster_df) < 20:
            print("Skipping cluster")
            continue
        
        # Estimate adaptive t-SNE distance threshold
        percentile = 50
        
        # Compute distance threshold for image filtering 
        min_distance, neighbor_pairs = estimate_min_tsne_distance(cluster_df, percentile)
        
        # Get closest pairs in cluster
        closest_pairs = generate_cluster_pairs(
            cluster_df,
            n_pairs=5,
            mode="closest",
            min_tsne_distance= min_distance
        )

        # Get farthest pairs in cluster
        farthest_pairs = generate_cluster_pairs(
            cluster_df,
            n_pairs=5,
            mode="farthest",
            min_tsne_distance= min_distance
        )

        # Get final cluster pairs
        cluster_pairs = pd.concat(
            [
                closest_pairs,
                farthest_pairs
            ],
            ignore_index=True
        )

        # Analyze cluster pairs
        neighbor_pairs = analyze_neighbor_pairs(cluster_df)

        # Visualize nearest-neighbor distance distribution
        create_cluster_histogram(neighbor_pairs, min_distance, "./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity/results" + f"/cluster_{cluster_id}_histogram.png", percentile)

        # Save cluster pairs into csv
        cluster_pairs.to_csv(controlled_morph_generation_dir_base_path + f"/cluster_{cluster_id}_pairs.csv",index=False)

        # Store cluster pairs for final export
        all_pairs.append(cluster_pairs)

        # Generate morph images and summary visualization
        process_cluster_generation(
            cluster_pairs_df=cluster_pairs,
            cluster_id=cluster_id,
            output_dir_path="./ControlledMorphGeneration/MorphGeneration_Pipeline_DistributionalSimilarity/results",
            alpha=0.5
        )

    # Concat pairs to generate
    final_df = pd.concat(
        all_pairs,
        ignore_index=True
    )

    # Save final sample pairs for generation
    final_df.to_csv(
        controlled_morph_generation_dir_base_path + f"/all_morph_pairs.csv",
        index=False
    )

    print(f"\nGenerated {len(final_df)} pairs.")

    return final_df


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n")
    
    # Run controlled morph generation pipeline
    ControlledMorphGeneration("../data/ManifoldAnalysis/manifold_dataset.csv")
    
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n")

    exectime = end - start
    print(f"Total Execution Time: {exectime.total_seconds():.2f} seconds")