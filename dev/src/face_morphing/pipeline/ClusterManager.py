# Libraries
import os, json, hdbscan, itertools
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

# Modules
from face_morphing.pipeline import Embeddings_Dim_Reduction


#Tuesday 06 October 2026 13:04:40 GMT by MAPA
class ClusterManager:
    # ==============================================
    # ---      NoSupML Clustering Algorithms     ---
    # ==============================================
    #Wed 13 May 22:26:55 GMT by MAPA
    @classmethod
    def run_HDBSCAN(cls, X, min_cluster_size=20, min_samples=10):
        print("Running HDBSCAN clustering...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )

        cluster_labels = clusterer.fit_predict(X)

        return cluster_labels, clusterer



    # ==============================================
    # ---  Manifold Operations and Core Methods  ---
    # ==============================================
    #Wed 13 May 22:26:55 GMT by MAPA
    # Save manifold as csv
    @classmethod
    def save_manifold_dataset(cls, dataset, output_path):
        dataset.to_csv(
            output_path,
            index=False
        )

        print(f"Saved: {output_path}")

    #Wed 13 May 22:26:55 GMT by MAPA
    # Load manifold csv
    @classmethod
    def load_manifold_dataset(cls, path):
        dataset = pd.read_csv(path)
        print(f"Loaded {len(dataset)} samples")
        return dataset

    #Wed 13 May 22:26:55 GMT by MAPA
    # Build manifold
    @classmethod
    def build_manifold(
        cls, 
        dataset_path, 
        dim_red_algorithm = "tsne",
        dim_red_algorithm_params = None,
        hdbscan_min_cluster_size=20, 
        hdbscan_min_samples=10,
        clustered_dataset_output_path = "../clustered_dataset.csv"):
        
        # Verify if content already exists
        if os.path.exists(clustered_dataset_output_path) and os.path.getsize(clustered_dataset_output_path) > 0:
            print(f" File already exists in '{clustered_dataset_output_path}'. Loading data from cache...")
            dataset = cls.load_manifold_dataset(clustered_dataset_output_path)
            
            # Rebuild variables for output
            X_tsne = dataset[['tsne_x', 'tsne_y']].to_numpy()
            cluster_labels = dataset['cluster'].to_numpy()
            
            return X_tsne, cluster_labels, dataset

        print("Starting manifold execution...")
        
        print("Loading dataset...") 
        dataset = pd.read_csv(dataset_path, converters={'embedding': json.loads})
        X = np.stack(dataset['embedding'].values)

        # Extract Dim Reduction Algorithm implementations
        DimReductor = Embeddings_Dim_Reduction.DIM_REDUCTION_ALGORITHMS[dim_red_algorithm]

        # Verify params existance
        params = dim_red_algorithm_params or {}

        # Run Dim Reduction
        reduced_X = DimReductor(
            X,
            n_components=2,
            **params
        )

        # Add resultant components for every sample in dataframe
        dataset[f'{dim_red_algorithm}_x'] = reduced_X[:,0]
        dataset[f'{dim_red_algorithm}_y'] = reduced_X[:,1]

        # Run Clustering
        cluster_labels, hdbscan = cls.run_HDBSCAN(
            reduced_X, 
            min_cluster_size=hdbscan_min_cluster_size, 
            min_samples=hdbscan_min_samples
        )

        # Add cluster label for each sample
        dataset['cluster'] = cluster_labels

        # hdbscan confidence on cluster asignment 
        dataset['cluster_prob'] = hdbscan.probabilities_

        # Save hbdscan manifold
        cls.save_manifold_dataset(dataset, clustered_dataset_output_path)

        return reduced_X, cluster_labels, dataset

    #Wed 11 June 14:08:13 GMT by MAPA
    @classmethod
    def get_sample_clusters(cls, dataset, n_clusters=3, mixed_clusters=False, mid_strategy="mean"):
        cluster_sizes = (
            dataset[dataset["cluster"] != -1]
            .groupby("cluster")
            .size()
            .sort_values(ascending=False)
        )

        # - Get top n clusters, mid n clusters and low n clusters

        # Top clusters
        top_clusters = cluster_sizes.head(n_clusters)
        
        if not mixed_clusters:
            return {
                "top": top_clusters,
                "mid": None,
                "bottom": None
            }
        
        # Bottom CLusters
        bottom_clusters = cluster_sizes.tail(n_clusters)

        # Mid clusters
        if mid_strategy == "mean":
            target = cluster_sizes.mean()
        elif mid_strategy == "median":
            target = cluster_sizes.median()
        else:  # position
            mid_start = (len(cluster_sizes) - n_clusters) // 2
            mid_clusters = cluster_sizes.iloc[mid_start : mid_start + n_clusters]
            return {
                "top": top_clusters,
                "mid": mid_clusters,
                "bottom": bottom_clusters
            }

        # Select the n clusters closest to the set mid size
        closest_indices = (cluster_sizes - target).abs().nsmallest(n_clusters).index
        mid_clusters = cluster_sizes.loc[closest_indices]

        return {
            "top": top_clusters,
            "mid": mid_clusters,
            "bottom": bottom_clusters
        }, cluster_sizes

    #Wed 11 June 14:08:13 GMT by MAPA
    @classmethod
    def get_clean_cluster(cls, dataset, cluster_id, min_prob=0.80):
        # Filter cluster according to it's cluster_id
        subset = dataset[dataset["cluster"] == cluster_id].copy()

        # Filter cluster according to it's dominant_race
        dominant_race = (subset["Dominant_Race"].mode()[0])

        # Filter cluster according to it's dominant_gender
        dominant_gender = (subset["Dominant_Gender"].mode()[0])

        # Create a new set with filtered samples. Samples are also filtered with a min_prob of belonging to the referred cluster
        subset = subset[
            (subset["Dominant_Race"] == dominant_race)&
            (subset["Dominant_Gender"] == dominant_gender)&
            (subset["cluster_prob"] >= min_prob)
        ]

        return subset

    #Tue 30 June 19:02:45 GMT by MAPA
    @classmethod
    def analyze_neighbor_pairs(cls, cluster_df):
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



    # ==============================================
    # --- Cluster Plot and Visualization methods ---
    # ==============================================
    #Wed 13 May 22:26:55 GMT by MAPA
    @classmethod
    def plot_hdbscan_clusters(cls, reduced_X, cluster_labels, dim_red_algorithm, output_plot_dir, show_plot=False):
        plt.figure(figsize=(12,10))

        scatter = plt.scatter(
            reduced_X[:,0],
            reduced_X[:,1],
            c=cluster_labels,
            cmap='tab20',
            s=8
        )

        plt.title(f"HDBSCAN Clusters on {dim_red_algorithm} Space")
        plt.xlabel(f"{dim_red_algorithm} 1")
        plt.ylabel(f"{dim_red_algorithm} 2")

        plt.colorbar(scatter)
        plt.savefig(output_plot_dir+ f"/hdbscan_clusters_on_{dim_red_algorithm}_visualization.png", dpi=300)
        if show_plot:
            plt.show()

    #Sun 24 May 20:47:30 GMT by MAPA
    # Plot HDBSCAN clusters with demographic data
    @classmethod
    def plot_interactive_hdbscan(cls, dataset, dim_red_algorithm, output_plot_dir, show_plot=False):
       
        fig = px.scatter(
            dataset,
            x=f'{dim_red_algorithm}_x',
            y=f'{dim_red_algorithm}_y',
            color='cluster',
            hover_data=[
                'file',
                'Dominant_Race',
                'Dominant_Gender',
                'Age',
                'cluster_prob'
            ],
            title=f'{dim_red_algorithm} - HDBSCAN Demographic Manifold',
            opacity=0.8,
            width=1200,
            height=900
        )

        fig.update_traces(marker=dict(size=6))

        # Save plot as html
        output_path = output_plot_dir + f"/hdbscan_clusters_on_{dim_red_algorithm}_interactive.html"
        fig.write_html(output_path)
        print(f"Interactive plot saved in : {output_path}")

        if show_plot:
            plt.show()

    #Sun 24 May 20:47:30 GMT by MAPA
    # Plot HDBSCAN clusters by demographic data
    @classmethod
    def plot_by_demographic(cls, dataset, demographic_col, dim_red_algorithm, output_plot_dir, show_plot=False):

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
            title=f'{dim_red_algorithm} - HDBSCAN Demographic Manifold - colored by {demographic_col}',
            width=1200,
            height=900
        )

        output_path = output_plot_dir + f"/hdbscan_clusters_on_{dim_red_algorithm}_by_{demographic_col}_interactive.html"
        fig.write_html(output_path)
        print(f"Interactive plot for demographic_feature: ({demographic_col}) saved in: {output_path}")
        if show_plot:
            fig.show()

    #Sun 24 May 20:47:30 GMT by MAPA
    # Plot manifold
    @classmethod
    def plot_manifold(cls, X_tsne, cluster_labels, dataset, dim_red_algorithm, output_plot_dir, show_plot):
        # Plot clusters
        cls.plot_hdbscan_clusters(X_tsne, cluster_labels, dim_red_algorithm, output_plot_dir, show_plot)

        # Plot interactive clusters viualization
        cls.plot_interactive_hdbscan(dataset, dim_red_algorithm, output_plot_dir, show_plot)

        # Plot interactive clusters visualization by demographic features
        for col in ["Age", "Dominant_Race", "Dominant_Gender"]:
            cls.plot_by_demographic(
                dataset=dataset, 
                demographic_col=col, 
                dim_red_algorithm=dim_red_algorithm, 
                output_plot_dir=output_plot_dir, 
                show_plot=show_plot
            )

    #Sun 24 May 21:54:40 GMT by MAPA
    # Plot clusters W Distance with a heatmap
    @classmethod
    def plot_cluster_wasserstein_heatmap(cls, df_w, dim_red_algorithm, output_plot_dir, show_plot=False):
        clusters = sorted(list(set(df_w['cluster_a']).union(set(df_w['cluster_b']))))

        # Create empty grid
        matrix = pd.DataFrame(np.nan, index=clusters, columns=clusters)

        # Fill grid
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
            title=f'HDBSCAN Cluster-to-Cluster Wasserstein Distance HeatMap on {dim_red_algorithm} Space'
        )

        fig.update_layout(width=1000, height=900)
        plt.savefig(output_plot_dir+ f"/hdbscan_clusters_on_{dim_red_algorithm}_cluster_wasserstein_heatmap.png", dpi=300)
        
        if show_plot:
            fig.show()

    #Sun 24 May 22:07:40 GMT by MAPA
    # SHow clusters summary
    @classmethod
    def plot_cluster_summary(cls, dataset, output_plot_dir, show_plot):
        summaries = []

        # Iterate over each cluster
        for cluster_id in sorted(dataset['cluster'].unique()):
            if cluster_id == -1:
                continue

            subset = dataset[dataset['cluster'] == cluster_id]

            # Race Stats
            race_counts = (subset['Dominant_Race'].value_counts(normalize=True))
            dominant_race = race_counts.idxmax()
            race_purity = race_counts.max()

            # Gender Stats
            gender_counts = (subset['Dominant_Gender'].value_counts(normalize=True))
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
                'age_range': f"{min_age} - {max_age}",
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

        # Create df
        df_summary = pd.DataFrame(summaries)

        # Interactive Plot
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

        # Plot config
        fig.update_traces(marker=dict(sizemode='area', opacity=0.85, line=dict(width=1,color='black')))
        fig.update_layout(template='plotly_white')

        output_path = output_plot_dir + "/cluster_demographic_summary.html"
        fig.write_html(output_path)
        print(f"Interactive Cluster Summary Plot saved in : {output_path}")

        if show_plot:
            fig.show()

        return df_summary

    #Mon 04 Aug 19:29:50 GMT by MAPA 
    @classmethod
    def create_percentile_summary(
            cls,
            generated_samples,
            cluster_id,
            values,
            best_distribution,
            trust_region,
            output_dir_path
        ):

        # Exit if no morphs were generated
        if len(generated_samples) == 0:
            return

        # Number of experimental morph pairs
        n = len(generated_samples)

        # Figure layout
        fig = plt.figure(figsize=(4*n, 14), constrained_layout=True)

        gs = GridSpec(
            4,
            n,
            height_ratios=[5,2.5,2.5,2.5],
            hspace=0.2,
            wspace=0.025,
            figure=fig
        )

        # Plot fitted distance distribution
        ax_hist = fig.add_subplot(gs[0, :])

        # Recover fitted distribution
        distribution = best_distribution["distribution"]
        params = best_distribution["params"]

        # Generate evaluation points
        x = np.linspace(
            np.min(values),
            np.max(values),
            500
        )

        # Evaluate fitted PDF
        pdf = distribution.pdf(x, *params)

        # Plot observed distance histogram
        ax_hist.hist(
            values,
            bins=30,
            density=True,
            alpha=0.5,
            edgecolor="black",
            label="Observed distances"
        )

        # Plot fitted probability density function
        ax_hist.plot(
            x,
            pdf,
            linewidth=3,
            color="red",
            label=f"{best_distribution['name']} PDF"
        )

        # Highlight trust region
        lower = trust_region["lower"]
        upper = trust_region["upper"]

        mask = (x >= lower) & (x <= upper)

        ax_hist.fill_between(
            x[mask],
            pdf[mask],
            alpha=0.25,
            color="green",
            label=f"{trust_region['confidence']*100:.0f}% Trust Region"
        )

        # Draw confidence interval limits
        ax_hist.axvline(
            lower,
            linestyle="--",
            linewidth=2,
            color="green"
        )

        ax_hist.axvline(
            upper,
            linestyle="--",
            linewidth=2,
            color="green"
        )

        
        # Display Statistics textbox
        stats_text = (
            f"{best_distribution['name']}\n"
            f"KS = {best_distribution['ks_statistic']:.4f}\n"
            f"p = {best_distribution['p_value']:.4f}\n"
            f"AIC = {best_distribution['AIC']:.2f}"
        )

        ax_hist.text(
            0.98,
            0.98,
            stats_text,
            transform=ax_hist.transAxes,
            fontsize=10,
            va="top",
            ha="right",
            bbox=dict(
                facecolor="white",
                alpha=0.95
            )
        )

        # Define percentile colormap
        cmap = mpl.colormaps["viridis"]

        # Mark selected experimental pairs on the distribution
        ymax = pdf.max()

        for sample in generated_samples:
            row = sample["row"]

            color = cmap(row["percentile"])

            # Draw pair location on the fitted distribution
            ax_hist.scatter(
                row["real_distance"],
                ymax*0.03,
                s=120,
                marker="v",
                color=color,
                edgecolor="black",
                zorder=20
            )

            # Annotate corresponding percentile
            ax_hist.text(
                row["real_distance"],
                ymax*0.08,
                f"P{int(row['percentile']*100)}",
                rotation=90,
                ha="center",
                fontsize=8
            )

        # Add percentile colorbar
        norm = mcolors.Normalize(vmin=0.10,vmax=0.90)

        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax_hist, label="Distribution Percentile")

        # Configure distribution plot
        ax_hist.set_xlabel("Nearest Neighbor Distance", fontsize=12)
        ax_hist.set_ylabel("Density", fontsize=12)
        ax_hist.set_title(
            f"Cluster {cluster_id} Distribution Fit",
            fontsize=16,
            fontweight="bold"
        )
        ax_hist.legend()

        # Build image panels
        first_ax1 = None
        first_ax2 = None
        first_ax3 = None

        for col, sample in enumerate(generated_samples):
            row = sample["row"]

            color = cmap(row["percentile"])

            # Load source images and generated morph
            img1 = mpimg.imread(row["file_1"])
            img2 = mpimg.imread(row["file_2"])
            morph = mpimg.imread(sample["morph_path"])

            # Source image 1
            ax1 = fig.add_subplot(gs[1, col])
            ax1.imshow(img1)
            ax1.set_title(
                f"P{int(row['percentile']*100)}\n"
                f"d={row['real_distance']:.3f}",
                fontsize=9
            )
            ax1.axis("off")

            # Source image 2
            ax2 = fig.add_subplot(gs[2, col])
            ax2.imshow(img2)
            ax2.axis("off")

            # Generated morph
            ax3 = fig.add_subplot(gs[3, col])
            ax3.imshow(morph)
            ax3.set_title(
                f"err={row['distance_error']:.3f}",
                fontsize=8
            )
            ax3.axis("off")

            # Highlight images using percentile color
            for ax in [ax1, ax2, ax3]:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(4)
                    spine.set_color(color)

            # Store first column for row labels
            if first_ax1 is None:
                first_ax1 = ax1
                first_ax2 = ax2
                first_ax3 = ax3

        # Store first column for row labels
        first_ax1.set_ylabel("SOURCE 1", fontsize=14, fontweight="bold")
        first_ax2.set_ylabel("SOURCE 2", fontsize=14, fontweight="bold")
        first_ax3.set_ylabel("MORPH",fontsize=14,fontweight="bold")

        # Figure title and configuration
        fig.suptitle(
            f"Controlled Morph Generation Experiment\n"
            f"Cluster {cluster_id}",
            fontsize=20,
            fontweight="bold"
        )
        output_path = os.path.join(output_dir_path, f"cluster_{cluster_id}_percentile_summary.png")
        plt.savefig(output_path, dpi=250, bbox_inches="tight")
        plt.close()

        print(f"Saved summary: {output_path}")

    #Wed 15 July 17:48:50 GMT by MAPA 
    @classmethod
    def plot_fitted_distribution(cls, values, best_distribution, trust_region, filepath, title):
        # Recover scipy distribution
        distribution = best_distribution["distribution"]

        # Recover estimated parameters
        params = best_distribution["params"]

        # Generate x-axis values
        x = np.linspace(np.min(values), np.max(values), 500)

        # Evaluate fitted PDF
        pdf = distribution.pdf(x,*params)

        # Create figure
        plt.figure(figsize=(10,6))

        # Histogram (normalized)
        plt.hist(
            values,
            bins=30,
            density=True,
            alpha=0.6,
            edgecolor="black",
            label="Observed distances"
        )

        # Plot fitted PDF
        plt.plot(
            x,
            pdf,
            linewidth=3,
            label=f"{best_distribution['name']} fit"
        )

        text = (
            f"{best_distribution['name']}\n"
            f"KS = {best_distribution['ks_statistic']:.4f}\n"
            f"p = {best_distribution['p_value']:.4f}\n"
            f"AIC = {best_distribution['AIC']:.2f}"
        )

        plt.text(
            0.98,
            0.98,
            text,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(
                facecolor="white",
                alpha=0.9
            )
        )

        # Trust region
        lower = trust_region["lower"]
        upper = trust_region["upper"]
        mask = (x >= lower) & (x <= upper)

        plt.fill_between(
            x[mask],
            pdf[mask],
            alpha=0.3,
            label=f"{trust_region['confidence']*100:.0f}% Trust Region"
        )

        # Vertical lines
        plt.axvline(
            lower,
            linestyle="--",
            linewidth=2,
            label=f"Lower = {lower:.4f}"
        )

        plt.axvline(
            upper,
            linestyle="--",
            linewidth=2,
            label=f"Upper = {upper:.4f}"
        )

        plt.xlabel("Nearest Neighbor Distance")
        plt.ylabel("Density")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filepath, dpi=250)
        plt.close()

    # Wed 02 Sep 20:13:30 GMT by MAPA
    @classmethod
    def plot_cluster_sizes_distribution(cls, cluster_sizes, path):
        # Create figure 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Standard linear scale distribution
        sns.histplot(
            cluster_sizes,
            bins=30,
            kde=True,
            color="#2b5c8f",
            edgecolor="black",
            ax=axes[0],
        )
        axes[0].set_title("Cluster Size Distribution", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Cluster Size (Number of Samples)", fontsize=10)
        axes[0].set_ylabel("Frequency", fontsize=10)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        # Plot 2: Logarithmic scale distribution
        sns.histplot(
            cluster_sizes,
            bins=30,
            log_scale=True,
            color="#2b5c8f",
            edgecolor="black",
            ax=axes[1],
        )
        axes[1].set_title("Cluster Size Distribution (Log Scale)",fontsize=12,fontweight="bold")
        axes[1].set_xlabel("Cluster Size (Log)", fontsize=10)
        axes[1].set_ylabel("Frequency", fontsize=10)
        axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

        # Adjust layout and save the single figure
        plt.tight_layout()
        plt.savefig(path+"/distribucion_cluster_size_combined.png", dpi=300, bbox_inches="tight")
   


    # ==============================================
    # ---       Cluster analysis methods         ---
    # ==============================================
    #Sun 24 May 21:54:40 GMT by MAPA
    # Execute Wasserstein distance with HDBSCAN clusters
    @classmethod
    def compute_cluster_wasserstein(cls, dataset, dim_red_algorithm="tsne"):
        clusters = sorted(dataset['cluster'].unique())

        # Remove outliers
        clusters = [c for c in clusters if c != -1]

        results = []

        for c1, c2 in itertools.combinations(clusters, 2):
            data1 = dataset[dataset['cluster'] == c1]
            data2 = dataset[dataset['cluster'] == c2]

            # Wasserstein in X
            wx = stats.wasserstein_distance(data1[f'{dim_red_algorithm}_x'], data2[f'{dim_red_algorithm}_x'])

            # Wasserstein in Y
            wy = stats.wasserstein_distance(data1[f'{dim_red_algorithm}_y'],data2[f'{dim_red_algorithm}_y'])

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

    #Sun 24 May 22:07:40 GMT by MAPA
    # Execute cluster analysis
    @classmethod
    def analyze_cluster(cls, embeddings_and_demographics_dataset, dim_red_algorithm, output_analysis_csv_path, output_plot_dir, show_plot=False):
        # SHow clusters summary for manual selection
        df_cluster_summary = cls.plot_cluster_summary(embeddings_and_demographics_dataset, output_plot_dir, show_plot)

        #Sun 24 May 20:47:30 GMT by MAPA
        # --- Distributional Similarity Analysis Wasserstein distance (distributional compatibility between manifold regions W(Ci​,Cj​))
        df_cluster_w = cls.compute_cluster_wasserstein(embeddings_and_demographics_dataset, dim_red_algorithm=dim_red_algorithm)
        df_cluster_w.to_csv(output_analysis_csv_path,index=False)

        # Plot similarity heatmap
        cls.plot_cluster_wasserstein_heatmap(df_cluster_w, dim_red_algorithm, output_plot_dir, show_plot)

        return df_cluster_w

    #Mon 13 July 17:48:50 GMT by MAPA 
    @classmethod
    def fit_candidate_distributions(cls, values, candidate_distributions_dict):
        # Store every distribution type 
        fitted_distributions = []

        # Estimate distribution parameters with scipy (max likelihood)
        for key, distribution in candidate_distributions_dict.items():
            # Get fitted params for each distribution candidate
            params = distribution.fit(values)
            
            # Evaluate Kolmogorov-Smirnov and p-value
            ks_statistic, p_value = stats.kstest(
                values,
                distribution.cdf,
                args=params
            )

            # Compute log likelihood
            log_likelihood = np.sum(distribution.logpdf(values, *params))

            # Compute AIC (Akaike Information Criterion)
            # Evaluates which distribution provides the best balance between model fit and simplicity, penalizing overly complex distributions to prevent overfitting.
            # Using: AIC = 2k - 2ln(L)
            k = len(params)
            aic = 2 * k - 2 * log_likelihood

            # Store fitted model information
            fitted_distributions.append({
                    "name":key,
                    "distribution": distribution,
                    "params": params,
                    "ks_statistic": ks_statistic,
                    "p_value": p_value,
                    "log_likelihood": log_likelihood,
                    "AIC": aic
                }
            )
        
        return pd.DataFrame(fitted_distributions)

    #Wed 15 July 17:48:50 GMT by MAPA 
    @classmethod
    def compute_trust_region(cls, best_distribution, confidence=0.95):
        # Recover fitted scipy distribution
        distribution = best_distribution["distribution"]

        # Recover estimated parameters
        params = best_distribution["params"]

        # Tail probability
        alpha = 1 - confidence

        # Lower confidence bound - Get the value where the desired acum probability ends
        lower = distribution.ppf(alpha / 2, *params)

        # Upper confidence bound - Get the value where the desired acum probability ends
        upper = distribution.ppf(1 - alpha / 2, *params)

        return {
            "distribution": best_distribution["name"],
            "confidence": confidence,
            "lower": lower,
            "upper": upper
        }

    #Mon 04 Aug 13:45:30 GMT by MAPA 
    # Computes representative distances from the fitted probability distribution using its inverse cumulative distribution function (PPF).
    @classmethod
    def sample_distribution_percentiles(cls, best_distribution, percentiles=np.arange(0.10, 1.00, 0.10)):
        # Extract distribution and its parameters
        distribution = best_distribution["distribution"]
        params = best_distribution["params"]

        # Row array for dataframe
        rows = []

        for p in percentiles:
            # Get target distance for each percentile
            target_distance = distribution.ppf(p,*params)

            # Append row
            rows.append({
                "percentile": p,
                "target_distance": target_distance
            })

        # Return dataframe
        return pd.DataFrame(rows)



    # ==============================================
    # ---       Cluster Quality Methods          ---
    # ==============================================
    # Wed 02 Sep 20:13:30 GMT by MAPA 
    # How precisely can we estimate the mean of the neighbor distances for each cluster?
    @classmethod
    def compute_t_student_metrics(cls, distances, confidence=0.95):
        # Convert distances to a numeric NumPy array
        distances = np.asarray(distances, dtype=float)

        # Remove invalid or non-finite distance values
        distances = distances[np.isfinite(distances)]
        n = len(distances)

        # Return empty metrics if there are insufficient samples
        if n < 2:
            return {
                "n": n,
                "mean": np.nan,
                "std": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "ci_width": np.nan
            }

        # Compute sample mean and sample standard deviation
        mean = np.mean(distances)
        std = np.std(distances, ddof=1)

        # Compute significance level from the selected confidence level
        alpha = 1 - confidence

        # Compute critical t-value for the confidence interval
        t_critical = stats.t.ppf(
            1 - alpha / 2,
            df=n - 1
        )

        # Compute standard error of the sample mean
        standard_error = std / np.sqrt(n)

        # Compute confidence interval around the sample mean
        ci_lower = mean - t_critical * standard_error
        ci_upper = mean + t_critical * standard_error

        # Compute total confidence interval width
        ci_width = ci_upper - ci_lower

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "ci_width": ci_width
        }

    # Wed 02 Sep 20:13:30 GMT by MAPA
    # How geometrically extended is the cluster?
    @classmethod
    def compute_compression_metrics(cls, embeddings):
        # Convert embeddings to a numeric NumPy array
        X = np.asarray(embeddings, dtype=float)

        # Return empty metrics if there are insufficient samples
        if len(X) < 2:
            return {
                "centroid_radius_mean": np.nan,
                "centroid_radius_std": np.nan,
                "centroid_radius_median": np.nan,
                "centroid_radius_q90": np.nan,
                "compression_cv": np.nan
            }

        # Compute the geometric centroid of the cluster
        centroid = np.mean(X, axis=0)

        # Compute the Euclidean distance from each embedding to the centroid
        radii = np.linalg.norm(X - centroid, axis=1)

        # Compute descriptive statistics of the centroid distances
        mean_radius = np.mean(radii)
        std_radius = np.std(radii, ddof=1)

        # Compute the coefficient of variation of the centroid distances
        cv = (
            std_radius / mean_radius
            if mean_radius > 0
            else np.nan
        )

        return {
            "centroid_radius_mean": mean_radius,
            "centroid_radius_std": std_radius,
            "centroid_radius_median": np.median(radii),
            "centroid_radius_q90": np.percentile(radii, 90),
            "compression_cv": cv
        }

    # Wed 02 Sep 20:13:30 GMT by MAPA
    # How close are the samples to each other inside a cluster?
    @classmethod
    def compute_density_metrics(cls, neighbor_pairs):
        # Extract nearest-neighbor distances from the pairwise results
        distances = np.asarray(neighbor_pairs["distance"], dtype=float)

        # Remove invalid or non-finite distance values
        distances = distances[np.isfinite(distances)]

        # Return empty metrics if no valid distances are available
        if len(distances) == 0:
            return {
                "mean_knn_distance": np.nan,
                "median_knn_distance": np.nan,
                "knn_density": np.nan
            }

        # Compute mean and median nearest-neighbor distances
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)

        # Estimate local density as the inverse of the mean neighbor distance
        density = (
            1.0 / mean_distance
            if mean_distance > 0
            else np.nan
        )

        return {
            "mean_knn_distance": mean_distance,
            "median_knn_distance": median_distance,
            "knn_density": density
        }

    # Wed 02 Sep 20:13:30 GMT by MAPA
    @classmethod
    def extract_distribution_quality(cls, fitted_results):
        # Select the best-fitting distribution
        best = fitted_results.iloc[0]

        # Extract goodness-of-fit and model selection metrics
        return {
            "best_distribution": best["name"],
            "ks_statistic": best["ks_statistic"],
            "ks_p_value": best["p_value"],
            "aic": best["AIC"]
        }

    # Wed 02 Sep 20:13:30 GMT by MAPA
    # Do the obtained model and parameters hold up when we perturb or resample the data?
    @classmethod
    def bootstrap_distribution_stability(cls, distances, candidate_distributions, n_iterations=100, confidence=0.90, random_state=42):
        # Initialize the random number generator for reproducible sampling
        rng = np.random.default_rng(random_state)

        # Convert distances to a numeric NumPy array
        distances = np.asarray(distances, dtype=float)

        # Remove invalid or non-finite distance values
        distances = distances[np.isfinite(distances)]

        results = []

        n = len(distances)

        # Generate bootstrap samples and evaluate distribution fitting stability
        for i in range(n_iterations):

            # Generate a bootstrap sample by sampling with replacement
            bootstrap_sample = rng.choice(
                distances,
                size=n,
                replace=True
            )

            # Fit the candidate distributions to the bootstrap sample
            fitted = cls.fit_candidate_distributions(
                bootstrap_sample,
                candidate_distributions
            )

            # Rank fitted distributions using KS statistic and AIC
            fitted = fitted.sort_values(
                by=["ks_statistic", "AIC"],
                ascending=[True, True]
            )

            # Remove models with non-finite quality metrics
            fitted = fitted[
                np.isfinite(fitted["ks_statistic"]) &
                np.isfinite(fitted["AIC"])
            ]

            if fitted.empty:
                continue

            # Select the best-fitting distribution
            best = fitted.iloc[0]

            # Compute the trust region for the selected distribution
            trust_region = cls.compute_trust_region(
                best,
                confidence=confidence
            )

            # Store the results from the current bootstrap iteration
            results.append({
                "iteration": i,
                "distribution": best["name"],
                "ks": best["ks_statistic"],
                "p_value": best["p_value"],
                "AIC": best["AIC"],
                "trust_lower": trust_region["lower"],
                "trust_upper": trust_region["upper"]
            })

        return pd.DataFrame(results)

    # Wed 02 Sep 20:13:30 GMT by MAPA
    @classmethod
    def summarize_bootstrap_stability(cls, bootstrap_results):
        n = len(bootstrap_results)

        # Return an empty result if no bootstrap iterations were completed
        if n == 0:
            return {}

        # Compute the proportion of iterations selecting each distribution
        distribution_stability = (bootstrap_results["distribution"].value_counts(normalize=True))

        # Identify the distribution selected most frequently
        dominant_distribution = (distribution_stability.index[0])

        # Compute the stability of the dominant distribution
        model_stability = (distribution_stability.iloc[0])

        # Summarize the stability of the fitted model and trust region
        return {
            "bootstrap_iterations": n,
            "dominant_distribution": dominant_distribution,
            "model_stability": model_stability,
            "mean_ks": bootstrap_results["ks"].mean(),
            "std_ks": bootstrap_results["ks"].std(),
            "mean_aic": bootstrap_results["AIC"].mean(),
            "std_aic": bootstrap_results["AIC"].std(),
            "mean_trust_lower": bootstrap_results["trust_lower"].mean(),
            "mean_trust_upper": bootstrap_results["trust_upper"].mean(),
            "std_trust_lower": bootstrap_results["trust_lower"].std(),
            "std_trust_upper": bootstrap_results["trust_upper"].std()
        }    