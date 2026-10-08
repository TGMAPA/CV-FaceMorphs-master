# Libraries
import datetime, os, ast, torch
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from argparse import Namespace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import stats

# Modules
from face_morphing.pipeline.ClusterManager import ClusterManager
from face_morphing.libs import LIB_MorphGAN # LIB_FaceMorph



# Tuesday 06 Oct 18:11:35 GMT by MAPA
class MorphGenerator:
    # ==============================================
    # ---      Gpu worker implementations        ---
    # ==============================================
    #Wednesday 07 October 2026 12:04:40 GMT by MAPA
    # Worker for pair subset processing (Morph Generation) in a specific GPU
    @classmethod
    def _percentile_gpu_worker(cls, gpu_id, pairs_chunk, cluster_id, output_dir_path, alpha,return_dict):
        device = torch.device(f"cuda:{gpu_id}")
        
        generated_samples = []

        for idx, row in pairs_chunk.iterrows():
            percentile_val = int(row['percentile'] * 100) if 'percentile' in row else 0
            morph_filename = (
                f"cluster_{cluster_id}"
                f"_p{percentile_val:02d}"
                f"_sample_{idx}.png"
            )
            morph_path = os.path.join(output_dir_path, morph_filename)

            params = Namespace(
                Sb1=row["file_1"],
                Sb2=row["file_2"],
                Morph=morph_path,
                Alpha=alpha
            )

            # Run morph generation
            try:
                LIB_MorphGAN.MorphFace(params)
                generated_samples.append({"row": row, "morph_path": morph_path})
            except Exception as e:
                print(f"[GPU {gpu_id} | {str(device)}]: Morph generation error {idx}: {e}")

        return_dict[gpu_id] = generated_samples

    #Wednesday 07 October 2026 12:04:40 GMT by MAPA
    # Cluster generation gpu worker for subset pair morph generation
    @classmethod
    def _cluster_generation_gpu_worker(cls, gpu_id, pairs_chunk, cluster_id, output_dir_path, alpha, model_path, return_dict):
        device = torch.device(f"cuda:{gpu_id}")
        
        generated_samples = []
        for idx, row in pairs_chunk.iterrows():
            temp_morph_path = os.path.join(output_dir_path, f"cluster_{cluster_id}_temp_{idx}.png")
            params = Namespace(Sb1=row["file_1"], Sb2=row["file_2"], Morph=temp_morph_path, Alpha=alpha)
            
            try:
                LIB_MorphGAN.MorphFace(params)
                generated_samples.append({
                    "row": row,
                    "morph_path": temp_morph_path
                })
            except Exception as e:
                print(f"[GPU {gpu_id} | {str(device)}]: Morph generation error {idx}: {e}")

        return_dict[gpu_id] = generated_samples


    # ==============================================
    # ---     Plot and Visualization methods     ---
    # ==============================================
    # Wed 11 June 2026 by MAPA
    @classmethod
    def plot_cluster_generation(cls, cluster_id, generated_samples, alpha, cluster_pairs_df, output_dir_path):
        # Create summary figure
        n_experiments = len(generated_samples)
        fig, axes = plt.subplots(
            3,
            n_experiments,
            figsize=(6 * n_experiments, 16)
        )

        # Fix indexing when there is only one experiment
        if n_experiments == 1:
            axes = np.array(axes).reshape(3, 1)

        # Populate figure
        for col, sample in enumerate(generated_samples):
            row = sample["row"]

            try:
                img1 = mpimg.imread(row["file_1"])
                img2 = mpimg.imread(row["file_2"])

                morph_img = mpimg.imread(sample["morph_path"])

                # Row 1 : Source Image 1
                axes[0, col].imshow(img1)
                axes[0, col].set_title(
                    f"Source 1\n"
                    f"{row['Race_1']} | {row['Gender_1']}\n"
                    f"Age: {row['Age_1']}\n"
                    f"{os.path.basename(row['file_1'])}",
                    fontsize=8
                )
                axes[0, col].axis("off")

                # Row 2 : Source Image 2
                axes[1, col].imshow(img2)
                axes[1, col].set_title(
                    f"Source 2\n"
                    f"{row['Race_2']} | {row['Gender_2']}\n"
                    f"Age: {row['Age_2']}\n"
                    f"{os.path.basename(row['file_2'])}",
                    fontsize=8
                )
                axes[1, col].axis("off")

                # Row 3 : Morph Result
                axes[2, col].imshow(morph_img)
                axes[2, col].set_title(
                    f"Morph Result\n"
                    f"Type: {row['pair_type']}\n"
                    f"t-SNE Dist: {row['tsne_dist']:.4f}",
                    fontsize=8
                )
                axes[2, col].axis("off")

            except Exception as e:
                print(f"Error creating panel for pair {col}: {e}")

        # Add row labels
        axes[0, 0].set_ylabel("SOURCE 1", fontsize=16, fontweight="bold")
        axes[1, 0].set_ylabel("SOURCE 2", fontsize=16, fontweight="bold")
        axes[2, 0].set_ylabel("MORPH RESULT", fontsize=16, fontweight="bold")

        # Global title
        closest_count = len(cluster_pairs_df[cluster_pairs_df["pair_type"] == "closest"])
        farthest_count = len(cluster_pairs_df[cluster_pairs_df["pair_type"] == "farthest"])

        fig.suptitle(
            f"Controlled Morph Generation Analysis\n"
            f"Cluster {cluster_id}\n"
            f"Closest Pairs: {closest_count} | "
            f"Farthest Pairs: {farthest_count}\n"
            f"Alpha: {alpha}",
            fontsize=20,
            fontweight="bold"
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save summary figure
        output_path = os.path.join(output_dir_path, f"cluster_{cluster_id}_summary.png")

        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved cluster summary: {output_path}")

   
    # ==============================================
    # ---    Morph generation implementations    ---
    # ==============================================
    # Mon 04 Aug 13:45:30 GMT by MAPA 
    # Selects representative image pairs whose nearest-neighbor distance is closest 
    # to the target distances obtained from the fitted probability distribution.
    @classmethod
    def select_pairs_from_distribution(cls, target_samples, neighbor_pairs, cluster_df, samples_per_percentile=2):
        selected_rows = []

        # Keep track of already selected neighbor pairs
        used_pairs = set()

        for _, target in target_samples.iterrows():
            # Extract params
            percentile = target["percentile"]
            target_distance = target["target_distance"]
            
            # Compute absolute distance error
            candidates = neighbor_pairs.copy()
            candidates["distance_error"] = np.abs( candidates["distance"] - target_distance)
            candidates = candidates.sort_values(by="distance_error")

            selected = 0

            # Select closest real pairs
            for _, pair in candidates.iterrows():
                idx1 = int(pair["idx1"])
                idx2 = int(pair["idx2"])

                # Avoid duplicates
                pair_key = tuple(sorted((idx1, idx2)))

                if pair_key in used_pairs:
                    continue

                used_pairs.add(pair_key)

                s1 = cluster_df.iloc[idx1]
                s2 = cluster_df.iloc[idx2]

                selected_rows.append({
                    "percentile": percentile,
                    "target_distance": target_distance,
                    "real_distance": pair["distance"],
                    "distance_error": pair["distance_error"],
                    "idx1": idx1,
                    "idx2": idx2,
                    "file_1": s1["file"],
                    "file_2": s2["file"],
                    "Age_1": s1["Age"],
                    "Age_2": s2["Age"],
                    "Race_1": s1["Dominant_Race"],
                    "Race_2": s2["Dominant_Race"],
                    "Gender_1": s1["Dominant_Gender"],
                    "Gender_2": s2["Dominant_Gender"],
                    "cluster_prob_1": s1["cluster_prob"],
                    "cluster_prob_2": s2["cluster_prob"]
                })

                selected += 1

                if selected >= samples_per_percentile:
                    break

        return pd.DataFrame(selected_rows)

    #Mon 04 Aug 13:45:30 GMT by MAPA 
    # Generates morph images for the representative pairs selected from the fitted distance distribution.
    @classmethod
    def generate_percentile_morphs(cls, experimental_pairs, cluster_id, output_dir_path, alpha=0.5):
        # Verify df content
        assert not experimental_pairs.empty, "There aren't any pair for morph generation."; 
        os.makedirs(output_dir_path, exist_ok=True)
        
        generated_samples = []

        for idx, row in experimental_pairs.iterrows():
            morph_filename = (
                f"cluster_{cluster_id}"
                f"_p{int(row['percentile']*100):02d}"
                f"_sample_{idx}.png"
            )

            morph_path = os.path.join(output_dir_path,morph_filename)

            params = Namespace(
                Sb1=row["file_1"],
                Sb2=row["file_2"],
                Morph=morph_path,
                Alpha=alpha
            )

            try:
                LIB_MorphGAN.MorphFace(params)
                generated_samples.append({"row": row, "morph_path": morph_path})
            except Exception as e:
                print( f"Error generating morph {idx}: {e}")

        print(f"Generated {len(generated_samples)} morphs.")

        return generated_samples

    #Wednesday 07 October 2026 12:04:40 GMT by MAPA
    @classmethod
    def gpuAcc_generate_percentile_morphs(cls, experimental_pairs, cluster_id, output_dir_path, alpha=0.5):
        # Verify df content
        assert not experimental_pairs.empty, "There aren't any pair for morph generation."; 

        os.makedirs(output_dir_path, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("There isn't any CUDA GPU available.")

        print(f" Starting morph Generation per Percentile for cluster {cluster_id}...")
        print(f" N_Worker_GPU: {num_gpus}")

        # Split dataframe pair in chunks for each available GPU
        chunks = np.array_split(experimental_pairs, num_gpus)

        # Multiprocessing setup
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Spawn method was already intialized

        for gpu_id in range(num_gpus):
            # Omit empty chunks
            if len(chunks[gpu_id]) == 0:
                continue

            p = mp.Process(
                target=cls._percentile_gpu_worker,
                args=(gpu_id, chunks[gpu_id], cluster_id, output_dir_path, alpha, return_dict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Gather each GPU outputs
        generated_samples = []
        for gpu_id in range(num_gpus):
            generated_samples.extend(return_dict.get(gpu_id, []))

        print(f"Generated {len(generated_samples)} morphs.")

        return generated_samples

    # Wed 11 June 2026 by MAPA
    @classmethod
    def process_cluster_generation(cls, cluster_pairs_df, cluster_id, output_dir_path="./results", alpha=0.5):
        # Verify df content
        assert not cluster_pairs_df.empty, "There aren't any pair for morph generation."; 
        os.makedirs(output_dir_path, exist_ok=True)

        print(f"\nGenerating cluster summary for cluster {cluster_id}")

        generated_samples = []

        # Generate all morphs for the current cluster
        for idx, row in cluster_pairs_df.iterrows():
            temp_morph_path = os.path.join(
                output_dir_path,
                f"cluster_{cluster_id}_temp_{idx}.png"
            )

            params = Namespace(Sb1=row["file_1"], Sb2=row["file_2"], Morph=temp_morph_path,Alpha=alpha)

            try:
                LIB_MorphGAN.MorphFace(params)
                generated_samples.append({
                    "row": row,
                    "morph_path": temp_morph_path
                })
            except Exception as e:
                print(f"Error generating morph for pair {idx}: {e}")

        # Validate generated samples
        if len(generated_samples) == 0:
            print(f"No morphs generated for cluster {cluster_id}")
            return

        print(f"Generated {len(generated_samples)} morphs.")

        print("Plotting Morph Generation...")
        # Plot samples generation per cluster
        cls.plot_cluster_generation(
            cluster_id,
            generated_samples,
            alpha,
            cluster_pairs_df,
            output_dir_path
        )

        # Remove temporary morph files
        for sample in generated_samples:
            try:
                if os.path.exists(sample["morph_path"]):
                    os.remove(sample["morph_path"])
            except Exception as e:
                print(f"Error removing temp file: {e}")

    #Wednesday 07 October 2026 12:04:40 GMT by MAPA
    @classmethod
    def gpuAcc_process_cluster_generation(cls, cluster_pairs_df, cluster_id, output_dir_path="./results", alpha=0.5):
        # Verify df content
        assert not cluster_pairs_df.empty, "There aren't any pair for morph generation."; 
        os.makedirs(output_dir_path, exist_ok=True)

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("There isn't any CUDA GPU available.")

        print(f" Starting morph Generation per Percentile for cluster {cluster_id}...")
        print(f" N_Worker_GPU: {num_gpus}")

        # Split dataframe pair in chunks for each available GPU
        chunks = np.array_split(cluster_pairs_df, num_gpus)

        # Multiprocessing setup
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Spawn method was already intialized

        for gpu_id in range(num_gpus):
            # Omit empty chunks
            if len(chunks[gpu_id]) == 0:
                continue

            p = mp.Process(
                target=cls._cluster_generation_gpu_worker,
                args=(gpu_id, chunks[gpu_id], cluster_id, output_dir_path, alpha, return_dict)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Gather each GPU outputs
        generated_samples = []
        for gpu_id in range(num_gpus):
            generated_samples.extend(return_dict.get(gpu_id, []))

        # Validate generated samples
        if len(generated_samples) == 0:
            print(f"No morphs generated for cluster {cluster_id}")
            return

        print(f"Generated {len(generated_samples)} morphs.")

        print("Plotting Morph Generation...")
        # Plot samples generation per cluster
        cls.plot_cluster_generation(
            cluster_id,
            generated_samples,
            alpha,
            cluster_pairs_df,
            output_dir_path
        )

        # Remove temporary morph files
        for sample in generated_samples:
            try:
                if os.path.exists(sample["morph_path"]):
                    os.remove(sample["morph_path"])
            except Exception as e:
                print(f"Error removing temp file: {e}")
        
        return generated_samples

    # Wed 07 Oct 2026 12:01:50 GMT by MAPA
    @classmethod
    def safe_parse_embedding(cls, val):
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val
    
    #Wed 11 June 14:08:13 GMT by MAPA      Last Mod: Wed 02 Sep 20:13:30 GMT by MAPA 
    @classmethod
    def ControlledMorphGeneration(
        cls, 
        output_plot_dir,
        clustered_mainfold_dataset = None,
        manifold_dataset_clustered_path = "../data/ManifoldAnalysis/manifold_dataset.csv",
        n_clusters = 15,
        mixed_clusters = True,
        mid_strategy="mean",
        cleaning_min_prob = 0.80,
        morph_gen_alpha = 0.5,
        gpuAcc = True
    ):
        print("\n" + "\033[0;34m" + "[Loading manifold clustered dataset...] " + str(datetime.datetime.now()) + "\033[0m")
        if clustered_mainfold_dataset is None:
            # Load clustered manifold dataset
            dataset = ClusterManager.load_manifold_dataset(manifold_dataset_clustered_path)
        else:
            dataset = clustered_mainfold_dataset

        print("\n" + "\033[0;34m" + "[Extracting top Clusters...] " + str(datetime.datetime.now()) + "\033[0m")
        # Get top populated HDBSCAN clusters
        raw_sample_clusters, cluster_sizes = ClusterManager.get_sample_clusters(dataset, n_clusters, mixed_clusters, mid_strategy)

        # Get mixed clusters
        sample_clusters = []
        for key, set in raw_sample_clusters.items():
            set = dict(set)
            for cluster_id, n in set.items():
                sample_clusters.append([cluster_id, n, key])

        print("Selected clusters:\n")
        for cluster in sample_clusters:
            print(cluster)

        # Controlled morph generation results directory
        controlled_morph_gen_results_dir_path = output_plot_dir + "/controlled_morph_gen_output"

        # Create the directory safely  
        os.makedirs(controlled_morph_gen_results_dir_path, exist_ok=True) 

        # Plot cluster_size distribution
        ClusterManager.plot_cluster_sizes_distribution(cluster_sizes, controlled_morph_gen_results_dir_path)

        cluster_idx = 0

        # Cluser quality metrics array
        all_cluster_quality = []

        total_samples = 0
        clean_samples = 0
        # Clean retention rate: full dataset
        for cluster_id, n in cluster_sizes.items():
            total_samples+=n
            cluster_df = ClusterManager.get_clean_cluster(dataset, cluster_id, cleaning_min_prob)
            clean_n = len(cluster_df)
            clean_samples+=clean_n

        print("\n- Full Cluster Space Cleaning metrics:")
        print("Total samples      : ", total_samples)
        print("Total Clean samples: ", clean_samples)
        print("Retention Rate     : ", clean_samples/total_samples*100)
        

        # Proccess n selected clusters
        print("\n" + "\033[0;34m" + f"[Starting Cluser Processing...] " + str(datetime.datetime.now()) + "\033[0m")
        for cluster_id, original_n, cluster_location in sample_clusters:
            # Create cluster's directory safely   
            cluster_controlled_morph_gen_results_dir_path = controlled_morph_gen_results_dir_path + f"/{cluster_location}_{cluster_idx}_cluster_{cluster_id}"
            os.makedirs(cluster_controlled_morph_gen_results_dir_path, exist_ok=True) 

            cluster_idx += 1

            print("\n" + "\033[0;34m" + f"[Processing cluster {cluster_id}] " + str(datetime.datetime.now()) + "\033[0m")

            # - Remove low-confidence samples and demographic inconsistencies
            print("\n" + "\033[0;34m" + f"[Cleaning Cluster {cluster_id}] " + str(datetime.datetime.now()) + "\033[0m")
            cluster_df = ClusterManager.get_clean_cluster(dataset, cluster_id, cleaning_min_prob)
            clean_n = len(cluster_df)
            print(
                f"Cluster {cluster_id}: "
                f"Original: {original_n}"
                f"Clean   : {clean_n}"
            )

            # - Compute cluster quality metrics
            print("\n" + "\033[0;34m" + f"[Computing cluster quality metrics...] " + str(datetime.datetime.now()) + "\033[0m")

            # Extract embeddings as an array [floats]
            embeddings = np.array(cluster_df["embedding"].apply(cls.safe_parse_embedding).tolist(), dtype=float)

            # - Compute innercluster pair to pair L1 histogram using knn (param: 2 neighs)
            neighbor_pairs = ClusterManager.analyze_neighbor_pairs(cluster_df)

            # - Fit some distribution types (χ², Gamma, Weibull and Lognormal) and compare
            # Extract distances values for distribution analysis
            distances = neighbor_pairs["distance"].values

            # ============================================================
            # - CLUSTER QUALITY METRICS

            # Compute cluster's retention rate (after cleaning samples)
            clean_retention_rate = clean_n/original_n
            
            # T-STUDENT
            #t_student = ClusterManager.compute_t_student_metrics(distances,confidence=0.95)

            # COMPRESSION
            compression = ClusterManager.compute_compression_metrics(embeddings)

            # DENSITY
            density = ClusterManager.compute_density_metrics(neighbor_pairs)

            # DISTRIBUTIONAL QUALITY
            # Define candidate distributions to fit
            candidate_distributions = {
                "Chi-Square": stats.chi2,
                "Gamma": stats.gamma,
                "Weibull": stats.weibull_min,
                "Lognormal": stats.lognorm
            }

            # Fit data to candidate distributions and compute KS (How well does the selected distribution represent the observed data?)
            # and AIC (How favorable is the selected model regarding the candidates, considering fit and complexity?) 
            # as Distribution quality metrics
            fitted_results = ClusterManager.fit_candidate_distributions(distances, candidate_distributions)

            # Sort values by its ks in order to know the best fitted distribution model
            fitted_results = fitted_results.sort_values(by=["ks_statistic", "AIC"], ascending=[True, True])

            # Select the best model
            print("\n" + "\033[0;34m" + f"[Selecting the best distribution model...] " + str(datetime.datetime.now()) + "\033[0m")
            best_distribution = fitted_results.iloc[0]
            print("Best fitted distributed: ", best_distribution["name"])
            print(best_distribution)

            distribution_quality = {
                "best_distribution": best_distribution["name"],
                "ks_statistic": best_distribution["ks_statistic"],
                "ks_p_value": best_distribution["p_value"],
                "aic": best_distribution["AIC"]
            }

            # BOOTSTRAP STABILITY (Do the obtained model and parameters hold up when we perturb or resample the data?)
            bootstrap_results = ClusterManager.bootstrap_distribution_stability(
                distances=distances,
                candidate_distributions=candidate_distributions,
                n_iterations=100,
                confidence=0.90,
                random_state=42
            )
            bootstrap_stability = ClusterManager.summarize_bootstrap_stability(bootstrap_results)

            # Cluster quality metrics dictionary
            cluster_quality = {
                "cluster_id": cluster_id,
                "n_samples": len(cluster_df),

                # Cleaning Retention rate
                "original_n": original_n,
                "clean_n": clean_n,
                "clean_retention_rate" : clean_retention_rate,

                # Compression
                "compression_mean_radius": compression["centroid_radius_mean"],
                "compression_std_radius": compression["centroid_radius_std"],
                "compression_median_radius": compression["centroid_radius_median"],
                "compression_q90_radius": compression["centroid_radius_q90"],

                # Density
                "density_mean_knn_distance": density["mean_knn_distance"],
                "density_median_knn_distance": density["median_knn_distance"],
                "density_knn": density["knn_density"],

                # Distribution
                "fitted_distribution": distribution_quality["best_distribution"],
                "ks": distribution_quality["ks_statistic"],
                "aic": distribution_quality["aic"],

                # Bootstrap
                "bootstrap_model_stability": bootstrap_stability["model_stability"]
            }

            # Append Cluster's ClusterManager to buffer
            all_cluster_quality.append(cluster_quality)

            print("\n" + "\033[0;34m" + f"[Cluster quality metrics Completed] " + str(datetime.datetime.now()) + "\033[0m")

            # - END - CLUSTER QUALITY METRICS
            # ============================================================

            # Save results
            fitted_results.to_csv( cluster_controlled_morph_gen_results_dir_path + f"/cluster_{cluster_id}_distribution_fit.csv",index=False)
            
            # - Determine trust region
            print("\n" + "\033[0;34m" + f"[Computing trust region...] " + str(datetime.datetime.now()) + "\033[0m")
            trust_region = ClusterManager.compute_trust_region(best_distribution, confidence = 0.90)
            print(
                f"Trust Region ({trust_region['confidence']*100:.1f}%): "
                f"[{trust_region['lower']:.4f}, {trust_region['upper']:.4f}]"
            )

            # - Save histogram with its distribution fit and the resultant trust region
            print("\n" + "\033[0;34m" + f"[Plotting resultant trust region...] " + str(datetime.datetime.now()) + "\033[0m")
            ClusterManager.plot_fitted_distribution(
                values=distances,
                best_distribution=best_distribution,
                trust_region=trust_region,
                filepath=cluster_controlled_morph_gen_results_dir_path + f"/cluster_{cluster_id}_distribution_fit_{best_distribution['name']}.png",
                title=f"Cluster {cluster_id}: Neighbor Distances Distribution Fit ({best_distribution['name']})"
            )

            # - For a fitted distribution F(x), select representative points from the CDF
            print("\n" + "\033[0;34m" + f"[Selecting representative points from the CDF...] " + str(datetime.datetime.now()) + "\033[0m")
            target_samples = ClusterManager.sample_distribution_percentiles(best_distribution)

            # identify the actual pairs with distances closest to these points
            print("\n" + "\033[0;34m" + f"[Selecting pairs...] " + str(datetime.datetime.now()) + "\033[0m")
            experimental_pairs = cls.select_pairs_from_distribution(
                target_samples,
                neighbor_pairs,
                cluster_df,
                samples_per_percentile=2
            )

            experimental_pairs.to_csv(
                cluster_controlled_morph_gen_results_dir_path +
                f"/cluster_{cluster_id}_experimental_pairs.csv",
                index=False
            )

            # generate two morphs for each level and plot gloabl cluster's summary
            print("\n" + "\033[0;34m" + f"[Generating morphs...] " + str(datetime.datetime.now()) + "\033[0m")
            generation_method = cls.gpuAcc_generate_percentile_morphs if gpuAcc else cls.generate_percentile_morphs
            generated_samples = generation_method(
                experimental_pairs,
                cluster_id,
                cluster_controlled_morph_gen_results_dir_path + f"/morphs",
                alpha=morph_gen_alpha
            )
            ClusterManager.create_percentile_summary(
                generated_samples=generated_samples,
                cluster_id=cluster_id,
                values=distances,
                best_distribution=best_distribution,
                trust_region=trust_region,
                output_dir_path=cluster_controlled_morph_gen_results_dir_path
            )

            # - Generate a global trust region based on each cluster’s trust parameters
            
        # Save all clusters quality metrics
        quality_df = pd.DataFrame(all_cluster_quality)
        quality_df.to_csv(
            controlled_morph_gen_results_dir_path + "/cluster_quality_summary.csv",
            index=False
        )

        print("\n" + "\033[0;34m" + f"[Controlled Morph Generation was Successfully completed] " + str(datetime.datetime.now()) + "\033[0m")
