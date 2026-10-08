# Libraries
import datetime, os, torch
from types import SimpleNamespace
from face_morphing.config import Config
from face_morphing.pipeline import DatasetLector

# Modules
from face_morphing.libs import LIB_DeepFace
from face_morphing.pipeline import Embeddings_Dim_Reduction
from face_morphing.pipeline.ClusterManager import ClusterManager
from face_morphing.pipeline.MorphGenerator import MorphGenerator



# Wed 07 Oct 2026 12:01:50 GMT by MAPA
def makedir(config: Config, dir_name):
    dir_path = config.outputs.experiment_dir + dir_name
    os.makedirs(dir_path, exist_ok=True)

    return dir_path

#Sun 05 Oct 2026 17:43:28 GMT by MAPA
def init_process(config: Config, title: str):
    # Clean cuda cache
    torch.cuda.empty_cache()

    # Input Dataset directory verification 
    assert os.path.exists(config.dataset.raw_dir), F"Dataset Image Directory '{config.dataset.raw_dir}' doesn't exists."

    # Output directory verification
    config.outputs.experiment_dir = config.outputs.experiment_dir + f"/{title}"
    if not config.outputs.allow_dir_overwrite:
        assert not os.path.exists(config.outputs.experiment_dir), f"Output Path '{config.outputs.experiment_dir}' already exists. Verify or allow directory overwrite."
    os.makedirs(config.outputs.experiment_dir, exist_ok=True)
    

#Sun 04 Oct 2026 13:49:45 GMT by MAPA
def run_pipeline(config: Config):
    print("\n" + "\033[0;35m" + f"[Starting Face Morphing Pipeline Execution for {config.dataset.raw_dir}...] " + str(datetime.datetime.now()) + "\033[0m")
    title = config.experiment_data.name+"_"+config.dataset.name if len(config.experiment_data.name)>0 else config.dataset.name

    # Pipeline execution init
    init_process(config, title)

    # ====================================
    # == Pipeline secuential execution  == #Sun 05 Oct 2026 17:43:28 GMT by MAPA
    # ====================================
    n = 1000

    # ------ Demographic info extraction ------ 
    # Demographic Extraction
    print("\n" + "\033[0;35m" + f"[Running Demographic Extraction...] " + str(datetime.datetime.now()) + "\033[0m")
    json_metadata_path = config.outputs.experiment_dir + f"/{title}_demographics_meta_data.json"
    LIB_DeepFace.Demographics4Folder(
        SimpleNamespace(
            SPath=config.dataset.raw_dir,
            JSON=json_metadata_path,
            N=n,
            os_png_tool=config.demographics.os_png_tool
        )
    )
    # Clean cuda cache
    torch.cuda.empty_cache()

    # Deepface data transformation
    print("\n" + "\033[0;35m" + f"[Running Structured Demographic Data Transformation...] " + str(datetime.datetime.now()) + "\033[0m")
    csv_metadata_path = config.outputs.experiment_dir + f"/{title}_demographics_meta_data.csv"
    LIB_DeepFace.transform_deepFacejson2csv(
        SimpleNamespace(
            jsonPath=json_metadata_path,
            csvPath=csv_metadata_path,
            sourceDataPath=config.dataset.raw_dir
        )
    )

    # ------ Data encoding (embedding generation) ------ 
    print("\n" + "\033[0;35m" + f"[Running Data Encoding (embedding generation)...] " + str(datetime.datetime.now()) + "\033[0m")
    title+= f"_{config.data_encoding.model}"
    embeddings_output_dir = makedir(config, "/Embeddings")
    json_embedding_data_path = embeddings_output_dir + f"/{title}_embedding_meta_data.json"
    csv_embedding_data_statuslog_path = embeddings_output_dir + f"/{title}_embedding_meta_data_statuslog.csv"
    LIB_DeepFace.GenerateJSONEmbeddings(
        SimpleNamespace(
            SPath=json_metadata_path, 
            model=config.data_encoding.model, 
            JSON=json_embedding_data_path, 
            csv_status_file=csv_embedding_data_statuslog_path, 
            detector_backend=config.data_encoding.detector_backend, 
            gpuAcc=config.data_encoding.gpuAcc,
            n_processes=config.data_encoding.n_processes
        )
    )
    # Clean cuda cache
    torch.cuda.empty_cache()

    # ------ Embedding space dim reduction ------ 
    print("\n" + "\033[0;35m" + f"[Embedding Space Dim Reduction...] " + str(datetime.datetime.now()) + "\033[0m")
    # Create Dataset by joining embedding's json with demographic structured file
    # cols (file Age embedding Dominant_Race Dominant_Gender)
    embeddings_and_demogrpahics_output_dir = makedir(config, "/Embeddings_and_Demographics")
    dataset_csv_path = embeddings_and_demogrpahics_output_dir + f"/{title}_embeddings_and_demographics_meta_data.csv"
    embeddings_and_demographics_dataset = DatasetLector.createDataset(
        demographic_csv_path=csv_metadata_path, 
        embeddings_json_path=json_embedding_data_path, 
        create_csv=True, 
        dataset_csv_path=dataset_csv_path
    )

    # Execute Dim Reduction Methods
    dim_reduction_output_dir = makedir(config, "/dim_reduction_figs")
    Embeddings_Dim_Reduction.Dataset_Dim_Reduction(
        dataset=embeddings_and_demographics_dataset,
        output_plot_dir= dim_reduction_output_dir,
        n_components=config.dim_reduction.n_components,
        n_neighbors=config.dim_reduction.n_neighbors,
        tSNE_perplexity=config.dim_reduction.tSNE_perplexity,
        exec_heavy_algorithms=config.dim_reduction.exec_heavy_algorithms,
        plot=config.dim_reduction.plot_results,
        show_plots=config.dim_reduction.show_plots_runtime,
        random_state=config.dim_reduction.random_state
    )

    # ------ Clustering and Manifold Plot ------
    print("\n" + "\033[0;35m" + f"[Clustering and Manifold Plot...] " + str(datetime.datetime.now()) + "\033[0m")
    # Build clustered manifold
    clustered_dataset_output_csv_path = config.outputs.experiment_dir + f"/{title}_clustered_dataset.csv"
    reduced_X, cluster_labels, embeddings_and_demographics_dataset = ClusterManager.build_manifold(
        dataset_path= dataset_csv_path,
        dim_red_algorithm = config.cluster_generation.selected_dim_reduction_algorithm,
        dim_red_algorithm_params = config.cluster_generation.dim_red_algorithm_params,
        hdbscan_min_cluster_size=config.cluster_generation.hdbscan_min_cluster_size, 
        hdbscan_min_samples=config.cluster_generation.hdbscan_min_samples,
        clustered_dataset_output_path=clustered_dataset_output_csv_path
    )

    # Plot resultant clustered manifold
    clustered_manifold_output_dir = makedir(config, "/clustered_manifold_figs")
    ClusterManager.plot_manifold(
        reduced_X,
        cluster_labels,
        embeddings_and_demographics_dataset,
        config.cluster_generation.selected_dim_reduction_algorithm,
        output_plot_dir = clustered_manifold_output_dir,
        show_plot=config.cluster_generation.show_plots_runtime
    )

    # Run resultant clustered manifold analaysis
    clustered_manifold_analysis_output_dir = makedir(config, "/clustered_manifold_analysis")
    ClusterManager.analyze_cluster(
        embeddings_and_demographics_dataset=embeddings_and_demographics_dataset,
        dim_red_algorithm=config.cluster_generation.selected_dim_reduction_algorithm,
        output_analysis_csv_path=clustered_manifold_analysis_output_dir + f"/{title}_cluster_analysis.csv",
        output_plot_dir=clustered_manifold_analysis_output_dir,
        show_plot=config.cluster_generation.show_plots_runtime
    )

    # ------ Controlled Morph Process ------ 
    print("\n" + "\033[0;35m" + f"[Controlled Morphing...] " + str(datetime.datetime.now()) + "\033[0m")
    # Clean cuda cache
    torch.cuda.empty_cache()
    MorphGenerator.ControlledMorphGeneration(
        output_plot_dir=config.outputs.experiment_dir,
        clustered_mainfold_dataset=embeddings_and_demographics_dataset,
        manifold_dataset_clustered_path=clustered_dataset_output_csv_path,
        n_clusters = config.controlled_morph_generation.n_clusters,
        mixed_clusters = config.controlled_morph_generation.mixed_clusters,
        mid_strategy = config.controlled_morph_generation.mid_strategy,
        cleaning_min_prob = config.controlled_morph_generation.cleaning_min_prob,
        morph_gen_alpha = config.controlled_morph_generation.morph_gen_alpha,
        gpuAcc=config.controlled_morph_generation.gpuAcc
    )

    # Clean cuda cache
    torch.cuda.empty_cache()

    print("\n" + "\033[0;35m" + f"[Face Morphing Pipeline was Successfully Completed. Results saved in {config.outputs.experiment_dir}] " + str(datetime.datetime.now()) + "\033[0m")
    
    #return final_results
