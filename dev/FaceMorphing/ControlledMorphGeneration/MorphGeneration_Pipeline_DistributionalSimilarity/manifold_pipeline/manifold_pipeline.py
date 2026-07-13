# Import modules
from dev.FaceMorphing.ControlledMorphGeneration.MorphGeneration_Pipeline_DistributionalSimilarity.manifold_pipeline.build_manifold import build_manifold
from dev.FaceMorphing.ControlledMorphGeneration.MorphGeneration_Pipeline_DistributionalSimilarity.manifold_pipeline.plot_manifold import plot_manifold
from dev.FaceMorphing.ControlledMorphGeneration.MorphGeneration_Pipeline_DistributionalSimilarity.manifold_pipeline.analyze_cluster import analyze_cluster

# Import libraries
import datetime


# Execute manifold pipeline to prepare manifold clustered dataset and start morph generation
def main():

    print("\n" + "\033[0;34m" + "[Building manifold...] " + str(start) + "\033[0m");
    # #Wed 13 May 22:26:55 GMT by MAPA
    # Build manifold or load if already exists
    X_tsne, cluster_labels, embeddings_and_demographics_dataset = build_manifold("../data/ManifoldAnalysis/manifold_dataset.csv")

    print("\n" + "\033[0;34m" + "[Ploting Manifold...] " + str(start) + "\033[0m");
    # #Sun 24 May 20:47:30 GMT by MAPA
    # Plot manifold
    plot_manifold(X_tsne, cluster_labels, embeddings_and_demographics_dataset)

    print("\n" + "\033[0;34m" + "[Executing Cluster Analysis...] " + str(start) + "\033[0m");
    # Cluster Anaylisis
    df_cluster_wasserstein = analyze_cluster(embeddings_and_demographics_dataset)
    


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )
