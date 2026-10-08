# Libraries
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

# Obtain dev root path based on this file's path 
# config.py -> face_morphing/ -> src/ -> dev/
CONFIG_FILE_DIR = Path(__file__).resolve().parent
DEV_DIR = CONFIG_FILE_DIR.parent.parent 
DEFAULT_CONFIG_PATH = DEV_DIR / "configs" / "default.yaml"


@dataclass
class ExperimentData:
    name: str = "Exp_01"
    n: int = -1

@dataclass
class DatasetConfig:
    raw_dir: str = "data/raw"
    name: str = "Dataset"

@dataclass
class DemographicsConfig:
    generate_eda: bool = True
    os_png_tool: str = "cv2" 
    run_plots: bool = True

@dataclass
class DataEncoding:
    model: str = "Facenet512"
    detector_backend: str = "skip"
    gpuAcc: bool = True
    n_processes: int = 4

@dataclass
class DimReduction:
    n_components: int = 2
    n_neighbors: int = 10
    tSNE_perplexity: int = 30
    exec_heavy_algorithms: bool = False
    plot_results: bool = False
    show_plots_runtime: bool = False
    random_state: int = 42

@dataclass
class ClusterGeneration:
    selected_dim_reduction_algorithm: str = "tsne"
    dim_red_algorithm_params: dict = field(
        default_factory=lambda: {
            "tSNE_perplexity": 30,
            "init": "pca",
            "learning_rate": "auto",
            "random_state": 42
        }
    )
    hdbscan_min_cluster_size: int = 20
    hdbscan_min_samples: int = 10
    show_plots_runtime: bool = False

@dataclass
class ControlledMorphGeneration:
    n_clusters: int = 15
    mixed_clusters: bool = True
    mid_strategy: str = "mean"
    cleaning_min_prob: float = 0.80
    morph_gen_alpha: float = 0.5
    gpuAcc: bool = True

@dataclass
class OutputsConfig:
    experiment_dir: str = "outputs/exp_001"
    allow_dir_overwrite: bool = False

@dataclass
class Config:
    experiment_data: ExperimentData = field(default_factory=ExperimentData)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
    data_encoding: DataEncoding = field(default_factory=DataEncoding)
    dim_reduction: DimReduction = field(default_factory=DimReduction)
    cluster_generation: ClusterGeneration = field(default_factory=ClusterGeneration)
    controlled_morph_generation: ControlledMorphGeneration = field(default_factory=ControlledMorphGeneration)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)

    def save(self, output_path: str | Path):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

        print(f"[INFO] Config saved at: {path}")

    @classmethod
    def load(cls, config_path: str | Path = DEFAULT_CONFIG_PATH) -> "Config":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file {path} doesn't exist.")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            experiment_data=ExperimentData(**data.get("experiment_data", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            demographics=DemographicsConfig(**data.get("demographics", {})),
            data_encoding=DataEncoding(**data.get("data_encoding", {})),
            dim_reduction=DimReduction(**data.get("dim_reduction", {})),
            cluster_generation=ClusterGeneration(**data.get("cluster_generation", {})),
            controlled_morph_generation=ControlledMorphGeneration(**data.get("controlled_morph_generation", {})),
            outputs=OutputsConfig(**data.get("outputs", {})),
        )


# Load config file using attached path
config = Config.load(DEFAULT_CONFIG_PATH)