from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml

# Obtener la ruta raíz del entorno de desarrollo (dev/) basada en la ubicación de este archivo
# config.py -> face_morphing/ -> src/ -> dev/
CONFIG_FILE_DIR = Path(__file__).resolve().parent
DEV_DIR = CONFIG_FILE_DIR.parent.parent 
DEFAULT_CONFIG_PATH = DEV_DIR / "configs" / "default.yaml"


@dataclass
class DatasetConfig:
    raw_dir: str = "data/raw"


@dataclass
class DemographicsConfig:
    generate_eda: bool = True
    deepface_actions: list[str] = field(default_factory=lambda: ["age", "gender", "race"])
    detector_backend: str = "opencv"


@dataclass
class OutputsConfig:
    experiment_dir: str = "outputs/exp_001"


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    demographics: DemographicsConfig = field(default_factory=DemographicsConfig)
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
            dataset=DatasetConfig(**data.get("dataset", {})),
            demographics=DemographicsConfig(**data.get("demographics", {})),
            outputs=OutputsConfig(**data.get("outputs", {})),
        )


# Cargar la configuración por defecto de forma segura usando la ruta anclada
config = Config.load(DEFAULT_CONFIG_PATH)