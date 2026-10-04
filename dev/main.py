# main.py

import argparse
from pathlib import Path
import argparse
from face_morphing.config import Config
from face_morphing.pipeline.orchestrator import run_pipeline

"""
inside /dev: 
       exec pip install -e . for face_morphing import as local module
       
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta relativa al archivo de configuración YAML")
    args = parser.parse_args()

    # Load yaml config file 
    config_obj = Config.load(args.config)
    
    # Inject config file for face_morphing pipeline execution 
    run_pipeline(config=config_obj)