# main.py

import argparse, datetime
from pathlib import Path
from src.face_morphing.config import Config
from src.face_morphing.pipeline.orchestrator import run_pipeline

"""
inside /dev: 
exec "pip install -e src/face_morphing" for face_morphing import as local module
exec "pip install -e src/stylegan2_ada" for stylegan2_ada import as local module
exec "pip install -e src/deepfaceMaster" for deepfaceMaster import as local module

"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta relativa al archivo de configuración YAML")
    args = parser.parse_args()

    # Load yaml config file 
    config_obj = Config.load(args.config)

    # Inject config file for face_morphing pipeline execution 
    run_pipeline(config=config_obj)


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;31m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;31m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )