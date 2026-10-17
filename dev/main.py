# main.py

import argparse, datetime, os, site, torch
import tensorflow as tf
from pathlib import Path
from src.face_morphing.config import Config
from src.face_morphing.pipeline.orchestrator import run_pipeline

"""
inside /dev: 
exec "pip install -e src/face_morphing" for face_morphing import as local module
exec "pip install -e src/stylegan2_ada" for stylegan2_ada import as local module
exec "pip install -e src/deepfaceMaster" for deepfaceMaster import as local module

"""

# Configure TF and gpu's variables
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Link nvidia libraries 
nvidia_paths = [
    os.path.join(sp, "nvidia", d, "lib")
    for sp in site.getsitepackages()
    if os.path.exists(os.path.join(sp, "nvidia"))
    for d in os.listdir(os.path.join(sp, "nvidia"))
    if os.path.isdir(os.path.join(sp, "nvidia", d, "lib"))
]
if nvidia_paths:
    os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

def configure_hardware_and_diagnose():
    print("=" * 65)
    print(" Adaptative GPU Hardware Review")
    print("=" * 65)

    # 1. Inspeccionar GPUs con PyTorch
    torch_available = torch.cuda.is_available()
    torch_gpus = torch.cuda.device_count() if torch_available else 0
    print(f" PyTorch: {torch_gpus} GPU(s) available")

    has_pascal_gpu = False
    for i in range(torch_gpus):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        print(f"   - GPU {i}: {name} (Compute Capability {cap[0]}.{cap[1]})")
        # Arquitectura Pascal corresponde a Compute Capability 6.x (GTX 1080 Ti, 1070, etc.)
        if cap[0] == 6:
            has_pascal_gpu = True

    # TF setup automation
    if has_pascal_gpu:
        print("\n [INFO] Pascal GPU detected (GTX 1080 Ti / Compute Capability 6.x).")
        print(" [INFO] TensorFlow redirected to CPU for cuDNN 5003 prevention.")
        print(" [INFO] Pytorch will continue using GPUs.")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception as e:
            print(f" [WARN] Couldn't hide gpu's from tf: {e}")
    else:
        tf_gpus = tf.config.list_physical_devices("GPU")
        if tf_gpus:
            print(f"\n TensorFlow: {len(tf_gpus)} GPU(s) enabled (Verified Compatibility)")
            for gpu in tf_gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass
        else:
            print("\n TensorFlow: Executing in CPU")

    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Ruta relativa al archivo de configuración YAML")
    args = parser.parse_args()

    # Diagnose hardware and setup tf
    configure_hardware_and_diagnose()

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