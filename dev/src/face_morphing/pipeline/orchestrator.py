# src/face_morphing/pipeline/orchestrator.py

from face_morphing.config import Config
from face_morphing.pipeline.dataset_lecture import test

#Sun 04 Oct 2026 13:49:45 GMT by MAPA
def run_pipeline( config: Config,):
    # print(f"--- Iniciando Workflow para: {dataset_dir} ---")
    
    print(f"--- Ejecutando pipeline en: {config.dataset.raw_dir} ---")
        
    # Pasas el objeto config a las etapas
    df_demographics = test(config)
    
    print(f"--- Guardando en: {config.outputs.experiment_dir} ---")
    
    # print(f"--- Workflow Finalizado. Resultados guardados en {output_dir} ---")
    #return final_results
