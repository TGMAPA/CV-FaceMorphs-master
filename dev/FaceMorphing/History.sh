python Main.py VideoMorph
python Main.py DelaunayImageMorph
python Main.py DelaunayImageDirMorph 
python Main.py GANImageDirMorph 

# Exec deepFace analyzer for single image
python Main.py Demographics4SingleFile --os_png_tool cv2
python Main.py Demographics4SingleFile --os_png_tool magick

# Exec deepFace analyzer for image directory
python Main.py Demographics4Folder --SPath ../data/FFHQ_Real --JSON ./ffhq_real_demographics_meta_data.json --os_png_tool cv2 > ./ffhq_deepface_demographics_log.out
python Main.py Demographics4Folder --SPath ./DATA --JSON ./testData_metadata.json --os_png_tool cv2 > ./testData_deepface_demographics_log.out

# Exec transformation of a deepFace generated json into structured csv
python Main.py DeepFaceJSON2CSV --jsonPath ../data/Demographics/ffhq_real_demographics_meta_data.json --csvPath ../data/Demographics/ffhq_real_demographics_meta_data_structured.csv --sourceDataPath ../data/FFHQ_Real

# Exec create image embedding with deepface
python Main.py SingleImageEmbeddingGeneration --input_path ./DATA/Sb1.png --model Facenet512

# Exec create image directory embeddings with deepface
python Main.py ImageDirectoryEmbeddingGeneration --SPath ../data/FFHQ_Real --model Facenet512 --JSON ./ffhq_real_deepface_embeddings_metadata.json --csv_status_file ./deepface_embeddings_metadata_status.csv --detector_backend skip --gpuAcc True --n_processes 4 > ./deepface_embeddings_generation.out
python Main.py ImageDirectoryEmbeddingGeneration --model Facenet512 --JSON ../data/Test_deepface_embeddings_metadata.json --csv_status_file ../data/Test_deepface_embeddings_metadata_status.csv --detector_backend skip 