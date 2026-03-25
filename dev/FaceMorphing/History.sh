python Main.py VideoMorph
python Main.py DelaunayImageMorph
python Main.py DelaunayImageDirMorph 
python Main.py GANImageDirMorph 

# Exec deepFace analyzer for single image
python Main.py Demographics4SingleFile --os_png_tool cv2
python Main.py Demographics4SingleFile --os_png_tool magick

# Exec deepFace analyzer for image directory
python Main.py Demographics4Folder --SPath ../data/FFHQ_Real --JSON ../data/ffhq_real_demographics_meta_data.json 
python Main.py Demographics4Folder --SPath ./DATA --JSON ./testData_metadata.json --os_png_tool cv2

# Exec transformation of a deepFace generated json into structured csv
python Main.py DeepFaceJSON2CSV --jsonPath ../data/ffhq_real_demographics_meta_data.json --csvPath ../data/ffhq_real_demographics_meta_data_structured.csv --sourceDataPath ../data/FFHQ_Real

# Exec create image embedding with deepface
python Main.py SingleImageEmbeddingGeneration --input_path ./DATA/Sb1.png --model Facenet512