python Main.py VideoMorph
python Main.py DelaunayImageMorph
python Main.py DelaunayImageDirMorph 
python Main.py GANImageDirMorph 

# Exec deepFace analyzer for image directory
python Main.py Demographics4Folder --SPath ../data/FFHQ_Real --JSON ../data/ffhq_real_demographics_meta_data.json 

# Exec transformation of a deepFace generated json into structured csv
python Main.py DeepFaceJSON2CSV --jsonPath ../data/ffhq_real_demographics_meta_data.json --csvPath ../data/ffhq_real_demographics_meta_data_structured.csv --sourceDataPath ../data/FFHQ_Real
