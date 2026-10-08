python Main.py MetaDemographicsVGG2 --SPATH /mnt/TB2/VGG-Face2/data/ --CSV_meta ./CSV/VGGF2_identity_meta.csv

python Main.py MetaDemographicsCELEBA --SPATH /mnt/TB2/DATASETS/CELEBA_FACES/Original/ --CSV_meta ./CSV/identity_CelebA.csv

python Main.py Demographics4Folder --SPath /mnt/TB2/FaceRecognition/BUPT/BUPT-Balancedface/images/race_per_7000/ --CSV BUPT_Balanced_demographics_N10.csv --N 10

python Main.py Demographics4Folder --SPath /home/nmp56/Documents/BUPT/race_per_7000 --JSON /home/nmp56/Documents/BUPT/BUPT_Equalized_Demographic.json --N 10 --os_png_tool convert
