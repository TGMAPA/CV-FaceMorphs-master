'''
████████ ██████  ██      █████  ██       █████  ███    ██     ████████ ██    ██ ██████  ██ ███    ██  ██████  
   ██    ██   ██ ██     ██   ██ ██      ██   ██ ████   ██        ██    ██    ██ ██   ██ ██ ████   ██ ██       
   ██    ██   ██ ██     ███████ ██      ███████ ██ ██  ██        ██    ██    ██ ██████  ██ ██ ██  ██ ██   ███ 
   ██    ██   ██ ██     ██   ██ ██      ██   ██ ██  ██ ██        ██    ██    ██ ██   ██ ██ ██  ██ ██ ██    ██ 
   ██    ██████  ██     ██   ██ ███████ ██   ██ ██   ████        ██     ██████  ██   ██ ██ ██   ████  ██████  

https://www.turing.ac.uk/people/researchers/roberto-leyva-fernandez                                                                                                              

'''


# conda activate genmorph

import argparse, datetime
from libs import LIB_FaceMorph, LIB_MorphGAN, LIB_DeepFace


def main():
	parser = argparse.ArgumentParser("Face Morphs");
	subparsers = parser.add_subparsers();

	subparser = subparsers.add_parser("VideoMorph", description = "The original code generating a video of morphs");
	subparser.add_argument("--Sb1", required = False, type = str, default = "./DATA/Sb1.png", help = "Subject one face"),
	subparser.add_argument("--Sb2", required = False, type = str, default = "./DATA/Sb2.png", help = "Subject two face"),
	subparser.add_argument("--Morph", required = False, type = str, default = "MorphedFace.mp4", help = "Morph output <MorphedFace.mp4>"),	
	subparser.add_argument("--Length", required = False, type = int, default = 5, help = "Video length in seconds")
	subparser.add_argument("--FPS", required = False, type = int, default = 20, help = "Frames Per Second")
	subparser.set_defaults(func = LIB_FaceMorph.VideoMorph);



	# -- Delaunay MorphFace   #Sun 26 Feb 2026 11:21:45 GMT by MAPA
	# Execute Delaunay Image morphing for 2 images
	subparser = subparsers.add_parser("DelaunayImageMorph", description = "Image 2 Image morphs");
	subparser.add_argument("--Sb1", required = False, type = str, default = "./DATA/Sb1.png", help = "Subject one's face"),
	subparser.add_argument("--Sb2", required = False, type = str, default = "./DATA/Sb2.png", help = "Subject two's face"),
	subparser.add_argument("--Morph", required = False, type = str, default = "MorphedFace.png", help = "Morph output <MorphedFace.png>"),	
	subparser.add_argument("--Alpha", type = float, default = 0.5, help = "Blending factor");
	subparser.set_defaults(func = LIB_FaceMorph.MorphFace);

	# Execute Directory Delaunay Image morphing 	
	subparser = subparsers.add_parser("DelaunayImageDirMorph", description = "Image Directory all-vs-all Delaunay morphs");
	subparser.add_argument("--ImageDir", required = False, type = str, default = "./DATA", help = "Image Directory with Subject's faces"),
	subparser.add_argument("--MorphDir", required = False, type = str, default = "./Morph_Results", help = "Morph output Directory"),	
	subparser.add_argument("--Alpha", type = float, default = 0.5, help = "Blending factor");
	subparser.add_argument("--DirProportion", type = float, default = 1.0, help = "Directory percentage to process");
	subparser.set_defaults(func = LIB_FaceMorph.Dir_Automation_MorphFace);



	# -- GAN MorphFace   #Thursday 05 March 2026 11:30:50 GMT by MAPA
	# Execute GAN Image morphing for 2 images
	subparser = subparsers.add_parser("GANImageMorph", description = "Image 2 Image Delaunay morphs");
	subparser.add_argument("--Sb1", required = False, type = str, default = "./DATA/Sb1.png", help = "Subject one's face"),
	subparser.add_argument("--Sb2", required = False, type = str, default = "./DATA/Sb2.png", help = "Subject two's face"),
	subparser.add_argument("--Morph", required = False, type = str, default = "MorphedFace.png", help = "Morph output <MorphedFace.png>"),	
	subparser.add_argument("--Alpha", type = float, default = 0.5, help = "Blending factor");
	subparser.set_defaults(func = LIB_MorphGAN.MorphFace);

	# Execute Directory GAN Image morphing	 
	subparser = subparsers.add_parser("GANImageDirMorph", description = "Image Directory all-vs-all GAN morphs");
	subparser.add_argument("--ImageDir", required = False, type = str, default = "./DATA", help = "Image Directory with Subject's faces"),
	subparser.add_argument("--MorphDir", required = False, type = str, default = "./Morph_Results", help = "Morph output Directory"),	
	subparser.add_argument("--Alpha", type = float, default = 0.5, help = "Blending factor");
	subparser.add_argument("--DirProportion", type = float, default = 1.0, help = "Directory percentage to process");
	subparser.set_defaults(func = LIB_MorphGAN.Dir_Automation_MorphFace);



	# -- DeepFace 	#Sun 15 March 13:25:35 GMT by MAPA
	# 
	subparser = subparsers.add_parser("MetaDemographicsVGG2", description = "Obtain demographic metrics from metrics");
	subparser.add_argument("--SPATH", required = True, type = str, help = "Source path of face images train/test");
	subparser.add_argument("--CSV_meta", required = True, type = str, help = "CSV meta data");
	subparser.set_defaults(func = LIB_DeepFace.MetaDemographicsVGG2);

	subparser = subparsers.add_parser("MetaDemographicsCELEBA", description = "Obtain demographic metrics from metrics");
	subparser.add_argument("--SPATH", required = True, type = str, help = "Source path of face images train/test");
	subparser.add_argument("--CSV_meta", required = True, type = str, help = "CSV meta data");
	subparser.add_argument("--Subsample", required = False, default = 10, type = int, help = "Number of subsamples");
	subparser.set_defaults(func = LIB_DeepFace.MetaDemographicsCELEBA);

	subparser = subparsers.add_parser("InsertDemographics", description = "Insert Demographic data to existing CSV");
	subparser.add_argument("--MetaDem", required = True, type = str, help = "Source path of face images train/test");
	subparser.add_argument("--MetaSRC", required = True, type = str, help = "CSV meta data");
	subparser.set_defaults(func = LIB_DeepFace.InsertDemographics);

	# Analyze image directory and get meta data
	subparser = subparsers.add_parser("Demographics4Folder", description = "Insert demographic from exsiting image folder");
	subparser.add_argument("--SPath", required = True, type = str, default = "./DATA", help = "Source path of face images");
	subparser.add_argument("--JSON", required = True, type = str, default = "./deepface_metadata.json", help = "JSON meta data to save");
	subparser.add_argument("--N", required = False, default = -1, type = int, help = "Number of samples per subfolder, default all");
	subparser.add_argument("--os_png_tool", required = False, default = "magick", type = str, help = "Default png system converter tool");	
	subparser.set_defaults(func = LIB_DeepFace.Demographics4Folder);



	Options = parser.parse_args();
	
	print(str(Options) + "\n");
	
	Response = Options.func(Options);


if __name__ == "__main__":
	start = datetime.datetime.now()
	print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
	main();
	end = datetime.datetime.now()
	print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

	exectime= start-end
	print("Exectime: ",exectime.total_seconds() )
