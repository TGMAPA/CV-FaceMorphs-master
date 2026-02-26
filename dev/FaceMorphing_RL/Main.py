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
from libs import LIB_FaceMorph

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

	subparser = subparsers.add_parser("ImageMorph", description = "Image 2 Image morphs");
	subparser.add_argument("--Sb1", required = False, type = str, default = "./DATA/Sb1.png", help = "Subject one's face"),
	subparser.add_argument("--Sb2", required = False, type = str, default = "./DATA/Sb2.png", help = "Subject two's face"),
	subparser.add_argument("--Morph", required = False, type = str, default = "MorphedFace.png", help = "Morph output <MorphedFace.png>"),	
	subparser.add_argument("--Alpha", type = float, default = 0.5, help = "Blending factor");
	subparser.set_defaults(func = LIB_FaceMorph.MorphFace);

	Options = parser.parse_args();
	
	print(str(Options) + "\n");
	
	Response = Options.func(Options);



if __name__ == "__main__":
	print("\n" + "\033[0;34m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
	main();
	print("\n" + "\033[0;34m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");
