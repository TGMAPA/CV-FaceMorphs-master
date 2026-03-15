# conda activate deepface
import argparse, datetime, LIB_DeepFace

def main():
	parser = argparse.ArgumentParser("FaceNet Bias Library");
	subparsers = parser.add_subparsers();

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

	subparser = subparsers.add_parser("Demographics4Folder", description = "Insert demographic from exsiting image folder");
	subparser.add_argument("--SPath", required = True, type = str, help = "Source path of face images");
	subparser.add_argument("--JSON", required = True, type = str, help = "JSON meta data to save");
	subparser.add_argument("--N", required = False, default = -1, type = int, help = "Number of samples per subfolder, default all");
	subparser.add_argument("--os_png_tool", required = False, default = "magick", type = str, help = "Default png system converter tool");	
	subparser.set_defaults(func = LIB_DeepFace.Demographics4Folder);

	Options = parser.parse_args();
	
	#Options = LIB_DeepFace.loadOptionsYAML(Options);

	print(str(Options) + "\n");

	Response = Options.func(Options);



if __name__ == "__main__":
	print("\n" + "\033[0;36m" + "[start] " + str(datetime.datetime.now()) + "\033[0m" + "\n");
	main();
	print("\n" + "\033[0;36m" + "[end] "+ str(datetime.datetime.now()) + "\033[0m" + "\n");
