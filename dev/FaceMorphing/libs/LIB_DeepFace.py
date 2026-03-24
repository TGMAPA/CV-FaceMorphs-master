# Import packages
import pandas, tqdm, glob, os, random, json, cv2
from types import SimpleNamespace

# Import modules
from deepface import DeepFace


def GetRace(Files, pbar, subsample = 10):
	# Some identities have 500 samples
	random.shuffle(Files)
	
	Race = {"asian": 0, "indian": 0, "black": 0, "white": 0, "middle eastern": 0, "latino hispanic": 0};
	Age = {"age" : 0};
	Gender = {"Woman" : 0, "Man" : 0};
	
	k = 1;
	
	for File in Files:
		if k >= subsample:
			break
		pbar.set_description( "Processing ...  %s" %(File) );
		#os.system("magick %s temp.png" %(File) ) # png files are needed
		os.system("convert %s temp.png" %(File) ) # png files are needed		
		try:
			objs = DeepFace.analyze( img_path = "temp.png", actions = ["age", "gender", "race"] )[0];
		except:
			continue
		obs = objs.get("race");
		for r in obs.keys():
			Race[r] += obs.get(r);
		
		
		Age["age"] += objs.get("age");


		obs = objs.get("gender");		
		for r in obs.keys():
			Gender[r] += obs.get(r);
				
		k += 1;
		
	for r in Race.keys():
		Race[r] /= k;
		
	for r in Gender.keys():
		Gender[r] /= k;

	Age["age"] /= k;
	
	return Race, Gender, Age;


def MetaDemographicsVGG2(Options):

	DF = pandas.read_csv(Options.CSV_meta);

	DF_ID = pandas.DataFrame();
	
	IDS = DF.CLASS_ID.unique();
	IDS.sort();
	
	DF_ID = DF_ID.assign( CLASS_ID =  IDS);	
	
	DF_ID = DF_ID.assign(Asian = 0);
	DF_ID = DF_ID.assign(Indian = 0);
	DF_ID = DF_ID.assign(African = 0);
	DF_ID = DF_ID.assign(Caucasian = 0);
	DF_ID = DF_ID.assign(MiddleEast = 0);
	DF_ID = DF_ID.assign(Latino = 0);
	DF_ID = DF_ID.assign(Male = 0);
	DF_ID = DF_ID.assign(Female = 0);
	DF_ID = DF_ID.assign(Age = 0);
	DF_ID = DF_ID.assign(Samples = 0);
	
	Races_1 = ["Asian","Indian","African","Caucasian","MiddleEast","Latino"];
	Races_2 = ["asian","indian","black","white","middle eastern","latino hispanic"];
	
	
	for k , row in (pbar := tqdm.tqdm(DF_ID.iterrows(), total = len(DF_ID) )):
		Files = glob.glob(Options.SPATH + "train/" + row["CLASS_ID"] + "/*"); # VGGF2 either come from train or test
		Files += glob.glob(Options.SPATH + "test/" + row["CLASS_ID"] + "/*");
		#
		Race, Gender, Age = GetRace(Files, pbar);
		
		print(Gender)
		
		for r1,r2 in zip(Races_1, Races_2):
			DF_ID.at[k, r1] = Race.get(r2)

		DF_ID.at[k, "Female"] = Gender.get("Woman");
		DF_ID.at[k, "Male"] = Gender.get("Man");
		DF_ID.at[k, "Age"] = Age.get("age");
		DF_ID.at[k, "Samples"] = len(Files);		
	
	print(DF_ID)
		
	DF_ID.to_csv(Options.CSV_meta.replace(".csv","_races.csv"));


def MetaDemographicsCELEBA(Options):

	DF = pandas.read_csv(Options.CSV_meta);
	
	DF_ID = pandas.DataFrame();
	
	IDS = DF.CLASS_ID.unique();
	IDS.sort();
	
	DF_ID = DF_ID.assign( CLASS_ID =  IDS);

	DF_ID = DF_ID.assign(Asian = 0);
	DF_ID = DF_ID.assign(Indian = 0);
	DF_ID = DF_ID.assign(African = 0);
	DF_ID = DF_ID.assign(Caucasian = 0);
	DF_ID = DF_ID.assign(MiddleEast = 0);
	DF_ID = DF_ID.assign(Latino = 0);
	DF_ID = DF_ID.assign(Male = 0);
	DF_ID = DF_ID.assign(Female = 0);
	DF_ID = DF_ID.assign(Age = 0);
	DF_ID = DF_ID.assign(Samples = 0);

	Races_1 = ["Asian","Indian","African","Caucasian","MiddleEast","Latino"];
	Races_2 = ["asian","indian","black","white","middle eastern","latino hispanic"];
	
	for k , row in (pbar := tqdm.tqdm(DF_ID.iterrows(), total = len(DF_ID) )):
		Files = list(DF[row["CLASS_ID"] == DF.CLASS_ID].File);
		Files = [Options.SPATH + File for File in Files];
		#
		Race, Gender, Age = GetRace(Files, pbar, Options.Subsample);

		for r1,r2 in zip(Races_1, Races_2):
			DF_ID.at[k, r1] = Race.get(r2)
		
		DF_ID.at[k, "Female"] = Gender.get("Woman");
		DF_ID.at[k, "Male"] = Gender.get("Man");
		DF_ID.at[k, "Age"] = Age.get("age");
		DF_ID.at[k, "Samples"] = len(Files);
		
	print(DF_ID)
	
	DF_ID.to_csv(Options.CSV_meta.replace(".csv","_races.csv"), index = False);	
	
	
def InsertDemographics(Options):
	DF_full = pandas.read_csv(Options.MetaSRC);
	
	DF_dem = pandas.read_csv(Options.MetaDem);
	
	for col in DF_dem:
		if col == "CLASS_ID":
			continue
		DF_full.insert(loc = len(DF_full.columns), column = col, value = 0)
	
	
	for k, row in tqdm.tqdm( DF_full.iterrows(), total = len(DF_full) ):
		sDF = DF_dem[DF_dem.CLASS_ID == row["CLASS_ID"]];
		for col in sDF:
			DF_full.at[k, col] = sDF[col].values[0];
	
	DF_full.to_csv(Options.MetaSRC.replace(".csv","_dem.csv"), index = False);

# Retrieve all leaf directories inside a source path.
def leafDirs(SPath):
	Folders = []
	for root, dirs, files in sorted(os.walk(SPath)):
		if not dirs:
			Folders.append(os.path.abspath(root))
	return Folders

# Generate png with magick command #Monday 23 March 2026 19:12:50 GMT by MAPA
def generate_magick_png(input_path, output_path, magick_cmd):
	try:
		os.system("%s %s %s" %(magick_cmd, input_path, output_path) )
	except:
		return False
	return True

# Create embeddings with deepFace
def Embeddings():
	pass

# ----- Single file demographic analyzer
# Get demographic metadata from a single file with magick (Original function)
def SingleSampleDemographic_magick(File, Options):
	#
	Demographics = {"File" : File.replace(Options.SPath, "") , "asian": 0, "indian": 0, "black": 0, "white": 0, "middle eastern": 0, "latino hispanic": 0, "age" : 0, "Woman" : 0, "Man" : 0};
	#
	try:
		os.system("%s %s temp.png" %(Options.os_png_tool, File) )
		objs = DeepFace.analyze( img_path = "temp.png", actions = ["age", "gender", "race"] )[0];
		os.system("rm temp.png")
	except:
		return None;
	#
	obs = objs.get("race");
	for r in obs.keys():
		Demographics[r] = float(obs.get(r));
	#
	Demographics["age"] = float(objs.get("age"));
	#
	obs = objs.get("gender");		
	for r in obs.keys():
		Demographics[r] = float(obs.get(r));
	#
	return Demographics;

# Get demographic metadata from a single file with opencv  #Tuesday 17 March 2026 23:30:50 GMT by MAPA
def SingleSampleDemographic_cv2(File, Options):

    Demographics = {
        "File": File.replace(Options.SPath, ""),
        "asian": 0, "indian": 0, "black": 0, "white": 0,
        "middle eastern": 0, "latino hispanic": 0,
        "age": 0, "Woman": 0, "Man": 0
    }

    try:
        img = cv2.imread(File)

        if img is None:
            return None

        objs = DeepFace.analyze(
            img_path=img,
            actions=["age", "gender", "race"],
            enforce_detection=False 
        )[0]

    except Exception as e:
        print(f"Error in {File}: {e}")
        return None

    # Process results
    for r in objs.get("race", {}).keys():
        Demographics[r] = float(objs["race"][r])

    Demographics["age"] = float(objs.get("age"))

    for r in objs.get("gender", {}).keys():
        Demographics[r] = float(objs["gender"][r])

    return Demographics


'''
██████   █████  ██████  ███████ ███████ ██████  
██   ██ ██   ██ ██   ██ ██      ██      ██   ██ 
██████  ███████ ██████  ███████ █████   ██████  
██      ██   ██ ██   ██      ██ ██      ██   ██ 
██      ██   ██ ██   ██ ███████ ███████ ██   ██ 
'''

# Get demographic metadata from a single file with a desired png convertor tool #Monday 23 March 2026 19:12:50 GMT by MAPA
def SingleSampleDemographic(Options):
	# Parameters
	input_file = Options.input_file
	temp_output_file = Options.temp_output_file
	os_png_tool = Options.os_png_tool
	remove_temp_file = Options.remove_temp_file
	#
	Demographics = {
        "File": input_file,
        "asian": 0, "indian": 0, "black": 0, "white": 0,
        "middle eastern": 0, "latino hispanic": 0,
        "age": 0, "Woman": 0, "Man": 0
    }
	#
	try:
		match os_png_tool:
			case "magick":
				# magick compression
				success = generate_magick_png(input_file, temp_output_file, "magick")
				if not success or not os.path.exists(temp_output_file):
					print(f"[error] PNG conversion failed: {input_file}")
					return None

				objs = DeepFace.analyze(
					img_path=temp_output_file,
					actions=["age", "gender", "race"],
					enforce_detection=False
				)[0]

				if remove_temp_file and os.path.exists(temp_output_file):
					os.remove(temp_output_file)
				
			case "cv2":
				# Opencv compression
				img = cv2.imread(input_file)

				if img is None:
					print(f"[error] OpenCV failed: {input_file}")
					return None

				# Exec deepface analyzer
				objs = DeepFace.analyze( img, actions = ["age", "gender", "race"] )[0];
			case _:
				print("Unknown png Compression...")
				return None
	except Exception as e:
		print(f"[exception] {input_file} : {e}")
		return None
	
	# Process results
	obs = objs.get("race");
	for r in obs.keys():
		Demographics[r] = float(obs.get(r));
	#
	Demographics["age"] = float(objs.get("age"));
	#
	obs = objs.get("gender");		
	for r in obs.keys():
		Demographics[r] = float(obs.get(r));
	#
	print(Demographics)
	return Demographics;

# Get demographic meta data as json from an image directory
def Demographics4Folder(Options):
	# Retrieve all leaf directories inside the source path.
	Folders = leafDirs(Options.SPath);
	
	Register = [];

	# Iterate through each folder found in the source path
	for Folder in Folders:
		print("Processing ... %s " %(Folder));
		
		# Retrieve all files inside the folder
		Files = glob.glob(Folder + "/*.*");
	
		if len(Files) == 0:
			continue
		
		# Shuffle files randomly to avoid sampling bias
		random.shuffle(Files);

		# List to store demographics extracted from sampled files
		Dem = [];

		# Number of samples to process
		N = None if Options.N == -1 else Options.N
		
		# Process only the first N files defined in Options.N
		# This acts as a sampling mechanism to limit processing time
		for File in Files[0:N]:
			print(f"---- Processing file: {File}")
			
			# Extract demographic attributes from file
			Demographics = SingleSampleDemographic(
				Options=SimpleNamespace(
					input_file=File,
					temp_output_file=f"temp_{os.path.basename(File)}.png",
					os_png_tool=Options.os_png_tool,
					remove_temp_file=True,
				)
			)

			# If the function returns no result, log the issue and skip the file
			if Demographics is None:
				print("[error] File : %s produced no output." %(File));
				continue

			# Append valid demographic result to the folder list
			Dem.append(Demographics)

		# Store folder-level results including:
		# - folder name (relative to source path)
		# - total number of files in the folder
		# - demographics extracted from sampled images
		Register.append( {
			"Folder" : Folder.replace(Options.SPath, ""), 
			"Samples" : len(Files), 
			"os_png_tool" : Options.os_png_tool,
			"Demographics" : Dem
			} 
		);
	
	# Convert the collected results into a formatted JSON string
	json_obj = json.dumps(Register, indent = 3);

	# Write the JSON output to the specified file
	with open(Options.JSON, "w") as fid:
		fid.write(json_obj)	
	fid.close()

	print("Directory Process was completed successfully")
	print(f"Processed: {len(Dem)} / {len(Files)}")

# Export generated json file with dataset's DeepFace-Demographics into structured csv  #Monday 23 March 2026 16:54:30 GMT by MAPA
def transform_deepFacejson2csv(Options):
	# Read parameters
	jsonPath = Options.jsonPath
	csvPath = Options.csvPath
	sourceDataPath = Options.sourceDataPath

	# Load JSON
	try:
		with open(jsonPath, "r") as f:
			data = json.load(f)
	except Exception as e:
		print(f"Error reading JSON: {e}")
		return

	print("JSON file:", jsonPath)
	print("CSV file:", csvPath)

	# DAtaframe rows
	rows = []

	for Register in data:
		for d in Register.get("Demographics", []):
			if d is None:
				continue

			rows.append({
				"file": sourceDataPath + '/'+ d.get("File").split("/")[-1],
				"Asian": d.get("asian", 0),
				"Indian": d.get("indian", 0),
				"African": d.get("black", 0),
				"Caucasian": d.get("white", 0),
				"MiddleEast": d.get("middle eastern", 0),
				"Latino": d.get("latino hispanic", 0),
				"Age": d.get("age", 0),
				"Female": d.get("Man", 0),
				"Male": d.get("Woman", 0)
			})

	DF = pandas.DataFrame(rows)
	DF.to_csv(csvPath, index=False)

	print("Dataframe was exported successfully...")
