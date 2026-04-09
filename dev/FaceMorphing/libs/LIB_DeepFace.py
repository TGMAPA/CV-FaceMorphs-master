# Import packages
import pandas, tqdm, glob, os, random, json, cv2, sys
from types import SimpleNamespace
from multiprocessing import Pool, cpu_count
import tensorflow as tf

repo_path = 'libs/utils/'
sys.path.append(repo_path)

# Import modules
from deepface import DeepFace

import Utils


DEEPFACE_MODELS = [
		"VGG-Face", 
		"Facenet", 
		"Facenet512", 
		"OpenFace", 
		"DeepFace", 
		"DeepID", 
		"ArcFace", 
		"Dlib", 
		"SFace",
		"GhostFaceNet",
	]


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

# Create embeddings with deepFace  #Tuesday 24 March 2026 11:55:40 GMT by MAPA
def GenerateEmbeddingFromImage(input_path, model, detector_backend = "opencv"):
	try:
		#Generate embedding
		embedding_objs = DeepFace.represent(
			img_path = input_path,
			model_name = model,
			detector_backend = detector_backend,
		)
	except Exception as error:
		return False, error

	# Return embedding_objs with format: 
	'''
	[
		{
		'embedding': [], 
		'facial_area': {
			'x': 177, 
			'y': 211, 
			'w': 680, 
			'h': 680, 
			'left_eye': (629, 480), 
			'right_eye': (375, 478)
		}, 
		'face_confidence': 0.93
		}
	]
	'''
	return True, embedding_objs

# Initialize GPU 
def init_worker():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

# Workers
def process_file(args):
    file, Options = args

    file_path = os.path.join(Options.SPath, file)

    try:
        status, output = GenerateEmbeddingFromImage(
            input_path=file_path,
            model=Options.model,
            detector_backend="skip"
        )

        if not status:
            return (file_path, False, output, None)

        # Extract embedding
        emb = output[0]
        emb["Model"] = Options.model
        emb["image_path"] = file_path

        return (file_path, True, "NA", emb)

    except Exception as e:
        return (file_path, False, str(e), None)

# Secuential Directory embedding generation
def secuentialDirectoryEmbeddingGeneration(
		csv_status_file,
		files,
		Options
		):
	
	output_obj = []

	# Row id
	id = 0

	# Bad generation Counter
	errors = 0

	with open(csv_status_file, "w") as csv_file:
		# write headers
		Utils.writeInCsv(csv_file, ["id", "path", "wasGenerationSuccessfull", "error"])

		for file in files:
			# Complete file path
			file_path = Options.SPath+"/" + file
			print("Proccessing file ",id," in "+ file_path +" ...")

			#Generate embedding
			status, output = GenerateEmbeddingFromImage(
								input_path=file_path,
								model=Options.model,
								detector_backend="skip"
							)
			
			if not status:
				# Write log in csv - INCOMPLETE GENERATION 
				Utils.writeInCsv(csv_file, [id, file_path, status, output])

				# Update
				errors+=1
				id+=1
				continue

			# write log in cs
			Utils.writeInCsv(csv_file, [id, file_path, status, "NA"])

			# Extend obj with extra meta data
			output[0]["Model"] = Options.model
			output[0]["image_path"] = file_path

			# Append embedding in json_obj
			output_obj.append(output[0])

			# Update id
			id+=1

	# Convert the collected results into a formatted JSON string
	json_obj = json.dumps(output_obj, indent = 3);

	return json_obj, errors

# GPU Accelerated Directory embedding generation
def GPUAccDirectoryEmbeddingGeneration(
		csv_status_file,
		files,
		Options
		):
	
	output_obj = []

	# Bad generation Counter
	errors = 0
	
	# INitialize GPU Pool
	with Pool(processes=Options.n_processes, initializer=init_worker) as p:

		args = [(file, Options) for file in files]

		with open(csv_status_file, "w") as csv_file:

			Utils.writeInCsv(csv_file, ["id", "path", "wasGenerationSuccessfull", "error"])

			for id, (file_path, status, error_msg, emb) in enumerate(
				p.imap_unordered(process_file, args)
			):

				print(f"Processing {id} -> {file_path}", flush=True)

				if not status:
					Utils.writeInCsv(csv_file, [id, file_path, status, error_msg])
					errors += 1
					continue

				Utils.writeInCsv(csv_file, [id, file_path, status, "NA"])
				output_obj.append(emb)

	# Convert the collected results into a formatted JSON string
	output_obj = json.dumps(output_obj, indent = 3);

	return output_obj, errors


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
				objs = DeepFace.analyze( img, actions = ["age", "gender", "race"], enforce_detection=False )[0];
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
	#print(Demographics)
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

		count = 0
		
		# Process only the first N files defined in Options.N
		# This acts as a sampling mechanism to limit processing time
		for File in Files[0:N]:
			print(f" Processing file: {count} {File}")
			
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

			count +=1

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

# Generate an embedding from a single image using deepFace and pretrained models  #Tuesday 24 March 2026 21:49:40 GMT by MAPA
def GenerateSingleEmbedding(Options):
	assert Options.model in DEEPFACE_MODELS, "Invalid model, choose from: " + str(DEEPFACE_MODELS);
	assert os.path.exists(Options.input_path), "Input Image path not found " + Options.input_path

	#Generate embedding
	embedding_objs = GenerateEmbeddingFromImage(
						input_path=Options.input_path,
						model=Options.model
					)
	
	if embedding_objs is None: 
		print("Embedding wasn't generated successfully...")
		return None

	print(embedding_objs)
	print("Embedding was successfully generated...")

# Generate embeddings into json an image directory using deepFace and pretrained models  #Tuesday 24 March 2026 21:54:40 GMT by MAPA
def GenerateDirectoryEmbeddings(Options):
	assert Options.model in DEEPFACE_MODELS, "Invalid model, choose from: " + str(DEEPFACE_MODELS);
	assert os.path.exists(Options.SPath), "Source Images Directory Path not found " + Options.SPath

	# Csv for keeping generation control
	csv_status_file = Options.csv_status_file.split("/")[-1].split(".")[0] +"_"+ Options.model + ".csv"

	# Json that stores every generated embedding
	output_json_path = Options.JSON.split("/")[-1].split(".")[0] +"_"+ Options.model + ".json"
	
	# Read directory
	all_files_path = sorted(os.listdir(Options.SPath))

	if not Options.gpuAcc:
		# Execute process - secuential
		output_obj, errors = secuentialDirectoryEmbeddingGeneration(csv_status_file, all_files_path, Options)
	else:
		# Execute process - gpuAcc
		output_obj, errors = GPUAccDirectoryEmbeddingGeneration(csv_status_file, all_files_path, Options)

	# Dump data in json
	with open(output_json_path, "w") as file:
		file.write(output_obj)

	print(f"Embeddings generated. Errors: {errors}")
	print("Embeddings were successfully generated...")
