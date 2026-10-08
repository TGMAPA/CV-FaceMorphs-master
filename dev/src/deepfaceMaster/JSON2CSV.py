import pandas, json, tqdm

fid = open('/home/nmp56/Documents/BUPT/BUPT_Global_Demographic.json')
data = json.load(fid)
fid.close()


DF = pandas.DataFrame(columns = ["CLASS_ID", "Asian", "Indian", "African", "Caucasian", "MiddleEast", "Latino", "Male", "Female", "Age", "Samples"]);

for Register in tqdm.tqdm( data, total = len(data) ):
	Demographics = Register.get("Demographics");

	n = len(Demographics);
	
	n = n + 1 if n == 0 else n;
	
	Asian, Indian, African, Caucasian, MiddleEast, Latino, Male, Female, Age = 0,0,0,0,0,0,0,0,0;
	
	for d in Demographics:
		Asian += d.get("asian");
		Indian += d.get("indian");
		African += d.get("black");
		Caucasian += d.get("white");
		MiddleEast += d.get("middle eastern");
		Latino += d.get("latino hispanic");
		Male += d.get("Man");
		Female += d.get("Woman");
		Age += d.get("age");
		
	
	DF = pandas.concat([DF, pandas.DataFrame([{
		"CLASS_ID" : Register.get("Folder").split("/")[-1], 
		"Asian" : Asian/n , 
		"Indian" : Indian/n , 
		"African" : African/n , 
		"Caucasian" : Caucasian/n , 
		"MiddleEast" : MiddleEast/n , 
		"Latino" : Latino/n , 
		"Male" : Male/n , 
		"Female" : Female/n , 
		"Age" : Age/n , 
		"Samples" : Register.get("Samples")
		}])], ignore_index = True);	
	
print(DF)	

DF.to_csv("BUPT_Global_Demographic.csv", index = False)

