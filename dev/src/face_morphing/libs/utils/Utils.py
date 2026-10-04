# Write rows in csv  #Thursday 26 March 2026 08:41:50 GMT by MAPA
def writeInCsv(file, body:list):

	string = ""

	for i in range(len(body)):
		newitem = body[i] if type(body[i]) == str else str(body[i])
		string+= newitem

		if i != len(body)-1:
			string+=","
		else:
			string+="\n"

	file.write(string)
	

