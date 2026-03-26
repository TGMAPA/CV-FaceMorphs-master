import os, json, datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# DeepFace is used to extract facial embeddings 
from libs import LIB_DeepFace

# Comparation between models
# (FaceNet512, ArcFace,GhostFaceNet)


#Wednesday 25 March 2026 22:04:30 GMT by MAPA
def generate_embeddings(input_img, models:list, logs=False):
    assert os.path.exists(input_img), "Invalid input_image"

    embeddings = {}

    for model in models:
        result = LIB_DeepFace.GenerateEmbeddingFromImage(input_path=input_img, model=model)

        # result es una lista de dicts, tomamos el primer elemento y extraemos 'embedding'
        if isinstance(result, list) and len(result) > 0 and 'embedding' in result[0]:
            embeddings[model] = np.array(result[0]['embedding'], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected embedding format for model {model}: {result}")

    if logs:
        for key in embeddings.keys():
            print("============ ", key)
            print(embeddings[key])

    print("Embeddings were successfully generated...")
    return embeddings

def cosine_sim(a, b):
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return cosine_similarity(a, b)[0][0]

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def manhattan_dist(a, b):
    return np.sum(np.abs(a - b))

def compareEmbeddings(embeddings):
    models = list(embeddings.keys())

    # Compare
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            m1, m2 = models[i], models[j]
            emb1, emb2 = embeddings[m1], embeddings[m2]
            print(f"Comparing {m1} vs {m2}:")
            print(f"  Cosine Similarity: {cosine_sim(emb1, emb2):.4f}")
            print(f"  Euclidean Distance: {euclidean_dist(emb1, emb2):.4f}")
            print(f"  Manhattan Distance: {manhattan_dist(emb1, emb2):.4f}\n")

def main():
    test_data_path = "./DATA/img5.jpg"
    write_in_file = True

    models2test = [
        "Facenet512",
        "ArcFace",
        "GhostFaceNet"

    ]
        
    # Generate embeddings
    embeddings = generate_embeddings(test_data_path, models2test)

    # Compare embeddings
    compareEmbeddings(embeddings)

    if write_in_file:
        #Open a file in write mode ('w') and dump the data
        with open('./Embeddings_Analysis/emebeddings.txt', 'w', encoding='utf-8') as f:
            f.write(str(embeddings))



if __name__ == "__main__":
	start = datetime.datetime.now()
	print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
	main();
	end = datetime.datetime.now()
	print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

	exectime= end - start
	print("Exectime: ",exectime.total_seconds() )