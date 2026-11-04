import numpy as np
import json, os
import pandas as pd
import hashlib

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
	
#Tue 07 April 13:25:35 GMT by MAPA
def string_to_array(text):
    return np.array(json.loads(text))


#Tue 03 November 18:33:58 GMT by MAPA
def valid_json(path, expected_type=None):
    """Comprueba que un JSON exista, pueda leerse y tenga el tipo esperado."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    try:
        with open(path, "r") as f:
            data = json.load(f)

        if expected_type is not None and not isinstance(data, expected_type):
            return False

        return True
    except (json.JSONDecodeError, OSError):
        return False

#Tue 03 November 18:33:58 GMT by MAPA
def valid_csv(path, min_rows=1):
    """Comprueba que un CSV exista, pueda leerse y tenga registros."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False

    try:
        df = pd.read_csv(path)
        return len(df) >= min_rows
    except Exception:
        return False

#Tue 03 November 18:33:58 GMT by MAPA
def count_demographic_samples(json_path):
    """Cuenta las imágenes registradas en el JSON demográfico."""
    if not valid_json(json_path, list):
        return 0

    with open(json_path, "r") as f:
        data = json.load(f)

    return sum(
        len(entry.get("Demographics", []))
        for entry in data
        if isinstance(entry, dict)
    )

#Tue 03 November 18:33:58 GMT by MAPA
def valid_embeddings_json(path):
    """Comprueba que el JSON de embeddings contenga embeddings utilizables."""
    if not valid_json(path, list):
        return False

    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        return False

    return all(
        isinstance(item, dict)
        and isinstance(item.get("embedding"), list)
        and len(item["embedding"]) > 0
        and item.get("image_path")
        for item in data
    )




def remove_duplicate_embeddings(
    input_json_path,
    output_json_path,
    duplicates_csv_path
):
    """
    Removes records containing exactly identical embeddings.

    The first occurrence is preserved.
    Original JSON and image files remain untouched.
    """

    with open(input_json_path, "r") as f:
        data = json.load(f)

    seen_embeddings = {}
    unique_data = []
    duplicates = []

    for item in data:

        embedding = item.get("embedding")

        if not isinstance(embedding, list) or not embedding:
            raise ValueError(
                f"Invalid embedding in: {item.get('image_path')}"
            )

        embedding_array = np.asarray(embedding, dtype=np.float64)

        if not np.all(np.isfinite(embedding_array)):
            raise ValueError(
                f"Non-finite embedding in: {item.get('image_path')}"
            )

        embedding_hash = hashlib.sha256(
            embedding_array.tobytes()
        ).hexdigest()

        if embedding_hash in seen_embeddings:

            original = seen_embeddings[embedding_hash]

            duplicates.append({
                "duplicate_image": item.get("image_path"),
                "original_image": original.get("image_path"),
                "embedding_hash": embedding_hash
            })

        else:
            seen_embeddings[embedding_hash] = item
            unique_data.append(item)

    # Save deduplicated embeddings
    with open(output_json_path, "w") as f:
        json.dump(unique_data, f, indent=4)

    # Save duplicate report
    duplicates_df = pd.DataFrame(
        duplicates,
        columns=[
            "duplicate_image",
            "original_image",
            "embedding_hash"
        ]
    )

    duplicates_df.to_csv(duplicates_csv_path, index=False)

    print("\n=== EMBEDDING DEDUPLICATION ===")
    print(f"Original samples      : {len(data)}")
    print(f"Unique samples        : {len(unique_data)}")
    print(f"Duplicates removed    : {len(duplicates)}")
    print(f"Deduplicated JSON     : {output_json_path}")
    print(f"Duplicates report     : {duplicates_csv_path}")

    return output_json_path, len(unique_data), duplicates_df