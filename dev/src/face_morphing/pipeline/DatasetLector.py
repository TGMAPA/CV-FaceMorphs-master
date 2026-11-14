# Libraries
import numpy as np
import pandas as pd
import shutil, os, json
from pathlib import Path

# Modules
from face_morphing.libs.utils import Utils


#Tue 07 April 13:25:35 GMT by MAPA
def createDataset(demographic_csv_path, embeddings_json_path, create_csv = True, dataset_csv_path = "joined_df.csv"):
    print("Creating Dataset...")

    if not create_csv:
        # Assume dataset csv existance, then read only
        try:
            # Read csv converting embedding values into np.array
            df = pd.read_csv(dataset_csv_path, converters={'embedding': Utils.string_to_array})
        except:
            print(f"Dataset wasn´t read successfully in expected path: {dataset_csv_path} ...")
            return None

        print("Dataset was loaded successfully...")
        return df

    # Match embeddings with images and extract demographic labels en csv
    df_demo = pd.read_csv(demographic_csv_path)
    
    # Open the file in read mode
    df_embeddings = pd.read_json(embeddings_json_path)

    # Validate required columns
    if "file" not in df_demo.columns:
        raise ValueError("Demographic CSV is missing the 'file' column.")

    if "image_path" not in df_embeddings.columns:
        raise ValueError("Embeddings JSON is missing the 'image_path' column.")

    # Convert every path to absolute paths for exact coincidence in merge
    df_demo['join_key'] = df_demo['file'].apply(lambda x: os.path.abspath(str(x)))
    df_embeddings['join_key'] = df_embeddings['image_path'].apply(lambda x: os.path.abspath(str(x)))

    # Check duplicate paths
    if df_demo["join_key"].duplicated().any():
        raise ValueError("Duplicate image paths found in demographic CSV.")

    if df_embeddings["join_key"].duplicated().any():
        raise ValueError("Duplicate image paths found in embeddings JSON.")

    # Diagnose matching
    demo_paths = set(df_demo["join_key"])
    embedding_paths = set(df_embeddings["join_key"])

    common_paths = demo_paths.intersection(embedding_paths)

    print(f"Demographic samples : {len(df_demo)}")
    print(f"Embedding samples   : {len(df_embeddings)}")
    print(f"Matching samples    : {len(common_paths)}")

    if not common_paths:
        raise ValueError(
            "No matching image paths found between demographics and embeddings."
        )

    # Join dataframes using the standardized route with the new key 
    df = pd.merge(
        df_demo,
        df_embeddings,
        on="join_key",
        how="inner",
        validate="one_to_one"
    )

    if df.empty:
        raise ValueError(
            "The resultant dataframe is empty. "
            "Verify that every existant relative path in any of both dataframes point to the correct path from the running root directory"
        )

    # Keep the actual path used to generate embeddings
    df["file"] = df["image_path"]

    # Remove unnecessary columns
    df.drop(
        columns=["join_key", "image_path", "Model"],
        errors="ignore",
        inplace=True
    )

    # Dominant race
    race_columns = [
        "Asian", "Indian", "African",
        "Caucasian", "MiddleEast", "Latino"
    ]

    gender_columns = ["Female", "Male"]

    df["Dominant_Race"] = df[race_columns].idxmax(axis=1)
    df["Dominant_Gender"] = df[gender_columns].idxmax(axis=1)

    # Remove original demographic probability columns
    extra_columns = ["facial_area", "face_confidence"]

    df.drop(
        columns=race_columns + gender_columns + extra_columns,
        errors="ignore",
        inplace=True
    )

    # Save dataset
    if create_csv:
        df.to_csv(dataset_csv_path, index=False)

    print(f"Dataset was generated successfully: {len(df)} samples")

    return df

#Tue 03 November 18:33:58 GMT by MAPA

def create_identity_filtered_dataset(
    raw_dir: str,
    identity_csv: str,
    output_dir: str,
    identity_col_name: str = "ID",
    file_path_col_name: str = "File",
    max_images_per_identity: int = 3,
    random_state: int = 42,
):
    """
    Creates a derived dataset with a maximum number of images
    per identity.

    Original images are never modified or deleted.
    """

    raw_dir = Path(raw_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    identity_csv = Path(identity_csv).expanduser().resolve()

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {raw_dir}")

    if not identity_csv.is_file():
        raise FileNotFoundError(f"Identity CSV not found: {identity_csv}")

    if max_images_per_identity < 1:
        raise ValueError("max_images_per_identity must be >= 1")

    # Create output directory before any write operation.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load identity metadata.
    identity_df = pd.read_csv(identity_csv)

    required_columns = [identity_col_name, file_path_col_name]
    missing_columns = [
        col for col in required_columns
        if col not in identity_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in identity CSV: {missing_columns}"
        )

    identity_df = identity_df[
        [file_path_col_name, identity_col_name]
    ].copy()

    # Normalize and validate records.
    identity_df[file_path_col_name] = (
        identity_df[file_path_col_name]
        .astype(str)
        .str.strip()
    )

    identity_df = identity_df.dropna(
        subset=[identity_col_name]
    )

    identity_df = identity_df[
        identity_df[file_path_col_name].ne("")
    ].copy()

    # Resolve source paths.
    identity_df["Source"] = identity_df[
        file_path_col_name
    ].apply(
        lambda filename: (raw_dir / filename).resolve()
    )

    identity_df["Exists"] = identity_df["Source"].apply(
        lambda path: path.is_file()
    )

    missing_count = (~identity_df["Exists"]).sum()

    if missing_count:
        print(
            f"[Identity Filtering] Warning: "
            f"{missing_count} images were not found."
        )

    identity_df = identity_df[
        identity_df["Exists"]
    ].copy()

    if identity_df.empty:
        raise ValueError(
            "No valid images were found. Check raw_dir and CSV paths."
        )

    # Ensure each image belongs to only one identity.
    identity_counts = (
        identity_df.groupby(file_path_col_name)[identity_col_name]
        .nunique()
    )

    conflicting_files = identity_counts[
        identity_counts > 1
    ].index.tolist()

    if conflicting_files:
        raise ValueError(
            "Some images have multiple identity assignments: "
            f"{conflicting_files[:10]}"
        )

    identity_df = identity_df.drop_duplicates(
        subset=[file_path_col_name]
    )

    # Reproducible selection: up to K images per identity.
    selected_df = (
        identity_df
        .groupby(identity_col_name, group_keys=False, sort=True)
        .sample(
            n=min(max_images_per_identity, len(identity_df)),
            random_state=random_state
        )
        if identity_df[identity_col_name].nunique() == 1
        else None
    )

    # Correct per-identity sampling, including groups with fewer than K.
    selected_df = (
        identity_df
        .groupby(identity_col_name, group_keys=False, sort=True)
        .apply(
            lambda group: group.sample(
                n=min(len(group), max_images_per_identity),
                random_state=random_state
            )
        )
        .reset_index(drop=True)
    )

    # Prepare output directory.
    filtered_dir = output_dir / "images"

    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)

    filtered_dir.mkdir(parents=True, exist_ok=True)

    # Copy selected images, preserving their relative structure.
    for row in selected_df.itertuples(index=False):

        source_path = Path(row.Source)
        relative_path = Path(getattr(row, file_path_col_name))
        destination_path = filtered_dir / relative_path

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source image disappeared: {source_path}"
            )

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(source_path, destination_path)

    # Save selected metadata.
    selected_metadata = selected_df[
        [file_path_col_name, identity_col_name]
    ].copy()

    filtered_csv_path = (
        output_dir / "identity_filtered_dataset.csv"
    )

    selected_metadata.to_csv(
        filtered_csv_path,
        index=False
    )

    # Summary.
    print("\n[Identity Filtering Completed]")
    print(f"Original images with valid identity: {len(identity_df)}")
    print(f"Unique identities: {identity_df[identity_col_name].nunique()}")
    print(f"Selected images: {len(selected_metadata)}")
    print(f"Maximum images per identity: {max_images_per_identity}")
    print(f"Output directory: {filtered_dir}")
    print(f"Identity metadata: {filtered_csv_path}")

    return str(filtered_dir), str(filtered_csv_path), selected_metadata


#Tue 03 November 18:33:58 GMT by MAPA
def add_identity_to_demographics_json(
    json_path: str,
    identity_metadata
):
    """
    Enriches Demographics JSON records with identity IDs.

    Matches images using their filename, even when the
    JSON contains absolute file paths.
    """

    json_path = Path(json_path)

    # Build filename -> identity mapping
    identity_map = dict(
        zip(
            identity_metadata["File"].astype(str).str.strip(),
            identity_metadata["ID"]
        )
    )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Unexpected JSON structure: expected a list of folders."
        )

    updated = 0
    unmatched = []

    for folder_data in data:

        demographics = folder_data.get("Demographics", [])

        for sample in demographics:

            file_path = sample.get("File")

            if not file_path:
                unmatched.append(file_path)
                continue

            # Extract filename from absolute or relative path
            filename = Path(file_path).name

            if filename in identity_map:
                sample["ID"] = identity_map[filename]
                updated += 1
            else:
                unmatched.append(filename)

    # Save enriched JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n[Demographic JSON Identity Enrichment]")
    print(f"Samples with identity assigned: {updated}")
    print(f"Samples without identity assigned: {len(unmatched)}")

    if unmatched:
        print(f"Unmatched samples (first 10): {unmatched[:10]}")

    return data
