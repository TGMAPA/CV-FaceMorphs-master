# Libraries
import numpy as np
import pandas as pd
import os

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

    # Convert every path to absolute paths for exact coincidence in merge
    df_demo['join_key'] = df_demo['file'].apply(lambda x: os.path.abspath(str(x)))
    df_embeddings['join_key'] = df_embeddings['image_path'].apply(lambda x: os.path.abspath(str(x)))

    # Join dataframes using the standardized route with the new key 
    df = pd.merge(df_demo, df_embeddings, on='join_key')

    if df.empty:
        raise ValueError(
            "The resultant dataframe is empty. "
            "Verify that every existant relative path in any of both dataframes point to the correct path from the running root directory"
        )

    # Drop columns
    df.drop(columns=['join_key', 'image_path', 'Model'], errors='ignore', inplace=True)

    # race columns index 1-7
    race_columns = ["Asian","Indian","African","Caucasian", "MiddleEast","Latino"]

    # Gender columns index 8-9
    gender_columns = ["Female", "Male"]

    # Dataframe length
    DF_len = df.shape[0]

    # New Column = Dominant_Race
    dominant_race = []

    # New Column = Dominant Gender
    dominant_gender = []

    # Get each sample's dominant race and gender
    for i in range(DF_len):
        # -- Row
        # Extract race's probs
        race_row_data = np.array(df.iloc[i, 1:7].values)

        # Extract dominant race
        race_index_max = race_row_data.argmax()

        # Sample Dominant race
        sample_dominant_race = race_columns[race_index_max]


        # -- Gender
        # Extract gender's probs
        gender_row_data = np.array(df.iloc[i, 8:10].values)

        # Extract dominant gender 
        gender_index_max = gender_row_data.argmax()

        # Sample dominant race
        sample_dominant_gender = gender_columns[gender_index_max]


        # add each values to arrays
        dominant_race.append(sample_dominant_race)
        dominant_gender.append(sample_dominant_gender)

    # Add new columns to df
    df['Dominant_Race'] = dominant_race
    df['Dominant_Gender'] = dominant_gender

    # Extra columns to delete
    extraCols2Delete = ["facial_area", "face_confidence"]

    # Join columns to delete from dataframe
    columns2delete = race_columns + gender_columns + extraCols2Delete

    # Delete original columns
    df.drop(columns=columns2delete, errors='ignore', inplace=True)

    # Save dataframe
    if create_csv: 
        df.to_csv(dataset_csv_path, index=False)

    print("Dataset was generated successfully...")

    return df