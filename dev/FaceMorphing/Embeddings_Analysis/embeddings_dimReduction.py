import pandas as pd
import seaborn as sns
import datetime, json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding


def string_to_array(text):
    return np.array(json.loads(text))

def createDataset(
        demographic_csv_path, 
        embeddings_json_path, 
        create_csv = True, 
        dataset_csv_path = "joined_df.csv"
        ):
    print("Creating Dataset...")

    if not create_csv:
        # Assume dataset csv existance, then read only
        try:
            # Read csv converting embedding values into np.array
            df = pd.read_csv(dataset_csv_path, converters={'embedding': string_to_array})
        except:
            print(f"Dataset wasn´t read successfully in expected path: {dataset_csv_path} ...")
            return None

        print("Dataset was loaded successfully...")
        return df

    # Match embeddings with images and extract demographic labels en csv
    df_demo = pd.read_csv(demographic_csv_path)
    
    # Open the file in read mode
    df_embeddings = pd.read_json(embeddings_json_path)

    # If column names differ between DataFrames
    df = pd.merge(df_demo, df_embeddings, left_on='file', right_on='image_path')

    # Drop columns
    df.drop(['image_path', 'Model'], axis=1, inplace=True)

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

    # Get each sample's dominant race
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
    df.drop(columns=columns2delete, inplace=True)

    # Save dataframe
    if create_csv: df.to_csv(dataset_csv_path, index=False)

    print("Dataset was generated successfully...")

    return df

def plot_embedding(X_transformed, labels, title, filename):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], hue=labels, palette='viridis', alpha=0.7)
    plt.title(title)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Graph '{title}' saved as: {filename}")

    plt.show()

def main():
    # Demographics dataset
    demographic_csv_path = "../../data/Demographics/ffhq_real_demographics_meta_data_structured.csv"

    # Embeddings dataset
    embeddings_json_path = "../../data/Embeddings/ffhq_real_deepface_embeddings_metadata_Facenet512.json"

    # Create dataset with cols:
        # file
        # Age
        # embedding
        # Dominant_Race
        # Dominant_Gender
    dataset = createDataset(
        demographic_csv_path, 
        embeddings_json_path, 
        create_csv=False, 
        dataset_csv_path='ffhq_real_embeddings_and_demographics.csv'
        )
    
    # Split dataset
    # X: dataset_embeddings
    X = np.array(dataset['embedding'].tolist())

    # y: labels_demograficas
    y = dataset[['Age', 'Dominant_Race', 'Dominant_Gender']]

    print("----- Extracted data:")
    print("X.shape: ", X.shape)
    print("Y.head: ", y.shape)

    print("===== Exec. DimReduction:")

    # -- Dimen Reduction
    # PCA
    print("----- Exec. PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Graph for each demographic label
    for col in y.columns:
        title = f"PCA Visualization - Colored by {col}"
        plot_embedding(X_pca, y[col], title, title.strip()+".png")

    # t-SNE
    print("----- Exec. t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Graph for each demographic label
    for col in y.columns:
        title = f"t-SNE Visualization - Colored by {col}"
        plot_embedding(X_tsne, y[col], title, title.strip()+".png")

    # Isomap
    print("----- Exec. Isomap...")

    # Reduce samples
    MAX_SAMPLES = len(X)/2 
    indices = np.random.choice(len(X), MAX_SAMPLES, replace=False)
    X_sub = X[indices]
    y_sub = y.iloc[indices] 
    
    isomap = Isomap(n_components=2, n_neighbors=10)
    X_isomap = isomap.fit_transform(X_sub)

    # Graph for each demographic label
    for col in y.columns:
        title = f"Isomap Visualization - Colored by {col}"
        plot_embedding(X_isomap, y_sub[col], title, title.strip()+".png")

    # LLE
    # print("----- Exec. LocallyLinearEmbedding...")
    # lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
    # X_lle = lle.fit_transform(X)

    # # Graph for each demographic label
    # for col in y.columns:
    #     title = f"LLE Visualization - Colored by {col}"
    #     plot_embedding(X_lle, y[col], title, title.strip()+".png")

    return None


if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n");
    main();
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n");

    exectime= end - start
    print("Exectime: ",exectime.total_seconds() )
