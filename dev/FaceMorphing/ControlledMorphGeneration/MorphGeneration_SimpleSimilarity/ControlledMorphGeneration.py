import pandas as pd
import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from argparse import Namespace
from libs import LIB_FaceMorph, LIB_MorphGAN

#Tuesday 28 April 2026 22:14:50 GMT by MAPA

def morph_GAN(file1, file2):
    pass

def process_controlled_generation(csv_path, output_dir_path="./results/morphed_faces_dense/", alpha=0.5):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
        print(f"Created directory: {output_dir_path}")

    print(f"Reading pairs from: {csv_path}")
    df_pairs = pd.read_csv(csv_path)

    # Iteration
    total_pairs = len(df_pairs) 
    print(f"Starting composite morphing process for {total_pairs} pairs...\n")

    for index, row in df_pairs.iterrows():
        # Temporary path for the standalone morph before composting
        temp_morph_path = os.path.join(output_dir_path, f"temp_{index}.png")
        # Final path for the composite figure
        final_composite_path = os.path.join(output_dir_path, f"composite_morph_{index:03d}.png")

        # Prepare parameters for the GAN library
        params = Namespace(
            Sb1 = row['file_1'], 
            Sb2 = row['file_2'], 
            Morph = temp_morph_path,
            Alpha = alpha
        )

        try:
            # Generate the morph image using your library
            LIB_MorphGAN.MorphFace(params)
            
            # Create the composite figure (Original 1 | Original 2 | Result)
            fig, axes = plt.subplots(1, 3, figsize=(15, 6))
            
            # Load images
            img1 = mpimg.imread(row['file_1'])
            img2 = mpimg.imread(row['file_2'])
            morph_img = mpimg.imread(temp_morph_path)
            
            # Plot Image 1
            axes[0].imshow(img1)
            axes[0].set_title(f"Source 1\n({row['Race_1']} {row['Gender_1']})\nPath: {row['file_1']}")
            axes[0].axis('off')
            
            # Plot Image 2
            axes[1].imshow(img2)
            axes[1].set_title(f"Source 2\n({row['Race_2']} {row['Gender_2']})\nPath: {row['file_2']}")
            axes[1].axis('off')
            
            # Plot the resulting Morph
            axes[2].imshow(morph_img)
            axes[2].set_title(f"Morphed Result\n(Alpha: {alpha})")
            axes[2].axis('off')
            
            # Adjust manually to give room for titles
            plt.subplots_adjust(top=0.85, bottom=0.1, wspace=0.2)
            
            # General title to the whole figure
            fig.suptitle(f"Morphing Analysis - Pair {index:03d}\nt-SNE Dist: {row['tsne_dist']:.4f}", fontsize=12)
            
            # Save the full figure and close it to free memory
            plt.savefig(final_composite_path, dpi=150, bbox_inches='tight')
            plt.close(fig) 
            
            # Clean up the temporary standalone morph file
            if os.path.exists(temp_morph_path):
                os.remove(temp_morph_path)

            if (index + 1) % 10 == 0:
                print(f"Progress: [{index + 1}/{total_pairs}] composites created...")
        
        except Exception as e:
            print(f"Error processing pair {index}: {e}")

    print(f"\nSuccessfully generated composite plates in: {output_dir_path}")

def main():
    # Base path where your pair CSVs are stored
    DATA_BASE_PATH = "../data/Simple_ControlledMorphGeneration_MorphPairs/"

    # Execute controlled generation process for DENSE (Similar) pairs
    # This generates the "High Risk" samples
    process_controlled_generation(
        csv_path = DATA_BASE_PATH + "morph_pairs_DENSE_SIMILAR.csv", 
        output_dir_path = "./ControlledMorphGeneration/MorphGeneration_SimpleSimilarity/results/morphed_faces_dense/", 
        alpha = 0.5
    )

    process_controlled_generation(
        csv_path = DATA_BASE_PATH + "morph_pairs_EXTREME_DISTANT.csv", 
        output_dir_path = "./ControlledMorphGeneration/MorphGeneration_SimpleSimilarity/results/morphed_faces_extreme/", 
        alpha = 0.5
    )
    

if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[start] " + str(start) + "\033[0m" + "\n")
    
    main()
    
    end = datetime.datetime.now()
    print("\n" + "\033[0;34m" + "[end] "+ str(end) + "\033[0m" + "\n")

    exectime = end - start
    print(f"Total Execution Time: {exectime.total_seconds():.2f} seconds")