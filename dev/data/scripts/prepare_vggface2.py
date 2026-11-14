
from pathlib import Path
import pandas as pd

# Dirs
RAW_DIR = Path("/mnt/TB1/morph_pipeline_data/VGG-Face2/data")
OUTPUT_CSV = Path("/mnt/TB1/morph_pipeline_data/VGG-Face2/identity_VGGFace2.csv")

# Extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

records = []

# Process train and test
for split in ["train", "test"]:

    split_dir = RAW_DIR / split

    if not split_dir.exists():
        print(f"[WARNING] Doesn't exist: {split_dir}")
        continue

    for identity_dir in sorted(split_dir.iterdir()):

        if not identity_dir.is_dir():
            continue

        identity_id = identity_dir.name

        for image_path in sorted(identity_dir.iterdir()):

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            records.append({
                "File": image_path.relative_to(RAW_DIR).as_posix(),
                "ID": identity_id
            })

# Create dataframe
df = pd.DataFrame(records)

# Delete duplicates
df = df.drop_duplicates(subset=["File"])

# Sort samples
df = df.sort_values(["ID", "File"]).reset_index(drop=True)

# Export
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_CSV, index=False)

# Data info
print("\n===== VGG-FACE2 DATASET =====")
print(f"Total images: {len(df):,}")
print(f"Total identities: {df['ID'].nunique():,}")
print(f"Train images: {df['File'].str.startswith('train/').sum():,}")
print(f"Test images: {df['File'].str.startswith('test/').sum():,}")

print("\nImages per identity:")
print(df.groupby("ID").size().describe())

print(f"\nCSV generated: {OUTPUT_CSV}")
print("\nFirst records:")
print(df.head())
