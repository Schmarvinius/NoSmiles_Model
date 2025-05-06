import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
FER_CSV = "fer2013.csv"  # Path to your original FER-2013 CSV
OUTPUT_IMAGE_DIR = "images"  # Directory to save images
OUTPUT_CSV = "happy_not_happy.csv"  # Output CSV for training script
SPLITS_TO_USE = ["Training", "PublicTest"]  # You can add "PrivateTest" if desired

# --- LOAD AND FILTER DATA ---
df = pd.read_csv(FER_CSV)
print(f"Original dataset size: {len(df)}")

# Filter by usage (optional)
df = df[df["usage"].isin(SPLITS_TO_USE)]
print(f"Filtered dataset size: {len(df)}")

# Map emotion to binary: 1 if happy (3), else 0
df["label"] = (df["emotion"] == 3).astype(int)

# --- CONVERT PIXELS TO IMAGES AND SAVE ---
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
rows = []

print("Converting pixel data to images and saving...")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(pixels)
    img = img.convert("RGB")  # EfficientNet expects 3 channels
    img_path = os.path.join(OUTPUT_IMAGE_DIR, f"img_{idx}.jpg")
    img.save(img_path)
    rows.append({"image_path": img_path, "label": row["label"]})

# --- SAVE NEW CSV ---
new_df = pd.DataFrame(rows)
new_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(new_df)} images and CSV to '{OUTPUT_CSV}'.")

# --- OPTIONAL: PRINT CLASS DISTRIBUTION ---
print("Class distribution:")
print(new_df["label"].value_counts())
