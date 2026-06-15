import pandas as pd
import os
import shutil

# --- CONFIGURATION ---
csv_file = 'labels.csv'       # Your CSV filename
source_folder = 'images'               # FOLDER NAME where your current images exist
destination_folder = 'renamed_dataset' # New folder for renamed images

# 1. Create the destination folder
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)
    print(f"Created folder: {destination_folder}")

# 2. Load the CSV data
df = pd.read_csv(csv_file)
print(f"Loaded CSV with {len(df)} labels.")

# 3. Loop through the CSV and copy/rename files
success_count = 0
missing_count = 0

for index, row in df.iterrows():
    original_filename = row['IMAGE']       # e.g., "90.png"
    medicine_name = row['MEDICINE_NAME']   # e.g., "Alatrol"
    
    # Safety check: Ensure the filename is a string
    if pd.isna(original_filename):
        continue

    # Sanitize the medicine name (Replace spaces with underscores, remove bad chars)
    # Example: "Napa Extend" -> "Napa_Extend"
    clean_med_name = str(medicine_name).replace(" ", "_").replace("/", "-").replace("\\", "-")
    
    # Create the new filename
    # Structure: MedicineName_OriginalID.png
    new_filename = f"{clean_med_name}_{original_filename}"
    
    # Get full file paths
    old_path = os.path.join(source_folder, original_filename)
    new_path = os.path.join(destination_folder, new_filename)
    
    # Copy and Rename
    if os.path.exists(old_path):
        shutil.copy(old_path, new_path)
        success_count += 1
        if success_count % 500 == 0:
            print(f"Processed {success_count} images...")
    else:
        # Note: This prints only if the image listed in CSV is missing from the folder
        # print(f"Missing: {original_filename}") 
        missing_count += 1

print("-" * 30)
print(f"Processing Complete.")
print(f"Successfully created: {success_count} images in '{destination_folder}/'")
if missing_count > 0:
    print(f"Warning: {missing_count} images listed in CSV were not found in '{source_folder}'.")