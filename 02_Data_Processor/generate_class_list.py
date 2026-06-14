import os
import re

# --- Configuration ---
# This is the folder containing your cleaned, renamed image files.
# It assumes you ran the previous script which created this folder.
IMAGE_DIR = 'labeled_dataset_kaggle' 
# The file to save the final class list to (optional)
OUTPUT_FILE = 'class_names_list.txt'
# ---------------------

def generate_unique_class_list():
    """
    Reads image filenames from a directory, extracts the medicine name 
    based on the naming convention (MEDICINE_NAME_index.jpg), 
    and generates a list of unique names.
    """
    
    print(f"Scanning directory: '{IMAGE_DIR}' for unique medicine names...")

    # Check if the directory exists
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: Directory '{IMAGE_DIR}' not found.")
        print("Please ensure you have run the image renaming script and that the folder exists.")
        return

    unique_names = set()
    
    # Regex pattern to capture the class name (everything before the last underscore)
    # Assumes the format is "MEDICINE_NAME_index.jpg" or "MEDICINE_NAME_index_count.jpg"
    # Example: Aceta_0.jpg -> Aceta
    # Example: Zithrin_3084.jpg -> Zithrin
    name_pattern = re.compile(r'^([a-zA-Z0-9_-]+)_[0-9]+\.jpg$')

    # Iterate over all files in the directory
    for filename in os.listdir(IMAGE_DIR):
        # Only process JPG files
        if filename.lower().endswith('.jpg'):
            # Remove the .jpg extension for processing
            base_name = filename[:-4] 
            
            # Find the position of the last underscore, which separates the name from the index
            last_underscore_pos = base_name.rfind('_')
            
            if last_underscore_pos != -1:
                # Extract the medicine name (everything up to the last underscore)
                medicine_name = base_name[:last_underscore_pos]
                
                # Check for the secondary underscore added for duplicates (e.g., Aceta_0_0)
                # If found, cut off the trailing _0, _1, etc.
                if medicine_name.count('_') > 0 and medicine_name[-2:].startswith('_'):
                    medicine_name = medicine_name[:medicine_name.rfind('_')]

                # Clean up and add to the set
                if medicine_name:
                    unique_names.add(medicine_name.strip())
            else:
                print(f"Skipping file: '{filename}' - Does not match expected format (e.g., Name_index.jpg)")

    # Convert the set to a sorted list
    class_list = sorted(list(unique_names))

    # Format the list as the requested Python string format
    formatted_output = "(" + ", ".join(f'"{name}"' for name in class_list) + ")"

    print("\n" + "="*50)
    print(f"✅ Found {len(class_list)} Unique Medicine Names (Classes):")
    print("="*50)
    print(formatted_output)
    print("="*50)

    # Optional: Save the list to a file
    try:
        with open(OUTPUT_FILE, 'w') as f:
            f.write(formatted_output)
        print(f"\nSaved the class list to: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Warning: Could not save to file. Error: {e}")


if __name__ == "__main__":
    generate_unique_class_list()