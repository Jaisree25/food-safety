import os

# Set your dataset directory
directory = "D:/food-safety/Dataset"

# Walk through all subdirectories
for root, _, files in os.walk(directory):
    # Build sets of base filenames for .jpg and .txt
    jpg_files = {os.path.splitext(f)[0] for f in files if f.endswith(".jpg")}
    txt_files = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}

    # Files that don't have a match
    txts_to_remove = txt_files - jpg_files
    jpgs_to_remove = jpg_files - txt_files

    # Remove unmatched .txt files
    for base in txts_to_remove:
        txt_path = os.path.join(root, base + ".txt")
        os.remove(txt_path)
        print(f"Deleted .txt: {txt_path}")

    # Remove unmatched .jpg files
    for base in jpgs_to_remove:
        jpg_path = os.path.join(root, base + ".jpg")
        os.remove(jpg_path)
        print(f"Deleted .jpg: {jpg_path}")
