import os
import shutil

def combine_subfolders(parent_dir, combined_name="combined_train"):
    combined_path = os.path.join(parent_dir, combined_name)
    os.makedirs(combined_path, exist_ok=True)

    for folder in os.listdir(parent_dir):
        folder_path = os.path.join(parent_dir, folder)
        if os.path.isdir(folder_path) and folder != combined_name:
            for filename in os.listdir(folder_path):
                src = os.path.join(folder_path, filename)
                dst = os.path.join(combined_path, filename)

                # If file already exists, rename to avoid overwrite
                if os.path.exists(dst):
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(os.path.join(combined_path, f"{base}_{counter}{ext}")):
                        counter += 1
                    dst = os.path.join(combined_path, f"{base}_{counter}{ext}")

                shutil.copy2(src, dst)
                print(f"Copied: {src} -> {dst}")

combine_subfolders("D:/food-safety/Dataset/train")
