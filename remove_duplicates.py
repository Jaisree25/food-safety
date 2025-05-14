import os
import cv2
import hashlib

def get_image_hash(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64))
    return hashlib.md5(img.tobytes()).hexdigest()

def get_all_hashes(folder):
    hashes = set()
    for fname in os.listdir(folder):
        if fname.endswith(".jpg"):
            path = os.path.join(folder, fname)
            h = get_image_hash(path)
            if h:
                hashes.add(h)
    return hashes

def remove_duplicates(train_folder, test_folder):
    train_hashes = get_all_hashes(train_folder)

    deleted = 0
    for fname in os.listdir(test_folder):
        if fname.endswith(".jpg"):
            img_path = os.path.join(test_folder, fname)
            txt_path = os.path.splitext(img_path)[0] + ".txt"

            h = get_image_hash(img_path)
            if h in train_hashes:
                os.remove(img_path)
                if os.path.exists(txt_path):
                    os.remove(txt_path)
                print(f"Deleted duplicate: {fname}")
                deleted += 1

    print(f"Done. {deleted} duplicates removed from test folder.")

train_dir = "D:/food-safety/Dataset/train/combined_train"
test_dir = "D:/food-safety/Dataset/valid/combined_valid"

remove_duplicates(train_dir, test_dir)
