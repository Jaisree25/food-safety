import os
import shutil
import random

BASE = "D:/food-safety/Dataset"
SETS = {
    "train": "combined_train",
    "valid": "combined_valid",
    "test":  "combined_test"
}
NEW_SUFFIX = "balanced"

def get_class(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
        if not lines:
            return None
        try:
            return int(lines[0].split()[0])
        except:
            return None

def balance_split(set_name, folder_name):
    full_path = os.path.join(BASE, set_name, folder_name)
    new_path = os.path.join(BASE, set_name, f"{NEW_SUFFIX}_{set_name}")
    os.makedirs(new_path, exist_ok=True)

    files = [f for f in os.listdir(full_path) if f.endswith(".txt")]

    glove = []
    noglove = []

    for f in files:
        txt = os.path.join(full_path, f)
        cls = get_class(txt)
        if cls == 1:
            glove.append(f)
        elif cls == 0:
            noglove.append(f)

    count = min(len(glove), len(noglove))
    print(f"[{set_name}] Using {count} glove + {count} no-glove samples")

    selected = random.sample(glove, count) + random.sample(noglove, count)

    for fname in selected:
        jpg = os.path.splitext(fname)[0] + ".jpg"
        src_txt = os.path.join(full_path, fname)
        src_jpg = os.path.join(full_path, jpg)

        dst_txt = os.path.join(new_path, fname)
        dst_jpg = os.path.join(new_path, jpg)

        if os.path.exists(src_jpg):
            shutil.copy2(src_jpg, dst_jpg)
        shutil.copy2(src_txt, dst_txt)

for split, folder in SETS.items():
    balance_split(split, folder)