import os
from collections import Counter

def load_labels_only(folder):
    labels = []
    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        txt_path = os.path.join(folder, fname)
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        label = int(parts[0])
                        labels.append(label)
                    except ValueError:
                        continue  # skip bad lines
    return labels

# Paths
train_labels = load_labels_only("D:/food-safety/Dataset/train/combined_train")
val_labels = load_labels_only("D:/food-safety/Dataset/valid/combined_valid")
test_labels = load_labels_only("D:/food-safety/Dataset/test/combined_test")

#train_labels = load_labels_only("D:/food-safety/Dataset/train/balanced_train")
#val_labels = load_labels_only("D:/food-safety/Dataset/valid/balanced_valid")
#test_labels = load_labels_only("D:/food-safety/Dataset/test/balanced_test")

# Results
print("Train:", Counter(train_labels))
print("Valid:", Counter(val_labels))
print("Test:", Counter(test_labels))
