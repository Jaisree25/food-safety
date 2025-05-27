import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10

def yolo_to_box(bbox, img_width, img_height):
    x_center, y_center, w, h = bbox
    x_center *= img_width
    y_center *= img_height
    w *= img_width
    h *= img_height
    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)
    return max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)

def load_cropped_data(folder):
    X = []
    y = []
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(folder, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        img = cv2.imread(img_path)
        if img is None or not os.path.exists(txt_path):
            continue
        height, width = img.shape[:2]
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(parts[0])
                bbox = list(map(float, parts[1:]))
                x1, y1, x2, y2 = yolo_to_box(bbox, width, height)
                cropped = img[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue
                cropped = cv2.resize(cropped, (IMAGE_SIZE, IMAGE_SIZE))
                cropped = cropped / 255.0
                X.append(cropped)
                y.append(class_id)
    return np.array(X), np.array(y)

# Load data
X_train, y_train = load_cropped_data("D:/food-safety/Dataset/train/combined_train")
X_val, y_val     = load_cropped_data("D:/food-safety/Dataset/valid/combined_valid")
X_test, y_test   = load_cropped_data("D:/food-safety/Dataset/test/combined_test")

#print("Train Before Aug:", Counter(y_train))
#print("Valid:", Counter(y_val))
#print("Test: ", Counter(y_test))

# Split by class
X_glove = X_train[y_train == 1]
y_glove = y_train[y_train == 1]

X_nogl = X_train[y_train == 0]
y_nogl = y_train[y_train == 0]

# Augment only no-glove images
augmenter = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

aug_images = []
aug_labels = []

aug_iter = augmenter.flow(X_nogl, y_nogl, batch_size=1)
for _ in range(len(X_nogl) * 2):  # augment each no-glove sample twice
    img, label = next(aug_iter)
    aug_images.append(img[0])
    aug_labels.append(label[0])

# Combine and shuffle
X_train_final = np.concatenate([X_glove, X_nogl, np.array(aug_images)])
y_train_final = np.concatenate([y_glove, y_nogl, np.array(aug_labels)])
X_train_final, y_train_final = shuffle(X_train_final, y_train_final, random_state=42)

#print("Train After Aug:", Counter(y_train_final))

# Model
model = Sequential([
    Conv2D(32, (3,3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train_final, y_train_final,
          validation_data=(X_val, y_val),
          batch_size=BATCH_SIZE,
          epochs=EPOCHS)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}") #98%

# Confusion matrix
'''
y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Glove", "Glove"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
'''

#model.save("classification_model.h5")