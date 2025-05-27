import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

IMAGE_SIZE = 256
GRID_SIZE = 4
BOXES_PER_CELL = 2
NUM_CLASSES = 1
OUTPUT = BOXES_PER_CELL * (5 + NUM_CLASSES)
IOU_THRESHOLD = 0.5

def yolo_label_to_grid(label_lines):
    grid = np.zeros((GRID_SIZE, GRID_SIZE, OUTPUT))
    for line in label_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = map(float, parts)
        grid_x = int(x * GRID_SIZE)
        grid_y = int(y * GRID_SIZE)
        for b in range(BOXES_PER_CELL):
            base = b * (5 + NUM_CLASSES)
            if grid[grid_y, grid_x, base + 4] == 0:
                grid[grid_y, grid_x, base:base+5] = [x, y, np.sqrt(w), np.sqrt(h), 1]
                grid[grid_y, grid_x, base + 5] = cls
                break
    return grid

def load_test_data(folder):
    x, y = [], []
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(folder, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            continue
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        with open(txt_path, 'r') as f:
            label_lines = f.readlines()
        grid = yolo_label_to_grid(label_lines)
        x.append(img)
        y.append(grid)
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def iou(box1, box2):
    x1_1, y1_1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    x2_1, y2_1 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    x1_2, y1_2 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    x2_2, y2_2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / (union_area + 1e-6)

def evaluate_classification_and_iou(model_path, test_folder):
    model = load_model(model_path, compile=False)
    x_test, y_test = load_test_data(test_folder)
    preds = model.predict(x_test)

    total_boxes = 0
    correct_class = 0
    correct_iou = 0

    for pred, true in zip(preds, y_test):
        for gy in range(GRID_SIZE):
            for gx in range(GRID_SIZE):
                for b in range(BOXES_PER_CELL):
                    base = b * (5 + NUM_CLASSES)
                    if true[gy, gx, base + 4] == 1:
                        total_boxes += 1

                        # True values
                        tx, ty = true[gy, gx, base:base+2]
                        tw, th = np.square(true[gy, gx, base+2:base+4])
                        tcls = true[gy, gx, base + 5]
                        tbox = [tx, ty, tw, th]

                        # Predicted values
                        px, py = pred[gy, gx, base:base+2]
                        pw, ph = np.square(pred[gy, gx, base+2:base+4])
                        pcls = pred[gy, gx, base + 5]
                        pbox = [px, py, pw, ph]

                        # IoU accuracy
                        if iou(tbox, pbox) >= IOU_THRESHOLD:
                            correct_iou += 1

                        # Classification accuracy
                        if (pcls > 0.5) == (tcls > 0.5):
                            correct_class += 1

    print(f"Total ground-truth boxes: {total_boxes}")
    print(f"Classification Accuracy: {correct_class} / {total_boxes} = {correct_class / total_boxes:.2%}")
    print(f"IoU ≥ {IOU_THRESHOLD} Accuracy: {correct_iou} / {total_boxes} = {correct_iou / total_boxes:.2%}")

evaluate_classification_and_iou(model_path="detection5_model.h5", test_folder="D:/food-safety/Dataset/test/combined_test")

