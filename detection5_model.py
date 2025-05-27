import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 256
GRID_SIZE = 4
BOXES_PER_CELL = 2
NUM_CLASSES = 1
OUTPUT = BOXES_PER_CELL * (5 + NUM_CLASSES)

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
                grid[grid_y, grid_x, base+5] = cls
                break
    return grid

def load_data(folder, augment=False):
    x, y = [], []
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
        img_path = os.path.join(folder, fname)
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if not os.path.exists(txt_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
        with open(txt_path, 'r') as f:
            label_lines = f.readlines()
        label_grid = yolo_label_to_grid(label_lines)
        x.append(img)
        y.append(label_grid)

        if augment:
            # Horizontal Flip
            flipped_img = cv2.flip(img, 1)
            flipped_grid = np.zeros_like(label_grid)
            for gy in range(GRID_SIZE):
                for gx in range(GRID_SIZE):
                    for b in range(BOXES_PER_CELL):
                        base = b * (5 + NUM_CLASSES)
                        if label_grid[gy, gx, base + 4] == 1:
                            x, y_val, sqrt_w, sqrt_h = label_grid[gy, gx, base:base+4]
                            w, h = sqrt_w**2, sqrt_h**2
                            cls = label_grid[gy, gx, base + 5]
                            new_x = 1.0 - x
                            new_grid_x = int(new_x * GRID_SIZE)
                            new_grid_y = int(y_val * GRID_SIZE)
                            for b2 in range(BOXES_PER_CELL):
                                new_base = b2 * (5 + NUM_CLASSES)
                                if flipped_grid[new_grid_y, new_grid_x, new_base + 4] == 0:
                                    flipped_grid[new_grid_y, new_grid_x, new_base:new_base+5] = [new_x, y_val, np.sqrt(w), np.sqrt(h), 1]
                                    flipped_grid[new_grid_y, new_grid_x, new_base + 5] = cls
                                    break
            x.append(flipped_img)
            y.append(flipped_grid)

            # Random brightness/contrast shift
            alpha = np.random.uniform(0.8, 1.2)  # contrast
            beta = np.random.uniform(-0.1, 0.1)  # brightness
            bright_contrast_img = np.clip(img * alpha + beta, 0, 1)
            x.append(bright_contrast_img)
            y.append(label_grid)

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

def build_model():
    inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # (256, 256, 3)

    x = Conv2D(32, 3, padding='same')(inputs)          # (256, 256, 32)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (128, 128, 32)

    x = Conv2D(64, 3, padding='same')(x)               # (128, 128, 64)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (64, 64, 64)

    x = Conv2D(128, 3, padding='same')(x)              # (64, 64, 128)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (32, 32, 128)

    x = Conv2D(256, 3, padding='same')(x)              # (32, 32, 256)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (16, 16, 256)

    x = Conv2D(512, 3, padding='same')(x)              # (16, 16, 512)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (8, 8, 512)

    x = Conv2D(1024, 3, padding='same')(x)             # (8, 8, 1024)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D()(x)                              # (4, 4, 1024)

    x = Conv2D(OUTPUT, 1, activation='sigmoid')(x)     # (4, 4, 12)

    return Model(inputs, x)


def custom_yolo_loss(y_true, y_pred):
    lambda_box = 5.0
    lambda_obj = 1.0
    lambda_noobj = 0.5
    lambda_class = 1.0
    loss = 0.0

    def giou_loss(true_box, pred_box):
        x1_t = true_box[..., 0] - true_box[..., 2] / 2
        y1_t = true_box[..., 1] - true_box[..., 3] / 2
        x2_t = true_box[..., 0] + true_box[..., 2] / 2
        y2_t = true_box[..., 1] + true_box[..., 3] / 2

        x1_p = pred_box[..., 0] - pred_box[..., 2] / 2
        y1_p = pred_box[..., 1] - pred_box[..., 3] / 2
        x2_p = pred_box[..., 0] + pred_box[..., 2] / 2
        y2_p = pred_box[..., 1] + pred_box[..., 3] / 2

        xi1 = tf.maximum(x1_t, x1_p)
        yi1 = tf.maximum(y1_t, y1_p)
        xi2 = tf.minimum(x2_t, x2_p)
        yi2 = tf.minimum(y2_t, y2_p)
        inter = tf.maximum(xi2 - xi1, 0) * tf.maximum(yi2 - yi1, 0)

        area_t = (x2_t - x1_t) * (y2_t - y1_t)
        area_p = (x2_p - x1_p) * (y2_p - y1_p)
        union = area_t + area_p - inter #area_true + area_pred - intersection
        iou = inter / (union + 1e-6)

        xc1 = tf.minimum(x1_t, x1_p)
        yc1 = tf.minimum(y1_t, y1_p)
        xc2 = tf.maximum(x2_t, x2_p)
        yc2 = tf.maximum(y2_t, y2_p)
        area_c = (xc2 - xc1) * (yc2 - yc1)

        giou = iou - (area_c - union) / (area_c + 1e-6) #area_c - union = area between boxes

        return 1.0 - giou #Want giou to be 1 --> Perfect match

    for b in range(BOXES_PER_CELL):
        base = b * (5 + NUM_CLASSES)
        true_box = y_true[..., base:base+4]
        pred_box = y_pred[..., base:base+4]
        true_obj = y_true[..., base+4]
        pred_obj = y_pred[..., base+4]
        true_cls = y_true[..., base+5:base+6]
        pred_cls = y_pred[..., base+5:base+6]

        true_box = tf.concat([true_box[..., 0:2], tf.square(true_box[..., 2:4])], axis=-1) #[x, y, w, h]
        pred_box = tf.concat([pred_box[..., 0:2], tf.square(pred_box[..., 2:4])], axis=-1) # [x, y, w, h]

        giou = giou_loss(true_box, pred_box)
        box_loss = tf.reduce_sum(giou * true_obj)

        obj_bce = tf.keras.backend.binary_crossentropy(true_obj, pred_obj)
        obj_loss = tf.reduce_sum(obj_bce * true_obj)
        noobj_loss = tf.reduce_sum(obj_bce * (1.0 - true_obj))

        cls_bce = tf.keras.backend.binary_crossentropy(true_cls, pred_cls)
        cls_loss = tf.reduce_sum(cls_bce * tf.expand_dims(true_obj, axis=-1))

        loss += lambda_box * box_loss + lambda_obj * obj_loss + lambda_noobj * noobj_loss + lambda_class * cls_loss
    return loss

x_train, y_train = load_data("D:/food-safety/Dataset/train/combined_train", augment=True)
x_val, y_val = load_data("D:/food-safety/Dataset/valid/combined_valid")
x_test, y_test = load_data("D:/food-safety/Dataset/test/combined_test")

model = build_model()
model.compile(optimizer=Adam(1e-4), loss=custom_yolo_loss)

checkpoint = ModelCheckpoint("detection5_model.h5", monitor="val_loss", save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1, min_lr=1e-6)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=16, callbacks=[checkpoint, early_stop, reduce_lr])

test_loss = model.evaluate(x_test, y_test, batch_size=16)
print(f"Final Test Loss: {test_loss:.4f}")