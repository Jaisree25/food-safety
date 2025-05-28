from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PREDICTED_FOLDER = 'static/predicted'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

IMAGE_SIZE = 256
GRID_SIZE = 4
BOXES_PER_CELL = 2
NUM_CLASSES = 1
OUTPUT = BOXES_PER_CELL * (5 + NUM_CLASSES)
CONF_THRESH = 0.6

model = tf.keras.models.load_model("detection5_model.h5", compile=False)

def draw_boxes(image, predictions):
    h, w = image.shape[:2]
    any_no_glove = False
    for gy in range(GRID_SIZE):
        for gx in range(GRID_SIZE):
            for b in range(BOXES_PER_CELL):
                base = b * (5 + NUM_CLASSES)
                conf = predictions[gy, gx, base + 4]
                if conf > CONF_THRESH:
                    x, y, bw, bh = predictions[gy, gx, base:base+4]
                    cx = int(x * w)
                    cy = int(y * h)
                    bw = int(bw * w / 2)
                    bh = int(bh * h / 2)
                    x1, y1 = cx - bw, cy - bh
                    x2, y2 = cx + bw, cy + bh
                    label = "Glove" if predictions[gy, gx, base+5] > 0.5 else "No Glove"
                    if label == "No Glove":
                        any_no_glove = True
                    color = (0, 255, 0) if label == "Glove" else (0, 0, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, any_no_glove

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            img = cv2.imread(upload_path)
            img_resized = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
            pred = model.predict(np.expand_dims(img_resized, axis=0))[0]

            img_drawn, any_no_glove = draw_boxes(img.copy(), pred)
            output_path = os.path.join(PREDICTED_FOLDER, filename)
            cv2.imwrite(output_path, img_drawn)

            bg_color = 'red' if any_no_glove else 'green'
            return render_template('result.html', image_url=url_for('static', filename=f'predicted/{filename}'), bg_color=bg_color)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)





