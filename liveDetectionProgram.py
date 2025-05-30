import os
import sys
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image


class detector(tf.keras.Model):
  def __init__(self):
    super(detector, self).__init__()
    self.backbone = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 2, strides = 1, activation = 'relu', padding= 'same'),
        #tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 2, strides = 1, activation = 'relu', padding= 'same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 2, strides = 1, activation = 'relu', padding= 'same'),
        #tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, 2, strides = 1, activation = 'relu', padding= 'same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.50),
        tf.keras.layers.Conv2D(512, 2, strides = 1, activation = 'relu', padding= 'same'),
        #tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(1024, 2, strides = 1, activation = 'relu', padding= 'same'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.GlobalAveragePooling2D()
    ])
    self.box_head = tf.keras.layers.Dense(4, activation = 'sigmoid')
  def call(self, x):
    return self.box_head(self.backbone(x))
  def build(self, input_shape):
        self.backbone.build(input_shape)
        super().build(input_shape)

def loss_fn(bboxes_true, box_preds):
    return tf.reduce_mean(tf.square(bboxes_true[:, 0] - box_preds))


def instantiate_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = os.path.join(script_dir, 'mode1.keras')

    model = detector()

    #reload the architecture of the model and the weights.
    dummy_input = tf.random.normal([1, 256, 256, 3])
    model(dummy_input)
    model.load_weights(paths)
    return model

def perform_detection(frame, detection_model):
    #resize the image to 256 x 256 and convert to tensors.
    resized_frame = cv2.resize(frame, (256,256))
    tensor = tf.convert_to_tensor(np.array(resized_frame)/255.0, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis = 0)

    #run that image through the model
    pred_bounds = detection_model(tensor)

    #get the bounding boxes.
    pred_bounds = pred_bounds[0].numpy()

    height, width, _ = frame.shape

    y1, x1, y2, x2 = pred_bounds

    #rescale all of the bounding box dimensions
    (top, left, bottom, right) = (y1 * height), (x1 * width), (y2 * height), (x2 * width)

    return (top, left, bottom, right) 


def open_camera():
    detection_model = instantiate_model()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open capture")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            raise Exception("Could not open frame")

        (top, left, bottom, right)  = perform_detection(frame, detection_model)
        
        #draw bounding boxes of our model predictions
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        cv2.imshow("Glove Detection Test", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

open_camera()
