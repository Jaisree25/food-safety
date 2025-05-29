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
        # Let Keras build the internal layers with the given shape
        self.backbone.build(input_shape)
        super().build(input_shape)

def loss_fn(bboxes_true, box_preds):
    return tf.reduce_mean(tf.square(bboxes_true[:, 0] - box_preds))




script_dir = os.path.dirname(os.path.abspath(__file__))


paths = os.path.join(script_dir, 'mode1.keras')


model = detector()
dummy_input = tf.random.normal([1, 256, 256, 3])
model(dummy_input)
model.load_weights(paths)

for var in model.trainable_variables:
    print(var.name, var.shape)
model.summary()
#model = tf.keras.models.load_model("new_model1/")

def perform_detection(frame):
    #resize the image to 256 x 256 and convert to tensors.
    resized_frame = cv2.resize(frame, (256,256))
    tensor = tf.convert_to_tensor(np.array(resized_frame)/255.0, dtype=tf.float32)
    tensor = tf.expand_dims(tensor, axis = 0)

    #run that image through the model
    pred_bounds = model(tensor)

    #get the bounding boxes.
    pred_bounds = pred_bounds[0].numpy()

    height, width, _ = frame.shape

    #print("frame height", height)
    #print("frame width", width)
    y1, x1, y2, x2 = pred_bounds
    #print("top_before", y1)
    #print("left_before", x1)
    #print("bottom_before", y2)
    #print("right_before", x2)

    

    #rescale all of the bounding box dimensions
    (top, left, bottom, right) = (y1 * height), (x1 * width), (y2 * height), (x2 * width)
    #print("top", top)
    #print("left", left)
    #print("bottom", bottom)
    #print("right", right)

    return (top, left, bottom, right) 

for var in model.trainable_variables:
    print(var.name, var.shape)




def open_camera():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise Exception("Could not open capture")
    while True:
        ret, frame = video_capture.read()
        if not ret:
            raise Exception("Could not open frame")

        (top, left, bottom, right)  = perform_detection(frame)
        
        cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

        cv2.imshow("Glove Detection Test", frame)

        if cv2.waitKey(1) == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

open_camera()
"""
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))


paths = os.path.join(script_dir, 'Dataset/yolo_dataset/images/combined_test/DeleteThis.png')

#model = tf.keras.models.load_model("new_model.keras", custom_objects={'detector': detector})
test_image = paths
#test_labels = "labels/combined_test/sikharam-8-_ani_jpg.rf.18a5592f12576127a15db89ad7899e65.txt"
img = Image.open(test_image).convert("RGB")
img = img.resize((256,256))
input_tensor = tf.convert_to_tensor(np.array(img)/255.0, dtype = tf.float32)
input_tensor = tf.expand_dims(input_tensor, axis = 0)

pred_bounds = model(input_tensor)
pred_bounds = pred_bounds[0].numpy()

img_width, img_height = img.size
ymin, xmin, ymax, xmax = pred_bounds
print(pred_bounds)
print(img_width)
print(img_height)

(ymin, xmin, ymax, xmax) = (int(ymin * img_height), int(xmin * img_width), int(ymax * img_height), int(xmax * img_width))

#with open(test_labels, 'r') as label_file:
#  for line in label_file.readlines():
#    class_id, x_center, y_center, bound_width, bound_height = map(float, line.strip().split())

#ymin_actual = (y_center - bound_height / 2) * img_height
#ymax_actual = (y_center + bound_height / 2) * img_height
#xmin_actual = (x_center - bound_width / 2) * img_width
#xmax_actual = (x_center + bound_width / 2) * img_width


import matplotlib.patches as patches

fig, ax = plt.subplots(1)
ax.imshow(img)
rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth = 1, edgecolor = 'r', facecolor = 'none')
#rect2 = patches.Rectangle((xmin_actual, ymin_actual), xmax_actual - xmin_actual, ymax_actual - ymin_actual, linewidth = 1, edgecolor = 'b', facecolor = 'none')
#ax.add_patch(rect2)

ax.add_patch(rect)
plt.show()


"""