#count persons
!pip install opencv-python
!wget https://pjreddie.com/media/files/yolov3.weights
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
!wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
from google.colab import drive
from google.colab import files
upload=files.upload()
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load YOLO
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
img = cv2.imread("im.jpg")
height, width, _ = img.shape

# Create blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)

boxes = []
confidences = []

# Detect persons
for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5 and class_id == 0:  # person
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))

# Apply Non-Max Suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

person_count = 0

# Draw boxes & count
if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        person_count += 1

# Display count on image
cv2.putText(
    img,
    f"Person Count: {person_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 0, 0),
    2
)

cv2_imshow(img)
