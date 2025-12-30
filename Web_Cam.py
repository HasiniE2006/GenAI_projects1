!pip install opencv-python opencv-python-headless numpy matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
!curl -O https://pjreddie.com/media/files/yolov3.weights
!curl -O https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
!curl -O https://github.com/pjreddie/darknet/blob/master/data/coco.names
import cv2
import numpy as np
import requests
# URL to download yolov3.cfg
cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
response = requests.get(cfg_url)
with open("yolov3.cfg", "wb") as file:
    file.write(response.content)
print("yolov3.cfg downloaded!")
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
response = requests.get(weights_url, stream=True)
with open("yolov3.weights", "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        file.write(chunk)
print("yolov3.weights downloaded!")
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
response = requests.get(names_url)
with open("coco.names", "wb") as file:
    file.write(response.content)
print("coco.names downloaded!")
import cv2
# Paths to files
config_path = "yolov3.cfg"
weights_path = "yolov3.weights"
classes_path = "coco.names"
# Load class names
with open(classes_path, "r") as file:
    class_names = file.read().strip().split("\n")
# Load YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("YOLO model loaded successfully!")
# Start the webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Prepare the frame for YOLO
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get YOLO output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    detections = net.forward(output_layers)

    # Initialize bounding boxes, confidences, and class IDs
    boxes, confidences, class_ids = [], [], []
# Process detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter weak detections
            if confidence > 0.5:
                box = detection[0:4] * np.array([width, height, width, height])
                center_x, center_y, w, h = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maxima Suppression to filter overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# Draw bounding boxes
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame
    cv2.imshow("YOLO Object Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

