!pip install ultralytics opencv-python-headless matplotlib
from google.colab import files
uploaded = files.upload()
from ultralytics import YOLO
# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model (fast & beginner-friendly)
import cv2
import matplotlib.pyplot as plt
# Get uploaded image name
image_path = list(uploaded.keys())[0]
# Read image using OpenCV
image = cv2.imread(image_path)
# Run YOLO detection
results = model(image)
# Draw bounding boxes
annotated_image = results[0].plot()
# Convert BGR to RGB for matplotlib
annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
# Display result
plt.figure(figsize=(10, 8))
plt.imshow(annotated_image)
plt.axis("off")
plt.show()
