!pip install torch torchvision matplotlib
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
#load the pretrained deeplabv3 model
import torchvision.models.segmentation as models
model =models.deeplabv3_resnet50(pretrained=True)
model.eval()#set the model to evaluation mode
image_path = r"C:\Users\admin\Pictures\cars.jpg"
input_image=Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485 , 0.456 , 0.406],std=[0.229, 0.224, 0.225])
])
input_tensor=preprocess(input_image).unsqueeze(0)
#performs interference
with torch.no_grad():
    output = model(input_tensor)["out"][0]
output_predictions = output.argmax(0)
colors=np.random.randint(0,255, size=(21, 3),dtype=np.uint8)
segmentation_map = colors[output_predictions.cpu().numpy()]
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(input_image)
plt.subplot(1, 2, 2)
plt.title("segmentation")
plt.imshow(segmentation_map)
plt.show()
