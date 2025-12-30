!pip install torch torchvision transforms diffusers datasets matplotlib
from diffusers import StableDiffusionPipeline
import torch
#load Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cpu") #use "cuda" if you have a gpu
#Input a text prompt
prompt = "generate an image of elephant fighting with lion on a street"
image = pipe(prompt).images[0]
#save and display the generated image
image.save("generated_image.png")
image.show()
