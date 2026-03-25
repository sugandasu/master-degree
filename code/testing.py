from eigen import EigenDepthModel, predict_depth, visualize_side_by_side
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EigenDepthModel().to(device)
model.load_state_dict(torch.load("eigen_model.pth", map_location=device))

model.eval()

dataset_path = "./../dataset/KITTI/"
df = pd.read_csv("dataset_files.csv")
print(df.head())

img = Image.open(dataset_path + df["rgb"].iloc[0]).convert('RGB')

image_width, image_height = img.size
print(f"Image size: {image_width} x {image_height}")  # Should print (width, height) e.g., (640, 192)

transform = transforms.Compose([
    transforms.Resize((image_height, image_width)),  # Resize to (192, 640)
    transforms.ToTensor(),
])

input_tensor = transform(img).unsqueeze(0).to(device)
depth_map, prediction = predict_depth(model=model, img=img, device=device)
visualize_side_by_side(input_tensor, prediction)