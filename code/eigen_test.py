from eigen import EigenDepthModel
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EigenDepthModel().to(device)

# 2. Muat bobot dari file
model.load_state_dict(torch.load("eigen_model.pth", map_location=device))

# 3. Set ke mode evaluasi
model.eval()

import matplotlib.pyplot as plt
import numpy as np

def tensor_to_image(pred_tensor):
    # 1. Remove from GPU/Graph and take the first image in the batch
    depth_map = pred_tensor[0].detach().cpu().numpy()
    
    # 2. Squeeze to get (H, W)
    depth_map = depth_map.squeeze()
    
    # Optional: If you trained on log-depth, convert back to meters
    # depth_map = np.exp(depth_map) 
    
    return depth_map

def save_depth_prediction(pred_tensor, save_path="prediction.png"):
    depth_map = tensor_to_image(pred_tensor)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map, cmap='magma') # 'magma' or 'plasma' are standard for depth
    plt.colorbar(label='Distance (meters)')
    plt.title('Predicted Depth Map')
    plt.axis('off') # Hide pixel coordinates
    
    plt.savefig(save_path)
    plt.show()
    
def visualize_side_by_side(input_tensor, prediction_tensor):
    # 1. Konversi Input Tensor ke NumPy (HWC format untuk plotting)
    # Asumsi input_tensor shape: [1, 3, H, W]
    img_rgb = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # 2. Konversi Prediction Tensor ke NumPy (Grayscale/Heatmap)
    # Asumsi prediction_tensor shape: [1, 1, H, W]
    depth_map = prediction_tensor.squeeze().detach().cpu().numpy()
    
    # 3. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Gambar Asli
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')
    
    # Prediksi Depth
    # Gunakan cmap 'magma' atau 'plasma' agar kedalaman terlihat jelas
    im = axes[1].imshow(depth_map, cmap='magma')
    axes[1].set_title("Predicted Depth Map")
    axes[1].axis('off')
    
    # Tambahkan colorbar untuk referensi jarak
    fig.colorbar(im, ax=axes[1], label='Relative Depth')
    
    plt.tight_layout()
    plt.show()

def predict_depth(model, img, device):
    # 1. Load and Preprocess the image
    
    # Use the same transforms as your training (228x304 for Eigen 2014)
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
        # Normalize if you used it during training
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension [1, 3, 228, 304]

    # 2. Forward Pass
    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        prediction = model(input_tensor)
        
    # 3. Upscale back to original image size for better viewing
    # The Eigen model outputs a smaller map (55x74), we upscale it back to 228x304
    prediction = F.interpolate(prediction, size=(228, 304), mode='bilinear', align_corners=False)
    
    # 4. Convert to CPU and NumPy
    depth_map = prediction.squeeze().cpu().numpy()
    
    # 5. Handle Log-Depth (Optional)
    # If your model predicts log-depth, convert back:
    # depth_map = np.exp(depth_map)
    
    return depth_map, prediction

# 1. Load image
img = Image.open("./../dataset/KITTI/rgb/r_2011_09_26_0001_2_0000000016.png").convert('RGB')

# 2. Transform to Tensor (This creates the 'input_tensor')
transform = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0).to(device) # Shape: [1, 3, 228, 304]
depth_map, prediction = predict_depth(model=model, img=img, device=device)
save_depth_prediction(prediction)
visualize_side_by_side(input_tensor, prediction)