import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class EigenDepthModel(nn.Module):
    def __init__(self):
        super(EigenDepthModel, self).__init__()
        
        # New Output Resolution (Input / 4)
        self.out_h = 48
        self.out_w = 160
        
        # --- SCALE 1: GLOBAL COARSE ---
        # Input: 3 x 192 x 640
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Pool to a fixed size to handle linear layer input
            nn.AdaptiveAvgPool2d((6, 8))
        )
        
        self.coarse_fc = nn.Sequential(
            nn.Linear(256 * 6 * 8, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Changed to 48 * 160 to match new aspect ratio
            nn.Linear(4096, self.out_h * self.out_w), 
        )

        # --- SCALE 2: LOCAL FINE ---
        self.fine_conv1 = nn.Sequential(
            nn.Conv2d(3, 63, kernel_size=9, stride=2, padding=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.fine_conv2 = nn.Sequential(
            nn.Conv2d(63 + 1, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=5, padding=2),
        )

    def forward(self, x):
        # x: [Batch, 3, 192, 640]
        
        # Scale 1
        c = self.coarse_conv(x)
        c = torch.flatten(c, 1)
        c = self.coarse_fc(c)
        c = c.view(-1, 1, self.out_h, self.out_w) 
        
        # Scale 2
        f = self.fine_conv1(x) # Results in [Batch, 63, 48, 160]
        
        # Match coarse map to fine feature map resolution
        c_resized = F.interpolate(c, size=(f.size(2), f.size(3)), mode='bilinear', align_corners=False)
        
        # Concatenate and refine
        combined = torch.cat((f, c_resized), dim=1)
        output = self.fine_conv2(combined)
        
        return output
    
def scale_invariant_loss(pred, target, lamda=0.5):
    # Mask to ignore zero (invalid) depth pixels
    mask = (target > 0.1).detach() # Avoid extreme zeros
    
    # Clamp predictions to avoid log(0) or log(negative)
    pred = torch.clamp(pred, min=1e-3)
    
    d = torch.log(pred[mask]) - torch.log(target[mask])
    
    term1 = torch.mean(d**2)
    term2 = torch.pow(torch.sum(d), 2) / (d.numel()**2)
    
    return term1 - lamda * term2

class DepthDataset(Dataset):
    def __init__(self, dataset_path, rgb_paths, depth_paths):
        self.dataset_path = dataset_path
        self.rgb_paths = rgb_paths
        self.depth_paths = depth_paths
        
        # Eigen 2014 specific resize
        self.img_transform = transforms.Compose([
            transforms.Resize((192, 640)),
            transforms.ToTensor(),
        ])
        
        # Depth maps must be resized with NEAREST to avoid interpolating distances
        self.depth_transform = transforms.Compose([
            transforms.Resize((190, 638), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        # Load RGB
        image = Image.open( self.dataset_path + self.rgb_paths[idx]).convert('RGB')
        image = self.img_transform(image)
        
        # Load Depth
        depth = Image.open(self.dataset_path + self.depth_paths[idx])
        # Convert 16-bit mm to meters (float32)
        depth_np = np.array(depth).astype(np.float32) / 1000.0
        depth_pil = Image.fromarray(depth_np)
        depth = self.depth_transform(depth_pil)
        
        return image, depth

    def __len__(self):
        return len(self.rgb_paths)
    
def tensor_to_image(pred_tensor):
    # 1. Remove from GPU/Graph and take the first image in the batch
    depth_map = pred_tensor[0].detach().cpu().numpy()
    
    # 2. Squeeze to get (H, W)
    depth_map = depth_map.squeeze()
    
    # Optional: If you trained on log-depth, convert back to meters
    # depth_map = np.exp(depth_map) 
    
    return depth_map

def predict_depth(model, img, device):
    preprocess = transforms.Compose([
        transforms.Resize((192, 640)),
        transforms.ToTensor(),
    ])
    
    input_tensor = preprocess(img).unsqueeze(0).to(device) # Add batch dimension [1, 3, 228, 304]

    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        prediction = model(input_tensor)
        
    # 3. Upscale back to original image size for better viewing
    # The Eigen model outputs a smaller map (55x74), we upscale it back to 228x304
    prediction = F.interpolate(prediction, size=(192, 640), mode='bilinear', align_corners=False)
    
    # 4. Convert to CPU and NumPy
    depth_map = prediction.squeeze().cpu().numpy()
    
    # 5. Handle Log-Depth (Optional)
    # If your model predicts log-depth, convert back:
    # depth_map = np.exp(depth_map)
    
    return depth_map, prediction
    
def visualize_side_by_side(input_tensor, prediction_tensor):
    img_rgb = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    depth_map = prediction_tensor.squeeze().detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Gambar Asli
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original RGB Image")
    axes[0].axis('off')

    # Gambar Depth Map
    im = axes[1].imshow(depth_map, cmap='magma')
    axes[1].set_title("Predicted Depth Map")
    axes[1].axis('off')
    
    # Tambahkan colorbar untuk referensi jarak
    fig.colorbar(im, ax=axes[1], label='Relative Depth')
    
    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.show()