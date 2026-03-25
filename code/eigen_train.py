from eigen import DepthDataset, EigenDepthModel, scale_invariant_loss
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

# model = EigenDepthModel().to(device)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Hyperparameter for scale-invariant loss (λ = 0.5 in paper)
LAMBDA = 0.5

# 1. Setup Data (Replace with your actual file paths)
rgb_files = ["./rgb/r_2011_09_26_0001_2_0000000016.png"] 
depth_files = ["./depth/gt_r_2011_09_26_0001_2_0000000016.png"]

dataset = DepthDataset(rgb_files, depth_files)
# Use a batch size > 1 if you have enough images
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 2. Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EigenDepthModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Adam usually converges faster for MDE

# 3. Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for i, (images, depths) in enumerate(train_loader):
        images, depths = images.to(device), depths.to(device)
        
        optimizer.zero_grad()
        preds = model(images)
        
        # Ensure predictions and depths are the same spatial size
        if preds.shape != depths.shape:
            preds = F.interpolate(preds, size=(depths.size(2), depths.size(3)), mode='bilinear')

        loss = scale_invariant_loss(preds, depths, lamda=0.5)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "eigen_model.pth")