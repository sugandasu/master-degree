import pandas as pd
from eigen import DepthDataset, EigenDepthModel, scale_invariant_loss
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

dataset_path = "./../dataset/KITTI/"
df = pd.read_csv("dataset_files.csv")
print(df.head())

# Image shape: torch.Size([3, 192, 640]), Depth shape: torch.Size([1, 190, 638])
dataset = DepthDataset(dataset_path, df["rgb"].tolist(), df["dense"].tolist())

for i in range(5):
    img, depth = dataset[i]
    print(f"Image shape: {img.shape}, Depth shape: {depth.shape}")

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EigenDepthModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i, (images, depths) in enumerate(train_loader):
        images, depths = images.to(device), depths.to(device)

        optimizer.zero_grad()
        preds = model(images)

        # Ensure predictions and depths are the same spatial size
        if preds.shape != depths.shape:
            preds = F.interpolate(
                preds, size=(depths.size(2), depths.size(3)), mode="bilinear"
            )

        loss = scale_invariant_loss(preds, depths, lamda=0.5)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {epoch_loss/len(train_loader):.4f}"
    )

torch.save(model.state_dict(), "eigen_model.pth")
