# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "opencv-python",
#     "pillow",
#     "torch",
#     "torchvision",
#     "tqdm",
# ]
# ///
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

os.system(f"caffeinate -is -w {os.getpid()} &")


class PixelWiseColorRegression(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels

        # r*rp + g * gp + b * bp + bias added together
        # weight of size channels * thing + bias
        self.p = nn.Parameter(torch.randn(channels, 1, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        return F.sigmoid(torch.sum(x * self.p, dim=1, keepdim=True) + self.b)


class MaxwellTriangle(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x / (x.sum(dim=1, keepdim=True) + 1e-6)

        return x


class GenericColorPipeline(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels

        # rw*MSE(r,rp) + gw*MSE(g,gp) + bw*MSE(b,bp) + bias added together
        # weight of size channels * thing + bias
        self.p = nn.Parameter(torch.randn(channels, 1, 1))

        self.w = nn.Parameter(torch.randn(channels, 1, 1))

        self.b = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        x = x / (x.sum(dim=1, keepdim=True) + 1e-6)

        x = torch.sum(self.w * (x - self.p) ** 2, dim=1, keepdim=True)

        x = F.sigmoid(x + self.b)

        x = x.repeat(1, 3, 1, 1)

        return x


import glob
import os

real_images = []
mask_images = []
mask_top_images = []
for file in glob.glob("data/*.png"):
    if "-" in file:
        continue
    else:
        new = file.replace(".png", "-blue.png")
        new_top = file.replace(".png", "-blue-top.png")
        if os.path.exists(new) and os.path.exists(new_top):
            real_images.append(file)
            mask_images.append(new)
            mask_top_images.append(new_top)

print(real_images)
print(mask_images)
print(mask_top_images)

from torchvision.transforms import v2

transforms = v2.Compose(
    [
        # v2.Lambda(lambda img: img.convert("HSV")),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.ColorJitter(brightness=0.2),
        # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = "mps" if torch.backends.mps.is_available() else "cpu"


class MyDataset(Dataset):
    def __init__(self, transform, files, masks) -> None:
        self.transform = transform
        self.files = files
        self.masks = masks

        assert len(self.files) == len(self.masks)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")  # Remove alpha channel
        if self.transform:
            image = self.transform(image)

        mask = Image.open(self.masks[idx]).convert("RGB")  # Remove alpha channel
        if self.transform:
            mask = self.transform(mask)

        top_mask = Image.open(mask_top_images[idx]).convert("RGB")
        if self.transform:
            top_mask = self.transform(top_mask)

        return image, mask, top_mask


dset = MyDataset(transforms, real_images, mask_images)
loader = DataLoader(dset, batch_size=1, shuffle=True)
from tqdm import trange, tqdm
import cv2
import numpy as np

net = GenericColorPipeline(3).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=3e-3)
criterion = nn.MSELoss()
if os.path.exists("model.pt"):
    net.load_state_dict(torch.load("model.pt", map_location=device))
else:
    for epoch in trange(1_000):
        for image, mask, _ in loader:
            image, mask = image.to(device), mask.to(device)
            pred = net(image)
            loss = criterion(pred, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm.write(f"Epoch {epoch}, Loss: {loss.item()}")

new_data = (
    transforms(Image.open("data/val-image-3.png").convert("RGB"))
    .unsqueeze(0)
    .to(device)
)
pred = net(new_data)
pred = pred.squeeze(0).detach().cpu().numpy()
print(pred.shape)
print(new_data.shape)
pred = (pred * 255).astype("uint8")  # ugh average imageio
pred = pred.transpose(1, 2, 0)
Image.fromarray(pred).save("prediction.png")
print("Model parameters:")
for name, param in net.named_parameters():
    print(f"{name}: {param.data}")

torch.save(net.state_dict(), "model.pt")
net.eval()


def apply_morphology(mask, erode_size, dilate_size, iterations):
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size))
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_RECT, (dilate_size, dilate_size)
    )

    morphed = cv2.erode(mask, kernel_erode, iterations=iterations)
    morphed = cv2.dilate(morphed, kernel_dilate, iterations=iterations)
    return morphed


def apply_morphology_batch(masks, erode_size, dilate_size, iterations):
    """Apply morphology to a batch of masks"""
    kernel_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_size, erode_size)
    )
    kernel_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)
    )

    morphed_batch = []
    for mask in masks:
        morphed = cv2.erode(mask, kernel_erode, iterations=iterations)
        morphed = cv2.dilate(morphed, kernel_dilate, iterations=iterations)
        morphed_batch.append(morphed)
    return morphed_batch


# Pre-compute all predictions and targets
print("Pre-computing predictions...")
all_preds, all_targets = [], []
with torch.no_grad():
    for image, _, top_mask in loader:
        image, top_mask = image.to(device), top_mask.to(device)
        pred = net(image)
        all_preds.append(pred.cpu())  # Keep as individual tensors
        all_targets.append(top_mask.cpu())  # Keep as individual tensors

print(f"Pre-computed {len(all_preds)} predictions")
print("Running grid search...")
best_score, best_params = float("inf"), None

from itertools import product

param_combinations = list(
    product(
        [1, 3, 5, 7, 9],  # erode sizes
        [1, 3, 5, 7, 9],  # dilate sizes
        [1, 2, 3, 4, 5],  # iterations
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  # thresholds
    )
)

for e, d, i, thresh in tqdm(param_combinations, desc="Grid search"):
    score_accum = 0

    for pred, target in zip(all_preds, all_targets):
        # Move to device only when needed
        pred = pred.to(device)
        target = target.to(device)

        # Apply threshold
        pred_thresh = (pred > thresh).float()

        # Convert to numpy for morphology
        pred_np = pred_thresh.squeeze(0).detach().cpu().numpy()
        pred_np = (pred_np * 255).astype("uint8").transpose(1, 2, 0)

        # Apply morphology
        morphed = apply_morphology(pred_np, e, d, i)

        # Convert back to tensor
        morphed_tensor = (
            torch.from_numpy(morphed.transpose(2, 0, 1) / 255.0)
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # Compute loss
        loss = criterion(morphed_tensor, target)
        score_accum += loss.item()

    avg_score = score_accum / len(all_preds)

    if avg_score < best_score:
        best_score, best_params = avg_score, (e, d, i, thresh)
        tqdm.write(f"New best: {best_params} -> {best_score:.4f}")


print(f"Best: {best_params} -> {best_score:.4f}")


new_data = (
    transforms(Image.open("data/val-image-3.png").convert("RGB"))
    .unsqueeze(0)
    .to(device)
)
pred = net(new_data)
e, d, i, t = best_params
pred = (pred > t).float()

pred_np = pred.squeeze(0).detach().cpu().numpy()
pred_np = (pred_np * 255).astype("uint8")
pred_np = pred_np.transpose(1, 2, 0)


pred_v2 = apply_morphology(pred_np, e, d, i)


Image.fromarray(pred_v2).save("prediction_v2.png")

Image.fromarray(
    (
        MaxwellTriangle()(new_data).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    ).astype("uint8")
).save("maxwell.png")
