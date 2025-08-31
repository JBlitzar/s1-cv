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


class GenericColorPipeline(nn.Module):
    def __init__(self, channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.channels = channels

        # r*rp + g * gp + b * bp + bias added together
        # weight of size channels * thing + bias
        self.p = nn.Parameter(torch.randn(channels, 1, 1))

        self.b = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        x = x * self.p

        x = F.sigmoid(torch.sum(x, dim=1, keepdim=True) + self.b)

        x = x.repeat(1, 3, 1, 1)

        return x


import glob
import os

real_images = []
mask_images = []
for file in glob.glob("data/*.png"):
    if "-" in file:
        continue
    else:
        if os.path.exists(file.replace(".png", "-blue.png")):
            real_images.append(file)
            mask_images.append(file.replace(".png", "-blue.png"))

print(real_images)
print(mask_images)

from torchvision.transforms import v2

transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
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

        return image, mask


dset = MyDataset(transforms, real_images, mask_images)
loader = DataLoader(dset, batch_size=1, shuffle=True)
from tqdm import trange, tqdm
import cv2
import numpy as np

net = GenericColorPipeline(3).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
criterion = nn.MSELoss()
if os.path.exists("model.pt"):
    net.load_state_dict(torch.load("model.pt", map_location=device))
else:
    for epoch in trange(1_000):
        for image, mask in loader:
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


best_score, best_params = -1, None
with torch.no_grad():
    print("Running baseline (no morphology)...")
    baseline_score = 0
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        pred = net(image)
        pred = (pred > 0.5).float()
        loss = criterion(pred, mask)
        baseline_score += loss.item()
    baseline_score /= len(loader)
    print(f"Baseline score: {baseline_score:.4f}")
    for e in [1, 3, 5, 7, 9]:
        for d in [1, 3, 5, 7, 9]:
            for i in [1, 2, 3, 4, 5]:
                score_accum = 0
                for image, mask in loader:
                    image, mask = image.to(device), mask.to(device)
                    pred = net(image)

                    pred = (pred > 0.5).float()

                    pred_np = pred.squeeze(0).detach().cpu().numpy()
                    pred_np = (pred_np * 255).astype("uint8")
                    pred_np = pred_np.transpose(1, 2, 0)

                    morphed = apply_morphology(pred_np, e, d, i)

                    morphed_tensor = (
                        torch.from_numpy(morphed.transpose(2, 0, 1) / 255.0)
                        .unsqueeze(0)
                        .float()
                        .to(device)
                    )

                    loss = criterion(morphed_tensor, mask)
                    score_accum += loss.item()

                avg_score = score_accum / len(loader)

                if best_score == -1 or avg_score < best_score:
                    best_score, best_params = avg_score, (e, d, i)

print(f"Best: {best_params} -> {best_score:.4f}")


new_data = (
    transforms(Image.open("data/val-image-3.png").convert("RGB"))
    .unsqueeze(0)
    .to(device)
)
pred = net(new_data)
pred = (pred > 0.5).float()

pred_np = pred.squeeze(0).detach().cpu().numpy()
pred_np = (pred_np * 255).astype("uint8")
pred_np = pred_np.transpose(1, 2, 0)


if best_params:
    e, d, i = best_params
    pred_v2 = apply_morphology(pred_np, e, d, i)
else:
    pred_v2 = pred_np

Image.fromarray(pred_v2).save("prediction_v2.png")
