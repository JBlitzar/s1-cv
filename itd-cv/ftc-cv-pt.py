# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
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


class PixelWiseSinosodialRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.randn(4, 1, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

    def forward(self, x):
        # batch, (h, s, v), width, height
        # sin and cos of h, s, and v.
        h = x[:, 0:1, :, :]
        s = x[:, 1:2, :, :]
        v = x[:, 2:3, :, :]

        hsv_feat = torch.cat([torch.sin(h), torch.cos(h), s, v], dim=1)

        out = torch.sum(hsv_feat * self.w, dim=1, keepdim=True) + self.b
        out = torch.sigmoid(out)
        return out


class SinosodialOneLayerCnn(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.randn(4, 1, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x):
        # batch, (h, s, v), width, height
        # sin and cos of h, s, and v.
        h = x[:, 0:1, :, :]
        s = x[:, 1:2, :, :]
        v = x[:, 2:3, :, :]

        hsv_feat = torch.cat([torch.sin(h), torch.cos(h), s, v], dim=1)

        feat = self.conv(hsv_feat)

        out = torch.sum(feat * self.w, dim=1, keepdim=True) + self.b

        out = torch.sigmoid(out)
        return out


import glob
import os

real_images = []
mask_images = []
for file in glob.glob("data/*.png"):
    if "-" in file:
        continue
    else:
        n = int(file.split("/")[1].split(".png")[0])
        mask_file = file.replace(".png", f"-blue.png")

        if os.path.exists(mask_file):
            real_images.append(file)
            mask_images.append(mask_file)

from torchvision.transforms import v2


def rgbToHsvTransform(image):
    image = image.convert("HSV")
    return image


transforms = v2.Compose(
    [
        rgbToHsvTransform,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
)


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
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)

        mask = Image.open(self.masks[idx])
        if self.transform:
            mask = self.transform(mask)

        return image, mask


dset = MyDataset(transforms, real_images, mask_images)
loader = DataLoader(dset, batch_size=1, shuffle=True)
from tqdm import trange

net = SinosodialOneLayerCnn()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in trange(10):
    for image, mask in loader:
        pred = net(image)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

new_data = transforms(Image.open("data/val-image-3.png")).unsqueeze(0)
pred = net(new_data)
pred = pred.squeeze(0).squeeze(0).detach().numpy()
print(pred.shape)
pred = (pred * 255).astype("uint8")  # ugh average imageio
Image.fromarray(pred).save("prediction.png")
