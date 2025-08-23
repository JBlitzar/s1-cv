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


class PixelWiseHSVRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.w = nn.Parameter(torch.randn(4, 1, 1))
        self.b = nn.Parameter(torch.randn(1, 1))

    def rgb_to_hsv_cos_sin(self, x):
        R, G, B = x[:, 0], x[:, 1], x[:, 2]

        Cmax = torch.max(x, dim=1).values
        Cmin = torch.min(x, dim=1).values
        delta = Cmax - Cmin + 1e-6

        H = torch.zeros_like(R)
        mask = Cmax == R
        H[mask] = ((G - B) / delta)[mask] % 6
        mask = Cmax == G
        H[mask] = ((B - R) / delta + 2)[mask]
        mask = Cmax == B
        H[mask] = ((R - G) / delta + 4)[mask]
        H = H * (torch.pi / 3)

        S = delta / (Cmax + 1e-6)

        V = Cmax

        H_cos = torch.cos(H)
        H_sin = torch.sin(H)

        return torch.stack([H_cos, H_sin, S, V], dim=1)

    def forward(self, x):
        hsv_feat = self.rgb_to_hsv_cos_sin(x)
        out = torch.sum(hsv_feat * self.w, dim=1, keepdim=True) + self.b
        out = torch.sigmoid(out)
        return out


import glob

real_images = []
mask_images = []
for file in glob.glob("data/task-*.png"):
    if "annotation" in file:
        continue
    else:
        real_images.append(file)
        n = int(file.split("-")[1].split(".png")[0])
        mask_images.append(file.replace(".png", f"-annotation-{n}-by-1-tag-blue-0.png"))

from torchvision.transforms import v2

transforms = v2.Compose(
    [
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

net = PixelWiseHSVRegression()
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
