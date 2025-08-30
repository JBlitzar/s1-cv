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
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
from tqdm import trange

net = GenericColorPipeline(3).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
criterion = nn.MSELoss()

for epoch in trange(1_000):
    for image, mask in loader:
        image, mask = image.to(device), mask.to(device)
        pred = net(image)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

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
