import os

from PIL import Image
from torch.utils.data import Dataset


class MyDrillingDataset(Dataset):
    def __init__(self, folder_path, gray=False, transform=None):
        self.folder_path = folder_path
        self.file_names = os.listdir(folder_path)
        self.gray = gray
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.file_names[idx])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if not already
        if self.gray:
            image = image.convert("L")
        if self.transform:
            image = self.transform(image)

        return image
