import os

import cv2
from torch.utils.data import DataLoader, Dataset


class ECUDataset(Dataset):
    def __init__(self, 
                 data_list_txt: str, 
                 image_prefix: str,
                 transform = None,
                 label_transform = None) -> None:
        self.image_prefix = image_prefix
        assert data_list_txt.endswith(".txt"), "data_list_txt should be a txt file"
        with open(data_list_txt, "r") as data_list_file:
            self.data_list = data_list_file.readlines()

        self.transform = transform
        self.label_transform = label_transform

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.data_list[idx].strip()
        # assert os.path.exists(image_path)
        label_path = self.get_label_path(image_path)

        image_path = os.path.join(self.image_prefix, image_path)
        label_path = os.path.join(self.image_prefix, label_path)
        assert os.path.exists(image_path), f"image_path, {image_path} not found"
        assert os.path.exists(label_path), f"label_path, {label_path} not found"
        image = cv2.imread(image_path)
        label = cv2.imread(label_path, -1)
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        # print(image.shape)
        return image, label
    
    def get_label_path(self, image_path: str) -> str:
        # images\001\im00010.jpg
        label_path = image_path.replace("images", "Skin").replace(".jpg", "_gt.png")
        return label_path 
