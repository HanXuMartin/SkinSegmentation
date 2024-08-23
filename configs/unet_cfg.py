import torch
from torch import nn
from torch.utils.data import DataLoader

from segmentation import transform
from segmentation.datasets import ECUDataset
from segmentation.model import Unet

input_size = (256,256) # (width, height)
batch_size = 5
num_works = 8

max_epoch = 30
checkpoint_save_interval = 5
exp_ID = 10

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

pre_transform = transform.Compose([
    transform.Resize(input_size),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std),
    ])

label_transform = transform.Compose([
    transform.Resize(input_size),
    transform.ImgBinary(),
    transform.ToTensor(),
    ])

train_dataset = ECUDataset(
    data_list_txt = r"docs\train_list.txt",
    image_prefix = "",
    transform = pre_transform,
    label_transform = label_transform,
    )

train_dataloader = DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = num_works,
    pin_memory = True,
    drop_last = False,
    persistent_workers = True
    )

model = Unet()

criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0001)

# test_dataset = ECUDataset(
#     data_list_txt = r"docs\val_list.txt",
#     image_prefix = "",
#     transform = pre_transform,
#     label_transform = label_transform,
#     )

# test_dataloader = DataLoader(
#     dataset = test_dataset,
#     batch_size = 1,
#     shuffle = True,
#     num_workers = num_works,
#     pin_memory = True,
#     drop_last = False,
#     persistent_workers = True
#     )

test_transform = transform.Compose([
    transform.Resize(input_size),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std),
    transform.Batchlization(input_size),
    ])