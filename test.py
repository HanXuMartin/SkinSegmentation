import argparse
import importlib
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

IMG_FORMAT_LIST = [".jpg", '.png']
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Prediction Settingsc')
    parser.add_argument('--config', required=True, help='model config file (.py)')
    parser.add_argument('--ckp', required=True, help='checkpoint path')
    parser.add_argument('--data', help='should be a dir or an image',
                        default=r"E:\DeepLearning\Segmentation\ECU\images\002\im02010.jpg")
    args = parser.parse_args()
    return args

def import_module_by_path(module_path):
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

args = parse_args()
cfg = import_module_by_path(args.config)

def generate_test_list(image_path: str) -> None:
    data_set_path = r"./docs/test_list.txt"
    with open(data_set_path, "w") as data_list:
        if Path(image_path).suffix.lower() in IMG_FORMAT_LIST:
            data_list.write(f"{image_path}\n")
        else:
            for root, dirs, files in os.walk(image_path):
                for file in files:
                    image_path = os.path.join(root, file)
                    if not Path(image_path).suffix.lower() in IMG_FORMAT_LIST: continue
                    data_list.write(f"{image_path}\n")

def main():
    image_path = args.data
    checkpoint_path = args.ckp
    ckp_basename = os.path.basename(checkpoint_path).replace(".pth", "")
    generate_test_list(image_path)
    with open("docs/test_list.txt", "r") as txt_file:
        image_list = txt_file.readlines()
    image_num = len(image_list)
    assert image_num > 0, "No test image found"

    model = cfg.model.to(DEVICE)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()

    trans = cfg.test_transform

    pbar = tqdm(total=image_num)
    pbar.set_description('Inference')

    for from_image_path in image_list:
        from_image_path = from_image_path.strip()
        assert os.path.exists(from_image_path)
        image = cv2.imread(from_image_path)
        image_shape = image.shape
        image = trans(image)
        image = image.to(DEVICE)
        image = model(image)
        mask = image > 0.5
        mask = mask.squeeze(0).cpu().numpy().transpose(1,2,0).astype(int)*255

        mask_image = np.uint8(mask)
        mask_image = cv2.resize(mask_image, image_shape[:2][::-1])
        image_dirbasename = os.path.basename(os.path.dirname(from_image_path))
        image_basename = os.path.basename(from_image_path)
        output_path = os.path.join(f"./prediction/{ckp_basename}", image_dirbasename)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        cv2.imwrite(os.path.join(output_path, image_basename), mask_image)
        pbar.update()

if __name__ == "__main__":
    main()