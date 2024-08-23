import argparse
import importlib
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Prediction Settings')
    parser.add_argument('--config', required=True, help='should be a .py file')
    args = parser.parse_args()
    return args

def import_module_by_path(module_path):
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

args = parse_args()
cfg = import_module_by_path(args.config)

MAX_EPOCH = cfg.max_epoch
EXP_ID = cfg.exp_ID
MODEL_NAME = cfg.model.__class__.__name__
CHECKPOINT_SAVE_INTERVAL = cfg.checkpoint_save_interval
CHECKPOINT_SAVE_PATH = f"./workdir/{MODEL_NAME}-{EXP_ID:02d}"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def format_time(total_sec: int) -> str:
    """change seconds into: hours:min:sec"""
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def main():
    if not os.path.exists(CHECKPOINT_SAVE_PATH):
        os.makedirs(CHECKPOINT_SAVE_PATH)
    dataloader = cfg.train_dataloader
    iter_num = len(dataloader)
    model = cfg.model.to(DEVICE)
    model.train()
    optimizer = cfg.optimizer
    criterion = cfg.criterion
    # return 0
    total_loss = 0
    for epoch in range(MAX_EPOCH):        
        for batch_idx, (images, labels) in enumerate(dataloader):
            start = time.time()
            # total_loss = 0.
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # print(images.device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print(loss)
            total_loss += loss.item()
            end = time.time()
            total_time = (end-start) * ((MAX_EPOCH-epoch) * iter_num - batch_idx - 1)
            eta_time = format_time(int(total_time))
            # print(f"Epoch{epoch}")
            print(f"Epoch {epoch+1}|{MAX_EPOCH}\t"
                  f"Batch {batch_idx+1}|{iter_num}\t"
                  f"Loss: {total_loss/(batch_idx+1+epoch*iter_num): .04f}\t"
                  f"ETA: {eta_time}")

        if epoch % CHECKPOINT_SAVE_INTERVAL == 0:
            model_name = f"{MODEL_NAME}-epoch_{epoch}.pth"
            save_path = os.path.join(CHECKPOINT_SAVE_PATH, model_name)
            torch.save(model.state_dict(), save_path)
            
if __name__ == "__main__":
    main()
