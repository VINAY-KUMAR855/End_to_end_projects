import torch
from torch.utils.data import DataLoader

from data.tusimple_dataset import TuSimpleDataset

def get_train_loader(root_dir, batch = 4):
    label_files = [
        "label_data_0313.json",
        "label_data_0531.json",
        "label_data_0601.json"
    ]
    dataset = TuSimpleDataset(root_dir, label_files)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True)
    return loader