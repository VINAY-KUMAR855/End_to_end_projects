import os
import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TuSimpleDataset(Dataset):
    def __init__(self, root_dir, label_files, img_width = 512, img_height = 256, num_rows = 128, num_cols = 256, num_lanes = 6):
        self.root_dir = root_dir
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_lanes = num_lanes
        self.img_width = img_width
        self.img_height = img_height

        # load all label 
        self.samples = []
        for label_file in label_files:
            label_path = os.path.join(root_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))

        print(f"Total TuSimple samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # Image
        img_path = os.path.join(self.root_dir, sample["raw_file"])
        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size
        img = img.resize(
            (self.img_width, self.img_height),
            Image.BILINEAR
        )
        img = np.asarray(img)
        img = img.astype(np.float32)/ 255.0 # scale
        img = torch.from_numpy(img).permute(2,0,1)
        # create targets
        lane_exist = np.zeros(self.num_lanes, dtype=np.float32)
        vertex_exist = np.zeros((self.num_lanes, self.num_rows),dtype = np.float32)
        lane_x = np.zeros((self.num_lanes, self.num_rows),dtype = np.int64)

        # TuSimple annontations
        h_samples = np.array(sample["h_samples"], dtype=np.float32)
        lanes = sample["lanes"]
        # convert y coordinates
        h_resized = h_samples * (self.img_height/original_height)

        # add values to the created targets
        for lane_id in range(min(len(lanes), self.num_lanes)):
            lane = np.array(lanes[lane_id], dtype= np.int32)
            # Find TuSImple points
            valid = lane>=0
            if valid.sum()<2:
                continue
            valid_y = h_resized[valid]
            valid_x = lane[valid]
            valid_x = valid_x*(self.img_width / original_width)
            # create our 128 model rows and maps those 128 rows to the 256-pixel image.
            row_y = np.arange(self.num_rows)
            row_y = row_y*(self.img_height/self.num_rows)
            # interpolate
            valid_rows = ((row_y>=valid_y.min()) & (row_y<=valid_y.max()))
            interpolated = np.interp(row_y[valid_rows],valid_y, valid_x)
            # Convert image x coordinate to 256 classification bins. our model produses 256 classes for each row value.
            x_class = (interpolated/self.img_width*self.num_cols)
            x_class = np.round(x_class).astype(np.int64)
            x_class = np.clip(x_class, 0, self.num_cols-1)
            # store targets
            rows = np.where(valid_rows)[0]
            vertex_exist[lane_id, rows] = 1
            lane_x[lane_id, rows] = x_class
            lane_exist[lane_id] = 1
        # final targets
        targets = {
            "lane_exist": torch.from_numpy(lane_exist),
            "vertex_exist": torch.from_numpy(vertex_exist),
            "lane_x": torch.from_numpy(lane_x)
        }

        return img, targets

            