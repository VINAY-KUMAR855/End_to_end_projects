import matplotlib.pyplot as plt
import numpy as np
from data.tusimple_loader import get_train_loader


loader = get_train_loader(
    root_dir="/kaggle/input/datasets/manideep1108/tusimple/TUSimple/train_set"
)

images, targets = next(iter(loader)) # batch of images, targets

image = images[0]

lane_exist = targets["lane_exist"][0]
vertex_exist = targets["vertex_exist"][0]
lane_x = targets["lane_x"][0]
image = image.permute(1, 2, 0).numpy() # [3, H, W] -> [H, W, 3]

# plot image
plt.figure(figsize=(12, 6))
plt.imshow(image)
# Draw every lane
num_lanes = lane_x.shape[0]
num_rows = lane_x.shape[1]
colors = ["red","green","black","yellow","pink","orange"]
for lane_id in range(num_lanes):
    # check if Does this lane exist?
    if lane_exist[lane_id] == 0:
        continue
    for row in range(num_rows):
        # Is there a lane point at this row?
        if vertex_exist[lane_id, row] == 0:
            continue
        # x class -> image x coordinate
        x_class = lane_x[lane_id, row].item()
        x = (x_class / 256) * 512
        # model row -> image y coordinate
        y = (row / 128) * 256
        plt.scatter(x,y,s=5,c=colors[lane_id])

plt.title("TuSimple Lane Targets")

plt.xlim(0, 512)
plt.ylim(256, 0)

plt.savefig("tusimple_lane_check.png")
