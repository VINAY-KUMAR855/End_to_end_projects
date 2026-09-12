import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from models.model import E2ENet
from data.tusimple_loader import get_train_loader
import config
from losses import E2ELoss


# Device congig
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create checkpoint directory
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

# Data loader
train_loader = get_train_loader(root_dir= config.DATA_ROOT, batch=config.BATCH_SIZE)

# model
model = E2ENet(
    channels=96,
    nums_lane = config.NUM_LANES,
    culomn_channels = config.NUM_COLS,
    row_channels = config.NUM_ROWS
)
model = model.to(DEVICE)

# criterion
criterion = E2ELoss(
    lambda_vertx = config.LAMBDA_1,
    lambda_lane=config.LAMBDA_2
)

# optimizer 
optimizer = optim.AdamW(model.parameters(),lr = config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

# Training

for epoch in range(config.NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    running_location_loss = 0.0
    running_vertex_loss = 0.0
    running_lane_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]")

    for images, targets in progress_bar:

        images = images.to(DEVICE)
        lane_exist = targets["lane_exist"].to(DEVICE)
        vertex_exist = targets["vertex_exist"].to(DEVICE)
        lane_x = targets["lane_x"].to(DEVICE)

        # zero-grad
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        lane_exit_out = outputs["lane_exit_out"]
        vertex_outputs = [
            outputs["vertex_wise_confidence_out_1"],
            outputs["vertex_wise_confidence_out_2"],
            outputs["vertex_wise_confidence_out_3"],
            outputs["vertex_wise_confidence_out_4"],
            outputs["vertex_wise_confidence_out_5"],
            outputs["vertex_wise_confidence_out_6"],
        ]
        location_outputs = [
            outputs["row_wise_vertex_location_out_1"],
            outputs["row_wise_vertex_location_out_2"],
            outputs["row_wise_vertex_location_out_3"],
            outputs["row_wise_vertex_location_out_4"],
            outputs["row_wise_vertex_location_out_5"],
            outputs["row_wise_vertex_location_out_6"],
        ]
        # loss
        losses = criterion(
            lane_exit_out, vertex_outputs, location_outputs,
            lane_exist, vertex_exist, lane_x
        )
        total_loss = losses["total"]
        # backward
        total_loss.backward()
        # update step
        optimizer.step()

        # update running losses
        running_loss += total_loss
        running_location_loss += losses["location"]
        running_vertex_loss += losses["vertex"]
        running_lane_loss += losses["lane"]
        # update progress bar
        progress_bar.set_postfix({
            "loss": f"{total_loss.item():.4f}",
            "location":f"{losses["location"].item():.4f}",
            "vertex":f"{losses["vertex"]:.4f}",
            "lane":f"{losses["lane"]:.4f}"
        })
    # average
    num_batches = len(train_loader)
    epoch_loss = (running_loss/ num_batches)
    epoch_location_loss = (running_location_loss/num_batches)
    epoch_vertex_loss = (running_vertex_loss/num_batches)
    epoch_lane_loss = (running_lane_loss/ num_batches)
    # print epoch results
    print()
    print(
        f"Epoch [{epoch+1}/{config.NUM_EPOCHS}]     "
        f"Total loss: {epoch_loss:.4f}   "
        f"Location Loss: {epoch_location_loss:.4f}   "
        f"Vertex Loss: {epoch_vertex_loss:.4f}   "
        f"Lane loss: {epoch_lane_loss:.4f}  "
    )
    # save check points
    checkpoint_path = os.path.join(
        config.CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"
    )
    torch.save({
        "epoch":epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_loss
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")
    print()
