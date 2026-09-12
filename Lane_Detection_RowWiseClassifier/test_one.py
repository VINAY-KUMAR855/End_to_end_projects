import torch
from PIL import Image
import numpy as np
import cv2
from models.model import E2ENet

DEVICE = torch.device("gpu" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "checkpoints/model.pth"
IMG_WIDTH = 512
IMG_HEIGHT = 256

NUM_ROWS = 128
NUM_COLS = 256
NUM_LANES = 6

# load model
model = E2ENet(
    channels=96,
    nums_lane=NUM_LANES,
    culomn_channels=NUM_COLS,
    row_channels=NUM_ROWS,
    initialed=False
)

checkpoint = torch.load(CHECKPOINT_PATH,map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# load image
img = Image.open("Test_img.jpg").convert("RGB")
original_width, original_height = img.size
image_resized = img.resize((IMG_WIDTH, IMG_HEIGHT),Image.BILINEAR)
img_np = np.asarray(image_resized).astype(np.float32)/ 255.0
image_tensor = torch.from_numpy(img_np).permute(2,0,1)
image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to(DEVICE)


with torch.no_grad():
    outputs = model(image_tensor)

lane_confidence = outputs["lane_exit_out"]
lane_confidence = torch.sigmoid(lane_confidence)
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
## draw predictions
vis = cv2.cvtColor(np.asarray(image_resized),cv2.COLOR_RGB2BGR)
# 128 row positions
row_y = np.arange(NUM_ROWS)
# Convert model row -> image y
row_y = row_y * (IMG_HEIGHT / NUM_ROWS)
LANE_THRESHOLD = 0.5
VERTEX_THRESHOLD = 0.5
for lane_id in range(NUM_LANES):
    if lane_confidence[0, lane_id].item()<LANE_THRESHOLD:
        continue
    vertex_score = torch.sigmoid(
        vertex_outputs[lane_id][0]
    )
    location_logits = location_outputs[lane_id][0]
    predicted_x_class = torch.argmax(
        location_logits,
        dim=0
    )
    predicted_x_class = predicted_x_class.cpu().numpy()
    vertex_score = vertex_score.cpu().numpy()
    # convert class to image x coordinates
    predicted_x = (predicted_x_class *IMG_WIDTH /NUM_COLS)
    # draw points
    points = []
    for row in range(NUM_ROWS):
        if vertex_score[row] < VERTEX_THRESHOLD:
            continue
        x = int(predicted_x[row])
        y = int(row_y[row])
        cv2.circle(
            vis,
            (x, y),
            2,
            (0, 255, 0),
            -1
        )
        points.append((x, y))
    for p1, p2 in zip(points[:-1], points[1:]):
        cv2.line(vis, p1, p2, (0,255,0), 2)

cv2.imwrite("Test_img_out.jpg", vis)


