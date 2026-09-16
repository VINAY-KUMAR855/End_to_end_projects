from ultralytics import YOLO
import cv2
from util import get_car, read_license_plate
from visualize import SaveOutput

results = {}

# load models
coco_model = YOLO("yolov8n.pt") # used for car detection
license_plate_detector = YOLO("./models/license_plate_detector.pt")

# load video
video_path = "./input_videos/sample_10sec_10fps.mp4"
cap = cv2.VideoCapture(video_path)

# read frames
ret = True
frame_nmr = -1
while ret:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr+=1
    results[frame_nmr] = {}
    # detect and track vehicles using bytetrack
    track_results = coco_model.track(
        frame,
        persist=True, 
        classes=[2, 3, 5, 7],
        tracker= "bytetrack.yaml",
        verbose=False
    )[0]
    track_ids = []
    if track_results.boxes.id is not None:
        boxes = track_results.boxes.xyxy.cpu().numpy()
        ids = track_results.boxes.id.cpu().numpy()

        for box,id in zip(boxes,ids):
            x1, y1, x2, y2 = box
            track_ids.append([x1, y1, x2, y2, int(id)])
    # detect licence plates
    license_plates = license_plate_detector(frame,verbose=False)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        # assign licence plate to car
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate,track_ids)

        if car_id !=-1:

            # crop the license plate
            license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2),:]

            #process licence plate
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            # detect license plate
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    "car": {'bbox':[xcar1, ycar1, xcar2, ycar2]},
                    "license_plate":{
                        "bbox":[x1, y1, x2, y2],
                        "text":license_plate_text,
                        "bbox_score":score,
                        "text_score":license_plate_text_score
                    }
                }



writer = SaveOutput(video_path,results)
writer.save_video()
cap.release()

