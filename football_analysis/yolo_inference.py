from ultralytics import YOLO
# import cv2

# model = YOLO("yolov8n.pt")
model = YOLO("./models/best.pt")

# cap = cv2.VideoCapture("./input_videos/output_5s_small.mp4")
# while True:
#     res, frame = cap.read()
#     if not res:
#         break

#     results = model.predict(frame, conf=0.1,imgsz=640,device="cpu")
#     annotated = results[0].plot()
#     cv2.imshow("img", annotated)
#     break;

# cap.release()
# cv2.waitKey(0)
# cv2.destroyAllWindows()

results = model.predict("./input_videos/output_5s_small.mp4")
print(results[0])
print("===============")

for box in results[0].boxes:
    print(box)


