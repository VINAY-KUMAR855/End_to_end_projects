import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        res, frame = cap.read()
        if not res:
            break
        frames.append(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps

def save_video(output_video_frames,output_video_path,fps):
    if not output_video_frames:
        return
    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()