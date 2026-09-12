from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
sys.path.append("../")
from utils import get_center_of_box, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracker(self, tracks):
        ''' This function adds the one position point to the objects like ball,etc.'''
        for object, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]
                    if object=="ball":
                        position = get_center_of_box(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,{}).get("bbox",[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])
        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 2 # 20
        detections = []
        for i in range(0,len(frames), batch_size):
            detection_batch = self.model.predict(frames[i:i+batch_size], conf=0.05,imgsz=640,device="cpu",verbose=False)
            detections.extend(detection_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path = None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,"rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {
            "players":[],
            "referees":[],
            "ball":[]
        }
        # tracks = {
        #     "players": [
        #         # Frame 0
        #         {1: { "bbox": [100, 100, 150, 300]},2: {"bbox": [400, 100, 450, 300]}},
        #         # Frame 1
        #         {1: {"bbox": [110, 105, 160, 305]},2: {"bbox": [390, 105, 440, 305]}}
        for frame_num, detection in enumerate(detections):

            cls_names = detection.names
            cls_names_inv = {k:v for v,k in cls_names.items()}

            # convert ultralytics format to supervision detect format
            detections_supervision = sv.Detections.from_ultralytics(detection)
              
            # convert GoalKeeper to player object.our model was unable to detect GoalKeeper due to small training data.
            # so, we after convert to player object, all plyers in the ground becomes players. so now the final classes are player, referee, ball
            for object_id, class_id in enumerate(detections_supervision.class_id):
                if cls_names[class_id]=="goalkeeper":
                    detections_supervision.class_id[object_id] = cls_names_inv["player"]

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detections_supervision)

            # append empty dictnorie
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            # below 2 loops tells go through every detected object in the current frame. 1st loop for players, referees. 2nd is for ball
            for frame_detection in detection_with_tracks:
                # frame_detection is (bbx=array([[],[],..]), mask=None, conf=[],class_id=[2,2,3..], track_id=[1,2,3,4,..])
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id]={"bbox":bbox} 
                    # tracks["players"][1st frame][2 track_id] = {"bbox": bbox} means In frame 5, player ID 17 is at this bounding box.
                if cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            for frame_detection in detections_supervision: # this is for ball. ball is no need for tracking
                bbox = frame_detection[0].tolist()
                cls_id =frame_detection[3]
                if cls_id == cls_names_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        # save tracks object in pickle file 
        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks,f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # (x1,y1,x2,y2) = (top left x, top left y, bottom right x, bottom right y)
        y2 = int(bbox[3])
        x_center,_ = get_center_of_box(bbox)
        width = get_bbox_width(bbox) # radius of the ellipse

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rext = x_center + rectangle_width//2
        y1_rect = (y2-rectangle_height//2)+15
        y2_rect = (y2+rectangle_height//2)+15

        if track_id is not None:
            cv2.rectangle(
                frame,
                (int(x1_rect),int(y1_rect)),
                (int(x2_rext),int(y2_rect)),
                color,
                cv2.FILLED
            )
            x1_text = x1_rect + 12
            if track_id>99:
                x1_text -=10
            cv2.putText(
                frame,
                str(track_id),
                (int(x1_rect),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])
        x,_ = get_center_of_box(bbox)
        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])

        cv2.drawContours(frame, [triangle_points],0,color,cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_ball_control(self, frame, frame_num, team_ball_control):
        # h, w = frame.shape[:2]
        # # Define box size and position relative to frame size
        # box_w = int(w * 0.25)
        # box_h = int(h * 0.15)
        # x1 = int(w * 0.62)
        # y1 = int(h * 0.80)
        # x2 = x1 + box_w
        # y2 = y1 + box_h

        # Draw a semi-transparent rectangle
        overlay = frame.copy()
        # cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.5
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Avoid division by zero
        total = team_ball_control_till_frame.shape[0]
        if total == 0:
            team_1 = 0.0
            team_2 = 0.0
        else:
            team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
            team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
            team_1 = team_1_num_frames / total
            team_2 = team_2_num_frames / total

        # Text positions inside the box
        # text_x = x1 + 10
        # text_y1 = y1 + int(box_h * 0.35)
        # text_y2 = y1 + int(box_h * 0.65)
        # cv2.putText(frame,f"Team 1: {team_1 * 100:.1f}%",(text_x, text_y1),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2)
        # cv2.putText(frame,f"Team 2: {team_2 * 100:.1f}%",(text_x, text_y2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 0),2)
        cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        return frame
    
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                frame = self.draw_ellipse(frame,player["bbox"],color, track_id)
                if player.get("has_ball",False):
                    frame = self.draw_traingle(frame,player["bbox"],(0,0,225))

            # Draw referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame,referee["bbox"],(0,255,255))

            # Draw ball
            for _,ball in ball_dict.items():
                frame = self.draw_traingle(frame,ball["bbox"],(0,255,0))    

            # Draw team ball control
            frame = self.draw_ball_control(frame,frame_num, team_ball_control)
            
            output_video_frames.append(frame)

        return output_video_frames


