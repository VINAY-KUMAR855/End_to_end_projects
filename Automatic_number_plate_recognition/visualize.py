import cv2


class SaveOutput:
    def __init__(self, video_path, results):
        self.cap = cv2.VideoCapture(video_path)
        self.results = results
        # output video requirements
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter('./output_videos/out.mp4', fourcc, fps, (width, height))

    def draw_border(self, img, top_left, bottom_right,color=(0, 255, 0), thickness=10, line_length_x=100, line_length_y=100):
        x1, y1 = top_left
        x2, y2 = bottom_right

        cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)  # top-left
        cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

        cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)  # bottom-left
        cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

        cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)  # top-right
        cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

        cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)  # bottom-right
        cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

        return img

    def draw_license_plate(self, img, text, plate_bbox):
        x1, y1, x2, y2 = map(int, plate_bbox)

        # Small padding around the detected license plate
        padding_x = 3
        padding_y = 3

        x1 -= padding_x
        y1 -= padding_y
        x2 += padding_x
        y2 += padding_y

        # Draw a thin red rectangle around the actual plate
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_thickness = 3

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            text_thickness
        )

        # Put text above the plate, aligned with its left edge
        text_x = x1
        text_y = y1 - 8

        cv2.rectangle(
            img,
            (text_x - 2, text_y - text_height - baseline - 2),
            (text_x + text_width + 2, text_y + 2),
            (255, 255, 255),
            -1
        )

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA
        )

        return img
    
    def save_video(self):
        frame_nmr = -1
        while True:
            res, frame = self.cap.read()
            if not res:
                break
            frame_nmr +=1
            frame_res = self.results.get(frame_nmr, {})
            for car_id, car_data in frame_res.items():
                # draw car
                car_x1, car_y1, car_x2, car_y2 = car_data["car"]["bbox"]
                frame = self.draw_border(frame,(int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)))
                # draw license plate
                frame = self.draw_license_plate(
                    frame,
                    car_data["license_plate"]["text"],
                    car_data["license_plate"]["bbox"]
                )

            self.out.write(frame)

        self.out.release()
        self.cap.release()
