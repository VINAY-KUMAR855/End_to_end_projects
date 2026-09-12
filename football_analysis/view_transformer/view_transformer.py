import numpy as np
import cv2

# perspective transformation 
# It converts a point from the camera image (pixel coordinates) into coordinates on a top-down representation of the football pitch.

class ViewTransformer:
    def __init__(self):
        # real court width and length
        court_width = 68
        court_length = 23.32 # (we want 4 parts = (each part = (half court = 105 / 2)/ 9 parts) * 4) # ((105/2)/9)*4 = 23.32

        # trapezoid vertices
        ## four points in the camera image. These define the football pitch area that we are interested in.
        self.pixel_vertices = np.array([[110, 1035], 
                               [265, 275], 
                               [910, 260], 
                               [1640, 915]])
        # scale_x = 640 / 1920
        # scale_y = 360 / 1080

        # self.pixel_vertices = np.array([
        #     [110 * scale_x, 1035 * scale_y],
        #     [265 * scale_x, 275 * scale_y],
        #     [910 * scale_x, 260 * scale_y],
        #     [1640 * scale_x, 915 * scale_y]
        # ], dtype=np.float32)
        
        # rectangle vertices
        self.target_vertices = np.array([
            [0,court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)
    def transform_point(self, point):
        p = (int(point[0]),int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False)>=0
        if not is_inside:
            return None

        reshaped_point = point.reshape(-1,1,2).astype(np.float32)
        tranform_point = cv2.perspectiveTransform(reshaped_point,self.perspective_transformer)
        return tranform_point.reshape(-1,2)

    def add_transformed_position_to_track(self,tracks):
        for object, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_info in track.items():
                    position = np.array(track_info["position_adjusted"])
                    position_transformed = self.transform_point(position)

                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]["position_transformed"] = position_transformed