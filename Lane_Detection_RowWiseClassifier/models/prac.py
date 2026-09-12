# import cv2
# import numpy as np

# canvas = np.zeros((720,1280,3))
# cv2.line(canvas,pt1=(0,0),pt2=(511,511),color = (0,255,0),thickness=5)
# cv2.imshow("img", canvas)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# Python program to explain 
# cv2.polylines() method

import cv2
import numpy as np
image = np.zeros((720,1280,3))
window_name = 'Image'
pts = np.array([[25, 70], [25, 145],
                [75, 190], [150, 190],
                [200, 145], [200, 70], 
                [150, 25], [75, 25]],
               np.int32)
print(pts.shape)
pts = pts.reshape((-1, 1, 2))
print(pts.shape)
isClosed = False

# Green color in BGR
color = (0, 255, 0)

# Line thickness of 8 px
thickness = 8

# Using cv2.polylines() method
# Draw a Green polygon with 
# thickness of 1 px
image = cv2.polylines(image, [pts], 
                      isClosed, color, 
                      thickness)

lanes = [
        [-2, -2, -2, -2, 632, 625, 617, 609, 601, 594, 586, 578, 570, 563, 555, 547, 539, 532, 524, 516, 508, 501, 493, 485, 477, 469, 462, 454, 446, 438, 431, 423, 415, 407, 400, 392, 384, 376, 369, 361, 353, 345, 338, 330, 322, 314, 307, 299],
        [-2, -2, -2, -2, 719, 734, 748, 762, 777, 791, 805, 820, 834, 848, 863, 877, 891, 906, 920, 934, 949, 963, 978, 992, 1006, 1021, 1035, 1049, 1064, 1078, 1092, 1107, 1121, 1135, 1150, 1164, 1178, 1193, 1207, 1221, 1236, 1250, 1265, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, 532, 503, 474, 445, 416, 387, 358, 329, 300, 271, 241, 212, 183, 154, 125, 96, 67, 38, 9, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, 781, 822, 862, 903, 944, 984, 1025, 1066, 1107, 1147, 1188, 1229, 1269, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2, -2]
       ]
h_samples= [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
mask = np.zeros((720,1280))
for lane in lanes:
    points = np.array([[x, y] for x,y in zip(lane, h_samples) if x!=-2])
    cv2.polylines(mask, [points], False, (255, 255))
cv2.imshow("img", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

