import cv2
import numpy as np
import os

# Read image
img = cv2.imread("./input_images/handWritten1.jpg")
# resize image
if img.shape[1]>1000:
    width = 1000
    height = int(img.shape[0] * width / img.shape[1])
    img = cv2.resize(img,(width, height))
else:
    height, width = img.shape[:2]


# some helpful variables and functions for perspective transformation
def order_points(points):
    points = points.reshape(4,2)
    rect = np.zeros((4,2),dtype=np.float32)
    rect[0] = points[np.argmin(points.sum(axis=1))] # top-left
    rect[2] = points[np.argmax(points.sum(axis=1))] # bottom right
    rect[1] = points[np.argmin(np.diff(points, axis=1))] # top-right
    rect[3] = points[np.argmax(np.diff(points, axis=1))] # bottom-left
    
    return rect

destination = np.array([
    [0, 0], # top left
    [width - 1, 0], # top right
    [width - 1, height - 1], # bottom right
    [0, height - 1] # bottom left
], dtype=np.float32)
# output directory creation
output_dir = "./output"
os.makedirs(output_dir, exist_ok=True)

# keep original 
original = img.copy()
# conver to gray scale
grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply gaussian blur
blur = cv2.GaussianBlur(grey, (5,5),0)
# Otsu Threshold
_, threshold = cv2.threshold(blur, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find counters
contours, _ = cv2.findContours(
    threshold,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)
# Largest first
contours = sorted(contours, key = cv2.contourArea, reverse=True)
# Find document
document_contour = None
max_area = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area<1000:
        continue
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.15*perimeter, True)
    if len(approx)==4 and area >max_area:
        document_contour = approx
        max_area = area

# Check detection
if document_contour is None:
    print("Document not found")
else:
    print("Document detectd")
    # draw contour
    debug = original.copy()
    cv2.drawContours(debug,[document_contour],-1,(0,255,0),10)
    # Perspective transformation
    points = document_contour.reshape(4, 2)
    points = order_points(points)
    matrix = cv2.getPerspectiveTransform(points, destination)
    warped = cv2.warpPerspective(img, matrix,(width, height))
    # Scan effect
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    scanned = cv2.adaptiveThreshold(
        warped_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        12
    )
    
    cv2.imshow("Original", original)
    cv2.imshow("Threshold", threshold)
    cv2.imshow("Detected_Document",debug)
    cv2.imshow("warped", warped)
    cv2.imshow("Scanned", scanned)
    # cv2.imwrite("./output/Original.jpg", original)
    # cv2.imwrite("./output/Threshold.jpg", threshold)
    # cv2.imwrite("./output/Detected_Document.jpg",debug)
    # cv2.imwrite("./output/warped.jpg", warped)
    # cv2.imwrite("./output/Scanned.jpg", scanned)


cv2.waitKey(0)
cv2.destroyAllWindows()
