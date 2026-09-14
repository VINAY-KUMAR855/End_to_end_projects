# Football Analyzer ⚽

A simple Computer Vision project to analyze football players from a
video.

## What This Project Does

-   Detects players, referees, goalkeepers, and the ball using a custom
    YOLOv5 model.
-   Tracks players using ByteTrack.
-   Estimates camera movement using optical flow.
-   Uses perspective transformation to convert image positions into
    pitch positions.
-   Calculates player speed and distance covered.
-   Assigns players to teams using jersey color and K-Means clustering.
-   Finds which player has the ball.
-   Calculates ball control percentage for each team.
-   Draws the results on the video.

> **Note:** Speed and distance are calculated only for a specific
> portion of the pitch. Therefore, some players may not have their speed
> displayed in the output video.

## Project Flow

``` text
Input Video
     ↓
YOLO Detection
     ↓
Player Tracking
     ↓
Camera Movement Estimation
     ↓
Perspective Transformation
     ↓
Speed & Distance
     ↓
Team Assignment
     ↓
Ball Possession
     ↓
Output Video
```

## Tech Stack

-   Python
-   YOLOv5
-   OpenCV
-   Ultralytics
-   Supervision
-   NumPy
-   Pandas
-   K-Means Clustering

## How to Run

Create and activate a virtual environment:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Install the required packages:

``` bash
pip install -r requirements.txt
```

Run the project:

``` bash
python3 main.py
```

Update the model path and input video path according to your project.

## Main Features

### Object Detection

A custom YOLOv5 model detects:

-   Players
-   Referees
-   Goalkeepers
-   Ball

### Player Tracking

ByteTrack gives players tracking IDs and follows them across video
frames.

### Camera Movement

Optical flow is used to estimate camera movement so that camera movement
does not directly affect player movement calculations.

### Perspective Transformation

The camera sees the pitch from an angle. Perspective transformation
converts image pixel positions into positions on a top-down pitch.

### Speed and Distance

The project calculates player speed and the total distance covered using
their transformed positions.

### Team Assignment

Players are assigned to teams using jersey colors. K-Means clustering is
used to identify the dominant jersey color.

### Ball Possession

The project finds the player closest to the ball and uses this
information to calculate team ball control.

## Output

The output video can show:

-   Player IDs
-   Player speed
-   Distance covered
-   Ball position
-   Ball possession
-   Team ball control
-   Camera movement

## Project Structure

``` text
football_analysis/
│
├── models/
├── input_videos/
├── output_videos/
├── trackers/
├── team_assigner/
├── camera_movement_estimator/
├── view_transformer/
├── speed_and_distance_estimator/
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```


