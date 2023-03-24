import mediapipe as mp
import cv2
import numpy as np
from datetime import datetime

image_pose = cv2.VideoCapture(0)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7)
# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils

# now = datetime.now()

while True:
    now = datetime.now()
    suc, frame = image_pose.read()
    cap_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cap_in_RGB = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    resultant = pose_video.process(cap_in_RGB)
    # Draw pose:
    mp_drawing.draw_landmarks(image=frame, landmark_list=resultant.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3,
                                                                           circle_radius=3),
                              connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2,
                                                                             circle_radius=2))
    h, w = frame.shape[:2]
    lm = resultant.pose_landmarks

    avgw = (int(lm.landmark[11].x * w) + int(lm.landmark[12].x * w)) / 2
    avgh = (int(lm.landmark[11].x * h) + int(lm.landmark[12].x * h)) / 2

    avgw1 = round(avgw)
    avgh1 = round(avgh)

    point1 = np.array([avgw1, avgh1])
    point2 = np.array((int(lm.landmark[0].x * w), int(lm.landmark[0].y * h)))

    # calculating Euclidean distance
    dist = np.linalg.norm(point2 - point1)

    cv2.imshow('Cap Pose', frame)
    # print(dist)
    cv2.waitKey(30)

    # print(dist)

    if dist > 200:
        current_time = now.strftime("%H:%M:%S")
        print("you are not giving any attenion Current Time =", current_time)
