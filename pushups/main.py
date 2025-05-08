import numpy as np
from sympy import false
from ultralytics import YOLO
from pathlib import Path
import cv2
import time
from ultralytics.utils.plotting import Annotator
count = 0
flag = False

def angle(a,b,c):
    d = np.arctan2(c[1] - b[1], c[0] - b[0])
    e = np.arctan2(a[1] - b[1], a[0] - b[0])
    angle_ = np.rad2deg(d - e)
    angle_ = angle_ + 360 if angle_ < 0 else angle_
    return 360 - angle_ if angle_ > 180 else angle_

def process(image, keypoints):

    left_ear_seen = keypoints[3][0] > 0 and keypoints[3][1] > 0
    right_ear_seen = keypoints[4][0] > 0 and keypoints[4][1] > 0

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_elbow = keypoints[7]
    right_elbow = keypoints[8]

    left_hand = keypoints[9]
    right_hand = keypoints[10]

    if left_ear_seen and not right_ear_seen:
        angle_elbow = angle(left_shoulder, left_elbow, left_hand)
    elif right_ear_seen and not left_ear_seen:
        angle_elbow = angle(right_shoulder, right_elbow, right_hand)
    else:
        return -1

    return int(angle_elbow)

path = Path(__file__).parent
image_path = path/ "pose.jpg"
model_path = path/"yolo11n-pose.pt"
image = cv2.imread(str(image_path))

model = YOLO(model_path)

cap = cv2.VideoCapture(0)
last_time = time.time()
writer = cv2.VideoWriter("out.mp4", cv2.VideoWriter_fourcc(*"avc1"),
                         10, (640, 480))
while cap.isOpened():
    ret, frame = cap.read()
    curr_time = time.time()
    last_time = curr_time
    results = model(frame)
    key = cv2.waitKey(1)
    if key ==ord("q"):
        break
    if not results:
        continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue

    keypoints = keypoints[0]
    if not keypoints:
        continue
    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0],result.orig_shape,5,True)
    annotated = annotator.result()
    angle_ = process(annotated,keypoints)
    if angle_ != -1:
        if flag and angle_ > 100:
            count += 1
            flag = False
        elif angle_ < 100:
            flag = True

    cv2.putText(frame, f"Push-ups: {count}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (25, 255, 25), 1)
    cv2.imshow("Pose", frame)
    writer.write(frame)
cap.release()
cv2.destroyAllWindows()
