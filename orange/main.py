from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from skimage.draw import disk

path = Path(__file__).parent
model = YOLO(path / "facial_best.pt")
oranges = cv2.imread(str(path / "oranges.png"))
mask = cv2.inRange(cv2.cvtColor(oranges, cv2.COLOR_BGR2HSV),
                   (10, 240, 200), (15, 255, 255))
mask = cv2.dilate(mask, np.ones((7, 7)))
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
bbox = cv2.boundingRect(max(contours, key=cv2.contourArea))
x, y, w, h = bbox
roi = oranges[y:y + h, x:x + w]
struct = np.zeros((11, 11), np.uint8)
rr, cc = disk((5, 5), 5)
struct[rr, cc] = 1
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame)[0]
    annotated = result.plot()
    masks = result.masks
    if not masks:
        continue

    global_mask = sum(mask.data.numpy()[0] for mask in masks)
    global_mask = cv2.resize(global_mask, (frame.shape[1], frame.shape[0])).astype('uint8')
    global_mask = cv2.dilate(global_mask, struct, iterations=2)
    pos = np.where(global_mask > 0)
    if pos[0].size == 0 or pos[1].size == 0:
        continue

    min_y, max_y = int(np.min(pos[0]) * 0.9), int(np.max(pos[0]) * 1.1)
    min_x, max_x = int(np.min(pos[1]) * 0.9), int(np.max(pos[1]) * 1.1)
    part = (frame * global_mask[..., None])[min_y:max_y, min_x:max_x]
    mask_roi = global_mask[min_y:max_y, min_x:max_x]
    part = cv2.resize(part, (w, h))
    mask_resized = (cv2.resize(mask_roi, (w, h)) * 255).astype('uint8')
    combined = cv2.add(cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask_resized)),
                       cv2.bitwise_and(part, part, mask=mask_resized))

    cv2.imshow("Image", combined)
    cv2.imshow("Parts", annotated)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
