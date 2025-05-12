import cv2
import time
from ultralytics import YOLO

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
cv2.namedWindow("YOLO", cv2.WINDOW_NORMAL)

state = "idle"
prev_time = 0
cur_time = 0
player1_hand = ""
player2_hand = ""
timer = 0
DELAY_WAIT = 10
game_result = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cv2.putText(frame, f"{state} - {(DELAY_WAIT - timer):.1f}", (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if state == "wait":
        timer = time.time() - prev_time
        if timer >= DELAY_WAIT:
            state = "result"
            timer = DELAY_WAIT
            if player1_hand == player2_hand:
                game_result = "draw"
            elif (player1_hand, player2_hand) in [("scissors", "rock"), ("rock", "paper"), ("paper", "scissors")]:
                game_result = "player 2 wins"
            else:
                game_result = "player 1 wins"
            print(f"{game_result}: {player1_hand} vs {player2_hand}")

    key = cv2.waitKey(1)
    if key == ord('q') or cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1:
        break
    if key == ord('r'):
        state, timer = "idle", 0
    if state == "result":
        continue

    results = model(frame, verbose=False)[0]
    if results and len(results.boxes.xyxy) == 2:
        labels = []
        for label, xyxy in zip(results.boxes.cls, results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, xyxy)
            label = results.names[label.item()].lower()
            labels.append(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, label, (x1 + 20, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        player1_hand, player2_hand = labels
        if state == "idle" and player1_hand == "rock" and player2_hand == "rock":
            state = "wait"
            prev_time = time.time()

    cv2.imshow("Camera", frame)
    cv2.imshow("YOLO", frame)

cap.release()
cv2.destroyAllWindows()
