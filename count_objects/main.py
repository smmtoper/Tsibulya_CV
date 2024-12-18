import cv2
import numpy as np
import sys

def detect_shapes(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "yellow": ((20, 100, 100), (30, 255, 255)),
        "green": ((35, 100, 100), (85, 255, 255)),
        "blue": ((100, 100, 100), (140, 255, 255)),
        "black": ((0, 0, 0), (180, 255, 50)),
    }

    detected_shapes = {
        "circle": [],
        "square": []
    }

    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 500:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            if radius > 10 and cv2.contourArea(contour) / (np.pi * radius ** 2) > 0.8:
                detected_shapes["circle"].append((color_name, center))
            else:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    detected_shapes["square"].append((color_name, center))

    return detected_shapes

def main():
    print("Запуск программы для определения количества объектов и их типов")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть камеру")
        sys.exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Не удалось получить кадр с камеры")
                break

            cv2.imshow("Camera", frame)
            detected_shapes = detect_shapes(frame)

            total_objects = len(detected_shapes["circle"]) + len(detected_shapes["square"])
            print(f"Общее количество объектов: {total_objects}")
            print(f"Количество кругов: {len(detected_shapes['circle'])}")
            print(f"Количество квадратов: {len(detected_shapes['square'])}")

            cv2.putText(frame, f"Total: {total_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Circles: {len(detected_shapes['circle'])}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2)
            cv2.putText(frame, f"Squares: {len(detected_shapes['square'])}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)

            cv2.imshow("Camera", frame)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        sys.exit("Программа завершена")


if __name__ == "__main__":
    main()
