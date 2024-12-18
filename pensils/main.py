import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow

def count_pencils_in_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low = (0, 100, 0)  
    up = (255, 255, 255)
    pencil_mask = cv2.inRange(hsv_image, low, up)
    pencil_mask = cv2.dilate(pencil_mask, None, iterations=2)
    contours, _ = cv2.findContours(pencil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pencil_count = 0
    pencil_ratios = []

    for contour in contours:

        (center_x, center_y), (height, width), angle = cv2.minAreaRect(contour)
        max_side = max(height, width)
        min_side = min(height, width)

        if max_side > 1000 and min_side > 60:
            pencil_ratios.append(max_side / min_side)  
        else:
            pencil_ratios.append(0)

    pencil_ratios = np.array(pencil_ratios)
    pencils_in_image = np.sum(pencil_ratios > 18) 

    return pencils_in_image

def main():

    image_files = [f"img({i}).jpg" for i in range(1, 13)]
    total_pencils = 0

    for image_path in image_files:
        if not os.path.exists(image_path):
            print(f"Файл не найден: {image_path}")
            continue
        try:
            count = count_pencils_in_image(image_path)
            total_pencils += count
            print(f"На изображении {image_path} найдено карандашей: {count}")
        except ValueError as e:
            print(e)

    print(f"Суммарное количество на всех изображниях: {total_pencils}")

if __name__ == "__main__":
    main()
