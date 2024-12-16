import numpy as np
from scipy.ndimage import label, binary_hit_or_miss
import os

def detect_stars(image):

    plus_kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]])
    
    cross_kernel = np.array([[1, 0, 1],
                             [0, 1, 0],
                             [1, 0, 1]])

    plus_matches = binary_hit_or_miss(image, plus_kernel)
    cross_matches = binary_hit_or_miss(image, cross_kernel)
    _, num_plus = label(plus_matches)
    _, num_cross = label(cross_matches)

    return num_plus, num_cross

def main():

    file_name = "stars.npy"
    if not os.path.exists(file_name):
        print(f"Файл {file_name} не найден")
        return

    image = np.load(file_name)
    binary_image = (image > 0).astype(int)
    num_plus, num_cross = detect_stars(binary_image)

    print(f"Количество 'плюсов': {num_plus}")
    print(f"Количество 'крестов': {num_cross}")
    print(f"общее количество звёздочек: {num_plus + num_cross}")


if __name__ == "__main__":
    main()
