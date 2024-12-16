import numpy as np
from scipy.ndimage import label
import os

def process_image(npy_path):

    image = np.load(npy_path)    
    binary_image = (image > 0).astype(int)
    labeled_image, num_labels = label(binary_image)
    results = []
    for wire_label in range(1, num_labels + 1):
        wire_mask = labeled_image == wire_label
        _, parts_count = label(wire_mask)
        results.append(parts_count)

    return len(results), results

def main():

    input_files = [f for f in os.listdir() if f.endswith('.npy')]
    for file in input_files:
        num_wires, wire_parts = process_image(file)
        print(f"Файл: {file}")
        print(f"количество проводов: {num_wires}")
        for i, parts in enumerate(wire_parts, 1):
            if parts > 1:
                print(f"Провод {i}: порван на {parts} частей")
            else:
                print(f"провод {i}: целый")
        print("-" * 30)

if __name__ == "__main__":
    main()
