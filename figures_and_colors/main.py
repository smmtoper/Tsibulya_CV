import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.color import rgb2hsv
import numpy as np

im = imread("balls_and_rects.png")

binary = im.mean(2)
binary[binary > 0] = 1
labeled = label(binary)
regions = regionprops(labeled)
im_hsv = rgb2hsv(im)

colors = []
shapes = []

for region in regions:
    cy, cx = region.centroid  
    hue = im_hsv[int(cy), int(cx)][0]  
    hue = round(hue * 360) 
    colors.append(hue)

    if region.major_axis_length / region.minor_axis_length > 1.2:
        shapes.append('rectangle')
    else:
        shapes.append('circle')

rounded_colors = list(set(colors))

count_by_color_shape = {}
for color in rounded_colors:
    for shape in ['circle', 'rectangle']:
        count = sum(1 for c, s in zip(colors, shapes) if c == color and s == shape)
        count_by_color_shape[(color, shape)] = count

print(f"Общее количество фигур: {len(colors)}")
print("\nКоличество фигур по оттенкам и типам:")

for color in sorted(rounded_colors):
    print(f"Оттенок {color}:")
    circle_count = count_by_color_shape.get((color, 'circle'), 0)
    rect_count = count_by_color_shape.get((color, 'rectangle'), 0)
    print(f"  Круги: {circle_count}")
    print(f"  Прямоугольники: {rect_count}")

plt.imshow(im)
plt.show()
