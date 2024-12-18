import numpy as np

image = np.load('ps.npy')
from skimage import measure
import matplotlib.pyplot as plt
labeled_image, num_objects = measure.label(image, connectivity=2, return_num=True)
plt.imshow(labeled_image, cmap='nipy_spectral')
plt.title(f'Общее количество объектов: {num_objects}')
plt.colorbar()
plt.show()

object_sizes = [np.sum(labeled_image == i) for i in range(1, num_objects + 1)]
print(f'Количество объектов для каждого типа: {object_sizes}')
