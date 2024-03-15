import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
plt.imshow(img, alpha = 0.5)
plt.show()

plt.imshow(img[:, 150:300])
plt.show()

plt.imshow(np.rot90(img, 3))
plt.show()

plt.imshow(np.flip(img, axis=1))
plt.show()