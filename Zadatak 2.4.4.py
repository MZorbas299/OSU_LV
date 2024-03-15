import numpy as np
import matplotlib.pyplot as plt

black = np.zeros([50, 50, 3])
black.fill(0)
white = np.ones([50, 50, 3])
white.fill(255)
stack1 = np.vstack([black, white])
stack2 = np.vstack([white, black])
stack3 = np.hstack([stack1, stack2])
plt.imshow(stack3)
plt.show()