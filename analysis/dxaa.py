import numpy as np
import matplotlib.pyplot as plt

A = np.load('boxesA/prediction_slidingA.npy')[150]

plt.figure()
plt.imshow(A, cmap='twilight_r')
plt.show()
