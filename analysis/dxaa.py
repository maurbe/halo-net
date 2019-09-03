import numpy as np
import matplotlib.pyplot as plt

A = np.load('boxesA/predictionA.npy')[150]
T = np.load('boxesT/predictionT.npy')[150]

plt.figure()
plt.imshow(A, cmap='twilight_r')

plt.figure()
plt.imshow(A, cmap='twilight_r')
plt.show()
