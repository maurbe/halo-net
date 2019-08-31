import numpy as np
import matplotlib.pyplot as plt

T = np.load('boxesT/predictionT.npy')
A = np.load('boxesA/predictionA.npy')

plt.figure()
plt.hist(T.flatten(), bins=100, histtype='step', log=True, color='green')
plt.hist(A.flatten(), bins=100, histtype='step', log=True, color='blue')
plt.show()