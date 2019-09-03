import numpy as np, os
import matplotlib.pyplot as plt

homedir = os.path.dirname(os.getcwd()) + '/'
gt = np.load(homedir + 'boxesA/gt_distancemap_normA.npy')[450]
pred = np.load(homedir + 'boxesA/predictionA.npy')[450]

plt.figure()
plt.imshow(gt, cmap='twilight_r')
plt.figure()
plt.imshow(pred, cmap='twilight_r')

plt.figure()

plt.show()