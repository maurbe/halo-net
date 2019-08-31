import numpy as np
import matplotlib.pyplot as plt

sizes = np.load('src/corrected_sizes_predictedT.npy')
plt.figure()
plt.hist(np.log10(sizes), 50, log=True, cumulative=-1, histtype='step')
plt.show()