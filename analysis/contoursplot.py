import numpy as np
import matplotlib.pyplot as plt
import os

homedir = os.path.dirname(os.getcwd()) + '/'
ID = np.load(homedir + 'source/cic_fieldsT/cic_idA.npy')[50].astype('int')
P  = np.load('boxesA/predictionA.npy')[50]

where0 = ID==0
IDp = np.random.permutation(np.arange(10, 10 + ID.max()+1))[ID]
IDp[where0] = 0.0

plt.figure()
plt.imshow(IDp, cmap='twilight_shifted')
plt.show()
