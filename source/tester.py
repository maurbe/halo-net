import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd

from matplotlib.colors import LinearSegmentedColormap
mcmap = LinearSegmentedColormap.from_list('mycmap', ['#3F1F47', '#5C3C9A', '#6067B3',
                                                     #   '#969CCA',
                                                     '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                     '#601A49', '#8C2E50', '#A14250',
                                                     '#B86759',
                                                     '#E0D9E1'][::-1])

cicA = np.load('cic_fieldsA/cic_yA.npy')
cic_idA = np.load('cic_fieldsA/cic_idA.npy')
distanceA = np.load('cic_fieldsA/distancemap_normA.npy')

densityA = np.load('cic_fieldsA/cic_densityA.npy')
densityA = nd.gaussian_filter(densityA, sigma=2.5, mode='wrap')
deltaA   = densityA/densityA.mean() - 1.0

mask = cicA<15*285
cicA[mask] = 0
cic_idA[mask] = 0
distanceA[mask] = 0

print(distanceA.mean())
print((distanceA==0).sum())

plt.figure()
plt.imshow(cic_idA[350], cmap='jet')

plt.figure()
plt.imshow(cicA[350], cmap='jet')

plt.figure()
plt.imshow(deltaA[350], cmap='magma')

plt.figure()
plt.imshow(distanceA[350], cmap=mcmap)
plt.show()

# ----------------
cicT = np.load('cic_fieldsT/cic_yT.npy')
cic_idT = np.load('cic_fieldsT/cic_idT.npy')
distanceT = np.load('cic_fieldsT/distancemap_normT.npy')

densityT = np.load('cic_fieldsT/cic_densityT.npy')
densityT = nd.gaussian_filter(densityT, sigma=2, mode='wrap')
deltaT   = densityT/densityT.mean() - 1.0

mask = cicT<15*285
cicT[mask] = 0
cic_idT[mask] = 0
distanceT[mask] = 0

plt.figure()
plt.hist(bins=100, histtype='step', x=distanceA.flatten(), color='red', log=True)
plt.hist(bins=100, histtype='step', x=distanceT.flatten(), color='green', log=True)
plt.show()