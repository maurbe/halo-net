import numpy as np
import matplotlib.pyplot as plt
plt.style.use('/Users/Mauro/Desktop/Biotop2/analysis/hmfA/src/thisstyle.mplstyle')
from matplotlib.colors import LogNorm

periodic_labels_predicted = np.load('periodic_labels_predictedT.npy')[64+50:-(64+50),
                                                                      64+50:-(64+50),
                                                                      64+50:-(64+50)]
cic_yT = np.load('/Users/Mauro/Desktop/Biotop2/source/cic_fieldsT/cic_yT.npy')[50:-50,
                                                                               50:-50,
                                                                               50:-50]
#---------
"""
un, inv, counts = np.unique(periodic_labels_predicted, return_inverse=True, return_counts=True)
counts[0] = 0
masses_pred = counts[inv]
np.save('temp_masses_pred.npy', masses_pred)
"""
# in Msun
masses_pred=np.load('temp_masses_pred.npy') * 5.66e10/64
cic_yT = cic_yT.flatten() * 5.66e10/64
#---------
print(masses_pred.min(), masses_pred.max())
print(cic_yT.min(), cic_yT.max())

mask1 = masses_pred>1e12
masses_pred=masses_pred[mask1]
cic_yT = cic_yT[mask1]

mask2 = cic_yT>1e12
masses_pred=masses_pred[mask2]
cic_yT = cic_yT[mask2]

Nbins=25
plt.figure()
hist, ybins, xbins, image = plt.hist2d(np.log10(1+cic_yT.flatten()), np.log10(1+masses_pred.flatten()),
                                        bins=Nbins, cmap='inferno', range=[[12,15], [12,15]]) #weights=1.0/cic_yT.flatten())[0]
#plt.contour(hist, levels=[10**3.5, 10**4.5, 10**5.5], extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()], linewidths=1)
plt.plot( [12,15], [12,15], linewidth=2)
plt.xlabel(r'$\log_{10}(M_{\mathrm{true}}/M_{\odot})$')
plt.ylabel(r'$\log_{10}(M_{\mathrm{pred}}/M_{\odot})$')
plt.savefig('comparisonT.png', dpi=200)
plt.show()
