import numpy as np, os
from analysis.helpers.hmf_functions import plot_regions
import matplotlib.pyplot as plt
from analysis.helpers.hmf_functions_revised import mainfunction
sim = 'T'
N = 150
for N in range(0,512,50):
    true_distances = np.load('boxes' + sim + '/gt_distancemap_norm' + sim + '.npy')[N]
    predicted_distances = np.load('boxes' + sim + '/prediction' + sim + '.npy')[N]

    # periodic_labels is a flattened version, check shape
    mainfunction(distance=predicted_distances)


    plt.figure()
    plt.imshow(true_distances, cmap='twilight_r', vmin=0, vmax=1)



    homedir = os.path.dirname(os.getcwd()) + '/'
    masses = np.load(homedir + 'source/cic_fields'+sim+'/cic_y'+sim+'.npy')[N].astype('int')
    masses[masses<16*285] = 0

    where0 = masses==0
    IDp = np.random.permutation(np.arange(10, 10 + masses.max()+1))[masses]
    IDp[where0] = 0.0

    plt.figure()
    plt.imshow(IDp)
    plt.show()