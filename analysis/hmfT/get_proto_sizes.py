import numpy as np, os
from analysis.helpers.hmf_functions import label_regions_periodic, plot_regions

def get_sizes(sim):
    homedir = os.path.dirname(os.getcwd()) + '/'
    predicted_distances = np.load(homedir + 'boxes' + sim + '/prediction' + sim + '.npy')

    plot_regions(distance=predicted_distances)

    #periodic_labels = label_regions_periodic(distance=predicted_distances, sim=sim, save_maxima=True)
    #np.save('saved_sizes' + sim + '/periodic_labels_predicted' + sim + '.npy', periodic_labels)

    #un, pure_sizes = np.unique(periodic_labels, return_counts=True)
    #np.save('saved_sizes' + sim + '/pure_sizes_predicted' + sim + '.npy', pure_sizes)

get_sizes(sim='T')