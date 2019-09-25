import numpy as np, os
from analysis.helpers.hmf_functions_revised import retrieve_corrected_regions

def get_sizes(sim):
    homedir = os.path.dirname(os.getcwd()) + '/'
    predicted_distances = np.load(homedir + 'boxes' + sim + '/prediction' + sim + '.npy')

    periodic_labels, dmean_values, corrected_sizes = retrieve_corrected_regions(
        distance=predicted_distances, sim=sim, preload=True)

    np.save('src/periodic_labels_predicted' + sim + '.npy', periodic_labels)
    np.save('src/dmean_values_predicted' + sim + '.npy', dmean_values)
    np.save('src/corrected_sizes_predicted' + sim + '.npy', corrected_sizes)

get_sizes(sim='T')
