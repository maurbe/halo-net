import numpy as np
from analysis.helpers.hmf_functions_revised import retrieve_corrected_regions

def get_sizes(sim):
    predicted_distances = np.load('boxes' + sim + '/prediction' + sim + '.npy')

    import time
    s = time.time()
    periodic_labels, peak_values, corrected_sizes = retrieve_corrected_regions(distance=predicted_distances, sim=sim)
    print(time.time()-s, 's')

    np.save('hmf' + sim + '/src/periodic_labels_predicted' + sim + '.npy', periodic_labels)
    np.save('hmf' + sim + '/src/peak_values_predicted' + sim + '.npy', peak_values)
    np.save('hmf' + sim + '/src/corrected_sizes_predicted' + sim + '.npy', corrected_sizes)

get_sizes(sim='T')
get_sizes(sim='A')
