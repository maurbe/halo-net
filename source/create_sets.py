"""
    Script to create training and validation sets of (x,y).
"""
from __future__ import division
import numpy as np, os
import scipy.ndimage as nd
from sklearn.preprocessing import scale

def create_boxes(sim, halo_mass_threshold, purpose):
    """
    Main function to call to create subbox pairs of (x,y)
    :param sim:         simulation name
    :return:            saves pairs of (x,y)
    """
    homedir = os.path.dirname(os.getcwd()) + '/'
    dims = 512

    cic_density = np.load('cic_fields' + sim + '/cic_density' + sim + '.npy') * dims**3
    density = nd.gaussian_filter(cic_density, sigma=2.5, mode='wrap') # 1 or higher?
    delta_scaled = scale(density.flatten()).reshape(density.shape)
    np.save(homedir + 'analysis/boxes' + sim + '/input_density_contrast' + sim + '.npy', delta_scaled.astype('float32'))

    # NO smoothing if possible and NO normalization
    distance_map = np.load('cic_fields' + sim + '/distancemap_norm' + sim + '.npy')
    cic_y = np.load('cic_fields' + sim + '/cic_y' + sim + '.npy')
    # now the filtering of the distance map (directly)
    mass_mask = (cic_y < halo_mass_threshold)
    distance_map[mass_mask] = 0.0
    np.save(homedir + 'analysis/boxes' + sim + '/gt_distancemap_norm' + sim + '.npy', distance_map.astype('float32'))

    buffer = 32
    shift = 64
    size = shift + 2 * buffer
    num_dimension = int(dims/shift)
    c = 0

    delta_scaled = np.pad(delta_scaled, pad_width=buffer, mode='wrap')
    distance_map = np.pad(distance_map, pad_width=buffer, mode='wrap')

    for k in range(num_dimension):
        for j in range(num_dimension):
            for i in range(num_dimension):
                print(c)

                box_de = delta_scaled[
                         i * shift : (i + 2) * shift,
                         j * shift : (j + 2) * shift,
                         k * shift : (k + 2) * shift]
                box_ds = distance_map[
                         i * shift : (i + 2) * shift,
                         j * shift : (j + 2) * shift,
                         k * shift : (k + 2) * shift]

                box_de = box_de.astype('float32')
                box_ds = box_ds.astype('float32')
                assert box_de.dtype == 'float32'
                assert box_ds.dtype == 'float32'

                savepath = 'sets' + sim + '/' + purpose + sim + '/'
                np.save(savepath + 'x_{}'.format(c), box_de[..., np.newaxis])
                np.save(savepath + 'y_{}'.format(c), box_ds[..., np.newaxis])

                c += 1

#create_boxes(sim='T', halo_mass_threshold=16*285, purpose='training')
create_boxes(sim='A', halo_mass_threshold=16*285, purpose='validation')
