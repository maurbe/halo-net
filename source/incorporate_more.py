"""
    Script to create the cic_density and distance maps (with binary_closing) for a simulation.
"""
from __future__ import division
from tqdm import tqdm
from cython_helpers.cython_cic import CICDeposit_3, CICDeposit_3_weights_adjusted, masking
import numpy as np, yt, edt, itertools
import scipy.ndimage as nd

def linear_kernel(Rs):
    r2 = np.arange(-round(Rs), round(Rs) + 1) ** 2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    r = np.sqrt(np.where(np.sqrt(dist2) <= Rs, dist2, np.inf))

    lin = np.zeros_like(r)
    for i in range(lin.shape[0]):
        for j in range(lin.shape[1]):
            for k in range(lin.shape[2]):
                if r[i, j, k] != np.inf:
                    lin[i, j, k] = 1.0
                else:
                    continue
    return lin

def create_mask(A, temp_co, buffer_pixels):
    if A == 1:
        mask = (temp_co <= A * (0.5 + buffer_pixels / 512.0))
    elif A == -1:
        mask = (temp_co >= A * (0.5 + buffer_pixels / 512.0))
    elif A == 0:
        mask = np.ones(len(temp_co), dtype=int)
    return mask


def create_cic_density(sourcepath, sim, buffer=4.0):
    """
    Main function to call for cic depositing particles onto the grid
    :param sourcepath:      path to particle snapshot file
    :param sim:             simulation name
    :param buffer:          buffer to get around the periodic boundary conditions issue
    :return:                saves (unsmoothed) raw cic_density
    Notes:                  Technique to account for PBC:
                                1. the grid is extended on each side by buffer pixels -> additonal region envelopping the box
                                2. the particle coords falling inside this additional region are added -> extended coords
                                3. the extended coords are deposited onto grid by means of cic -> caution when setting the
                                    parameters to the cic call (e.g. left_edge has to be defined appropriately)
                                4. trim the extended cube back to the correct (smaller) size and save
    """
    dims = 512
    ds8 = yt.load(sourcepath)
    dd8 = ds8.all_data()
    coords8 = np.asarray(dd8['all', 'Coordinates'])  # strip the units

    possible_vals = [1.0, 0.0, -1.0]
    shifts = np.asarray(list(itertools.product(possible_vals, repeat=3)))

    extended_coords = coords8.copy()
    for shift in tqdm(shifts):
        if np.allclose([shift], 0):
            # this is necessary, since the (0, 0, 0) case is already the coords8 case!
            continue

        shf_x = shift[0]
        shf_y = shift[1]
        shf_z = shift[2]

        temp_coords = coords8.copy()
        temp_coords[:, 0] += shf_x
        temp_coords[:, 1] += shf_y
        temp_coords[:, 2] += shf_z

        mask_x = create_mask(A=shf_x, temp_co=temp_coords[:, 0], buffer_pixels=buffer)
        mask_y = create_mask(A=shf_y, temp_co=temp_coords[:, 1], buffer_pixels=buffer)
        mask_z = create_mask(A=shf_z, temp_co=temp_coords[:, 2], buffer_pixels=buffer)

        final_mask = np.logical_and(mask_z, np.logical_and(mask_y, mask_x))
        extended_coords = np.concatenate((extended_coords, temp_coords[final_mask]))

    extended_dims = 512 + 2 * int(buffer)
    extended_npositions = np.int64(len(extended_coords))
    extended_field = np.zeros(shape=(extended_dims, extended_dims, extended_dims), dtype='float64')
    extended_leftEdge = np.array([-0.5, -0.5, -0.5]) - buffer / 512.0
    extended_gridDimension = np.asarray([extended_dims, extended_dims, extended_dims], dtype='int32')
    cellSize = np.float64(1.0 / dims)  # this remains unchanged
    extended_masses = np.ones(len(extended_coords)) * 0.279 / dims ** 3  # this remains unchanged

    cic_density = CICDeposit_3(extended_coords[:, 0], extended_coords[:, 1], extended_coords[:, 2],
                            extended_masses,
                            extended_npositions,
                            extended_field, extended_leftEdge, extended_gridDimension, cellSize)
    np.save('cic_fields' + sim + '/cic_density' + sim + '.npy', cic_density[int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer)])



def create_distance_map(sourcepath, sim, buffer=8.0):
    """
    Main function to call for creating distance maps
    :param sourcepath:      path to particle snapshot file
    :param sim:             simulation name
    :param buffer:          buffer to get around the periodic boundary conditions issue
    :return:                saves (unsmoothed) raw cic_density
    Notes:                  Same technique to account for PBC.
                            However, the halo mass label of each halo particle is deposited.
    """

    dims = 512
    ds8 = yt.load(sourcepath)
    dd8 = ds8.all_data()
    coords8 = np.asarray(dd8['all', 'Coordinates'])  # strip the units

    halo_particle_ids = np.load('halos' + sim + '/halo_particle_ids_grouped' + sim + '.npy', encoding='latin1')
    possible_vals = [1.0, 0.0, -1.0]
    shifts = np.asarray(list(itertools.product(possible_vals, repeat=3)))

    extended_dims = 512 + 2 * int(buffer)
    cic_y = np.zeros(shape=(extended_dims, extended_dims, extended_dims))
    for halo_indices in tqdm(halo_particle_ids):  # maybe merge the two for loops, this and not the next but the next next!

        true_size = len(halo_indices)
        temp_coords = coords8[halo_indices]

        import matplotlib.pyplot as plt
        extended_coords = temp_coords.copy()
        #plt.figure()
        #plt.plot(extended_coords[:, 0], extended_coords[:, 1], 'b.')
        #plt.xlim(-0.5 - 16.0 / 512., +0.5 + 16.0 / 512.)
        #plt.ylim(-0.5 - 16.0 / 512., +0.5 + 16.0 / 512.)
        #plt.grid()
        for shift in shifts:
            if np.allclose([shift], 0):
                # this is necessary, since the (0, 0, 0) case is already the coords8 case!
                continue

            shf_x = shift[0]
            shf_y = shift[1]
            shf_z = shift[2]

            shift_coords = temp_coords.copy()
            shift_coords[:, 0] += shf_x
            shift_coords[:, 1] += shf_y
            shift_coords[:, 2] += shf_z

            mask_x = create_mask(A=shf_x, temp_co=shift_coords[:, 0], buffer_pixels=buffer)
            mask_y = create_mask(A=shf_y, temp_co=shift_coords[:, 1], buffer_pixels=buffer)
            mask_z = create_mask(A=shf_z, temp_co=shift_coords[:, 2], buffer_pixels=buffer)

            final_mask = np.logical_and(mask_z, np.logical_and(mask_y, mask_x))
            extended_coords = np.concatenate((extended_coords, shift_coords[final_mask]))
        #plt.figure()
        #print(extended_coords.shape)
        #plt.plot(extended_coords[:, 0], extended_coords[:, 1], 'g.')
        #plt.xlim(-0.5-16.0/512., +0.5+16.0/512.)
        #plt.ylim(-0.5 - 16.0 / 512., +0.5 + 16.0 / 512.)
        #plt.grid()
        #plt.show()

        extended_field = np.zeros(shape=(extended_dims, extended_dims, extended_dims), dtype='float64')
        extended_leftEdge = np.array([-0.5, -0.5, -0.5]) - buffer / 512.0
        extended_gridDimension = np.asarray([extended_dims, extended_dims, extended_dims], dtype='int32')
        cellSize = np.float64(1.0 / dims)  # this remains unchanged

        # we are directly depositing the halo mass into cells (top-down)
        temp = masking(posx=extended_coords[:, 0],
                       posy=extended_coords[:, 1],
                       posz=extended_coords[:, 2],
                       mass=np.ones(len(extended_coords), dtype='float64'),  # just deposit 1's, assign the mass later!
                       npositions=np.int64(len(extended_coords)),
                       field=extended_field,
                       leftEdge=extended_leftEdge,
                       gridDimension=extended_gridDimension,
                       cellSize=cellSize)

        isolated_region = nd.find_objects(temp.astype(int))[0]
        x_start, x_end = isolated_region[0].start, isolated_region[0].stop
        y_start, y_end = isolated_region[1].start, isolated_region[1].stop
        z_start, z_end = isolated_region[2].start, isolated_region[2].stop
        subvolume = temp[isolated_region]  # isolate subvolume, modify and then re-assign the max but directly to cic_y!!

        # the border_value arg is NOT in the documentation, but it is generally an arg to binary_erosion which is done here!
        # have to choose this combination of border_values, since otherwise all border pixels are set to 0, which breaks pbc!
        subvolume = nd.morphology.binary_closing(input=subvolume, structure=linear_kernel(Rs=3), border_value=1)
        subvolume = nd.morphology.binary_opening(input=subvolume, structure=linear_kernel(Rs=3), border_value=0) * true_size
        # * true_size gives us the mass (in terms of particle number)!

        # additional edt here, then set the pixels in cic_y to zero where distance < 3 (?)
        #:Todo:    decide on a threshold, currently 3.0
        temp_edt = edt.edt(subvolume, black_border=False)
        mask_filtering_1s_and_2s = (temp_edt < 3)   # set border pixels (with distance 1 or 2) to 0
        subvolume[mask_filtering_1s_and_2s] = 0.0
        # --------------------------------------------

        cic_y[x_start:x_end, y_start:y_end, z_start:z_end] = \
            np.max(np.stack([cic_y[x_start:x_end, y_start:y_end, z_start:z_end], subvolume]), axis=0)


    cic_y = cic_y[int(buffer):-int(buffer), int(buffer):-int(buffer), int(buffer):-int(buffer)].astype('float32')
    np.save('cic_fields' + sim + '/cic_y' + sim + '.npy', cic_y)

    # now the edt...
    cic_y = np.load('cic_fields'+sim+'/cic_y'+sim+'.npy')
    print(cic_y.shape, 'has to be 512!')
    expanded_cic_y = np.pad(cic_y, pad_width=128, mode='wrap')
    distances = edt.edt(expanded_cic_y.astype(int), black_border=False)[128:-128, 128:-128, 128:-128]
    np.save('cic_fields' + sim + '/distancemap' + sim + '.npy', distances.astype('float32'))

#create_cic_density('simT/wmap5almostlucie512.std', sim='T')
#create_cic_density('simA/wmap5almostlucie512r2.std', sim='A')

#create_distance_map('simT/wmap5almostlucie512.std', sim='T')
#create_distance_map('simA/wmap5almostlucie512r2.std', sim='A')


def displacement(sourcepath, sim, buffer=4.0):
    """
    Main function to call for cic depositing particles onto the grid
    :param sourcepath:      path to particle snapshot file
    :param sim:             simulation name
    :param buffer:          buffer to get around the periodic boundary conditions issue
    :return:                saves (unsmoothed) raw cic_density
    Notes:                  Technique to account for PBC:
                                1. the grid is extended on each side by buffer pixels -> additonal region envelopping the box
                                2. the particle coords falling inside this additional region are added -> extended coords
                                3. the extended coords are deposited onto grid by means of cic -> caution when setting the
                                    parameters to the cic call (e.g. left_edge has to be defined appropriately)
                                4. trim the extended cube back to the correct (smaller) size and save
    """
    dims = 512
    ds8 = yt.load(sourcepath)
    dd8 = ds8.all_data()
    coords8 = np.asarray(dd8['all', 'Coordinates'])  # strip the units
    velos8 = np.asarray(dd8['all', 'Velocities'])
    print velos8.shape

    possible_vals = [0.0]#[1.0, 0.0, -1.0]
    shifts = np.asarray(list(itertools.product(possible_vals, repeat=3)))

    extended_coords= coords8.copy()
    extended_velos = velos8.copy()
    for shift in tqdm(shifts):
        if np.allclose([shift], 0):
            # this is necessary, since the (0, 0, 0) case is already the coords8 case!
            continue

        shf_x = shift[0]
        shf_y = shift[1]
        shf_z = shift[2]

        temp_velos  = velos8.copy()
        temp_coords = coords8.copy()
        temp_coords[:, 0] += shf_x
        temp_coords[:, 1] += shf_y
        temp_coords[:, 2] += shf_z

        mask_x = create_mask(A=shf_x, temp_co=temp_coords[:, 0], buffer_pixels=buffer)
        mask_y = create_mask(A=shf_y, temp_co=temp_coords[:, 1], buffer_pixels=buffer)
        mask_z = create_mask(A=shf_z, temp_co=temp_coords[:, 2], buffer_pixels=buffer)

        final_mask = np.logical_and(mask_z, np.logical_and(mask_y, mask_x))
        extended_coords= np.concatenate((extended_coords, temp_coords[final_mask]))
        extended_velos = np.concatenate((extended_velos, temp_velos[final_mask]))

    extended_dims = 512 + 2 * int(buffer)
    extended_npositions = np.int64(len(extended_coords))
    extended_field = np.zeros(shape=(extended_dims, extended_dims, extended_dims), dtype='float64')
    extended_leftEdge = np.array([-0.5, -0.5, -0.5]) - buffer / 512.0
    extended_gridDimension = np.asarray([extended_dims, extended_dims, extended_dims], dtype='int32')
    cellSize = np.float64(1.0 / dims)  # this remains unchanged
    extended_vx = extended_velos[:, 0]
    extended_vy = extended_velos[:, 1]
    extended_vz = extended_velos[:, 2]

    cic_vx = CICDeposit_3_weights_adjusted(extended_coords[:, 0], extended_coords[:, 1], extended_coords[:, 2],
                          extended_vx,
                          extended_npositions,
                          extended_field.copy(), extended_field.copy(),
                          extended_leftEdge, extended_gridDimension, cellSize)
    cic_vy = CICDeposit_3_weights_adjusted(extended_coords[:, 0], extended_coords[:, 1], extended_coords[:, 2],
                          extended_vy,
                          extended_npositions,
                          extended_field.copy(), extended_field.copy(),
                          extended_leftEdge, extended_gridDimension, cellSize)
    cic_vz = CICDeposit_3_weights_adjusted(extended_coords[:, 0], extended_coords[:, 1], extended_coords[:, 2],
                          extended_vz,
                          extended_npositions,
                          extended_field.copy(), extended_field.copy(),
                          extended_leftEdge, extended_gridDimension, cellSize)
    np.save('cic_fields' + sim + '/cic_vx' + sim + '.npy', cic_vx[int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer)])
    np.save('cic_fields' + sim + '/cic_vy' + sim + '.npy', cic_vy[int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer)])
    np.save('cic_fields' + sim + '/cic_vz' + sim + '.npy', cic_vz[int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer)])

displacement('simT/wmap5almostlucie512.std', sim='T')