"""
    Script to create the cic_density and distance maps (raw and normalized) (with binary_closing) for a simulation snapshot.
"""
from __future__ import division
from tqdm import tqdm
from source.cython_helpers.cython_cic import CICDeposit_3, masking
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
    cic_density = cic_density.astype('float32')
    np.save('cic_fields' + sim + '/cic_density' + sim + '.npy', cic_density[int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer),
                                                                            int(buffer):-int(buffer)])

def create_mass_label_map(sourcepath, sim, buffer=4):
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

    # save np.load
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    halo_particle_ids = np.load('halos' + sim + '/halo_particle_ids_grouped' + sim + '.npy', encoding='latin1')
    # restore np.load for future normal usage
    np.load = np_load_old
    halo_particle_ids = halo_particle_ids[::-1] # reverse it for hierarchical depositon
    # assert that the array is in fact sorted from small to large, deposit from small to large
    assert all(len(halo_particle_ids[i]) <= len(halo_particle_ids[i+1])
               for i in range(len(halo_particle_ids)-1))

    possible_vals = [1.0, 0.0, -1.0]
    shifts = np.asarray(list(itertools.product(possible_vals, repeat=3)))

    extended_dims = 512 + 2 * int(buffer)
    cic_y = np.zeros(shape=(extended_dims, extended_dims, extended_dims))
    cic_id= np.zeros(shape=(extended_dims, extended_dims, extended_dims))

    for ID, halo_indices in enumerate(tqdm(halo_particle_ids)):  # maybe merge the two for loops, this and not the next but the next next!

        true_size = len(halo_indices)
        temp_coords = coords8[halo_indices]

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
        subvolume_mass = nd.morphology.binary_opening(input=subvolume, structure=linear_kernel(Rs=3),
                                                      border_value=0) * true_size
        subvolume_ID = nd.morphology.binary_opening(input=subvolume, structure=linear_kernel(Rs=3),
                                                      border_value=0) * (ID+1)
        # * true_size gives us the mass (in terms of particle number)!

        cic_y[x_start:x_end, y_start:y_end, z_start:z_end] = \
            np.max(np.stack([cic_y[x_start:x_end, y_start:y_end, z_start:z_end], subvolume_mass]), axis=0)

        cic_id[x_start:x_end, y_start:y_end, z_start:z_end] = \
            np.max(np.stack([cic_id[x_start:x_end, y_start:y_end, z_start:z_end], subvolume_ID]), axis=0)


    cic_y = cic_y[int(buffer):-int(buffer), int(buffer):-int(buffer), int(buffer):-int(buffer)].astype('float32')
    np.save('cic_fields' + sim + '/cic_y' + sim + '.npy', cic_y)

    cic_id = cic_id[int(buffer):-int(buffer), int(buffer):-int(buffer), int(buffer):-int(buffer)].astype('float32')
    np.save('cic_fields' + sim + '/cic_id' + sim + '.npy', cic_id)

def create_distance_maps(sim):
    # now the normalized edt
    cic_id = np.load('cic_fields' + sim + '/cic_id' + sim + '.npy')
    assert (cic_id.shape == (512, 512, 512))
    expanded_cic_id = np.pad(cic_id, pad_width=128, mode='wrap')
    distances = edt.edt(expanded_cic_id.astype(int), black_border=False)[128:-128, 128:-128, 128:-128]
    np.save('cic_fields' + sim + '/distancemap_raw' + sim + '.npy', distances.astype('float32'))

    # now normalize the distances cluster by cluster
    unique_ids = np.unique(cic_id)[1:]  # cut the BG=0
    isolated_region = nd.find_objects(cic_id.astype(int))  # unique isolated regions
    # the following is necessary, since some clusters are inside larger ones and thus LOST.
    # however, find_objects just assigns "None", if a label (e.g. 4) is missing -> have to filter those out
    isolated_region = [x for x in isolated_region if x is not None]

    print(len(unique_ids), len(isolated_region))
    assert (distances.shape == (512, 512, 512))
    assert len(unique_ids)==len(isolated_region)
    assert cic_id.shape==distances.shape
    normalized_distances = np.zeros_like(cic_id)
    for ID, region in tqdm(zip(unique_ids, isolated_region)):
        subvolume_dist = distances[region]
        subvolume_id   = cic_id[region]
        sub_mask = (subvolume_id==ID)
        normalized_distances[region][sub_mask] = subvolume_dist[sub_mask]/(subvolume_dist[sub_mask].max()+1e-8)

    np.save('cic_fields' + sim + '/distancemap_norm' + sim + '.npy', normalized_distances.astype('float32'))


#create_cic_density('simT/wmap5almostlucie512.std', sim='T')
#create_cic_density('simA/wmap5almostlucie512r2.std', sim='A')

#create_mass_label_map('simT/wmap5almostlucie512.std', sim='T')
#create_distance_maps(sim='T')

#create_mass_label_map('simA/wmap5almostlucie512r2.std', sim='A')
#create_distance_maps(sim='A')