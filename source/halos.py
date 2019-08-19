"""
    Script to filter the entire halo catalog wrt a mass threshold.
"""
import numpy as np, h5py, pandas as pd
from tqdm import tqdm

def get_halo_props(sim, threshold):
    """
    Main function to call
    :param sim:         simulation name
    :param threshold:   mass threshold to filter individual halos, i.e. number of particles a halo has to have to be accepted as a halo
    :return:            saves 4 files:  halo_particle_ids_grouped*.npy  ->  "list" of "lists" of halos containing particle ids
                                        halo_particle_ids*.npy          ->  flattened version of halo_nr_parts_grouped*.npy
                                                                            only contains halo particles
                                        halo_nr_parts_grouped*.npy      ->  gives the actual number of particles ("mass")
                                                                            of each halo
                                        halo_nr_parts*.npy              ->  flat list of assigned mass labels of only halo particles
    """
    source = 'catalog' + sim + '/catalog' + sim

    # the halo_particles file contains the properties of the individual halo_particles!
    halo_particles = h5py.File(source + '.h5', 'r')
    # we also need the halo-keys; get them from the .txt file
    halo_names = np.loadtxt(source + '.txt', dtype='string')[:, 0]
    # the hop_halos file contains the properties of the idividual halos!
    hop_halos = pd.read_csv(source + '.csv', delimiter='\t')

    halo_nr_parts       = []        # FLATTENED: how many particles a halo contains == Mass of halo
    halo_particle_ids   = []        # FLATTENED: the id's of all halo particles
    halo_nr_parts_grouped = []      # GROUPED: how many particles a halo contains == Mass of halo
    halo_particle_ids_grouped = []  # GROUPED: the id's of all halo particles

    for i, name in enumerate(tqdm(halo_names)):
        # particle_inds :   particle id's in this halo
        particle_inds       = (halo_particles[name]['particle_index'][:]).astype(int)

        if len(particle_inds) >= threshold:
            halo_nr_parts += ( (len(particle_inds)*np.ones(len(particle_inds))).tolist())
            halo_particle_ids+=(particle_inds).tolist()

            halo_nr_parts_grouped.append(len(particle_inds))
            halo_particle_ids_grouped.append(particle_inds)

            # good assertion check...
            assert int(hop_halos['nr_part'][i]), len(particle_inds)

    halo_nr_parts       = np.asarray(halo_nr_parts)
    halo_particle_ids   = np.asarray(halo_particle_ids)

    halo_nr_parts_grouped = np.asarray(halo_nr_parts_grouped)
    halo_particle_ids_grouped = np.asarray(halo_particle_ids_grouped)

    print(halo_nr_parts.shape)
    print(halo_particle_ids.shape)

    print(halo_nr_parts_grouped.shape)
    print(halo_particle_ids_grouped.shape)

    np.save('halos' + sim + '/halo_nr_parts' + sim + '.npy', halo_nr_parts)
    np.save('halos' + sim + '/halo_particle_ids' + sim + '.npy', halo_particle_ids)

    np.save('halos' + sim + '/halo_nr_parts_grouped' + sim + '.npy', halo_nr_parts_grouped)
    np.save('halos' + sim + '/halo_particle_ids_grouped' + sim + '.npy', halo_particle_ids_grouped)
    print('Done!')

get_halo_props(sim='T', threshold=8*100)  # essentially take "all" (>800) halos and cut at a later time (in create_sets)
get_halo_props(sim='A', threshold=8*100)
#get_halo_props(sim='B', threshold=8*100)