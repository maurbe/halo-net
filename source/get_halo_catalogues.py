"""
    Script to generate the halo catalogues by running HOP over the z=0 snapshot.
"""
import numpy as np, pandas as pd, yt, csv
from yt.analysis_modules.halo_finding.halo_objects import HOPHaloFinder

def create_catalog(loadpath, savepath, name, cos_par):
    """
    Main function to call
    :param loadpath:    path to the particle z=0 snapshot
    :param savepath:    path to save the finder outputs
    :param name:        name of the simulation
    :param cos_par:     cosmological parameters as dictionary
    :return:            saves 4 files:  *.txt -> paths and halo name keys
                                        *.out -> properties of the individual halos
                                        *.csv -> same as *.out but in csv format
                                        *.h5  -> contains the individual particle ids in each halo
    Notes:              calling yt.load() with bbox is necessary, since a couple of particles are not folded
                        within the simulation domain
    """
    bbox = [[-0.50000001, 0.50000001],
            [-0.50000001, 0.50000001],
            [-0.50000001, 0.50000001]]

    ds      = yt.load(loadpath, cosmology_parameters=cos_par, bounding_box=bbox)
    dd      = ds.all_data()
    masses  = dd['all', 'Mass']
    assert len(masses) == 512**3


    def myField(field, data):
        """
        Notes:          helper function to assign another field to the yt dataset (ds)
                        here we assign a simple ID to each particle -> index-field
                        since the particle hierarchy in the simulation remains fixed, just use arange
        """
        return ds.arr(np.arange(512**3, dtype='float64'))
    ds.add_field(('all', 'particle_index'), function=myField, units='dimensionless', particle_type=True)

    # calling the HOP halo-finder
    halos = HOPHaloFinder(ds=ds, threshold=100)
    halos.dump(savepath + name)
    print('Dumped it!')

    # convert the *.out file to more readable *.csv
    file_to_convert = open(savepath + name + '.out')
    l = []
    for i, line in enumerate(csv.reader(file_to_convert, delimiter='\t', skipinitialspace=True)):
        if i > 2:
            l.append(line)
    Group, Mass, nr_part = [], [], []
    for line in l:
        Group.append(int(line[0]))
        Mass.append(float(line[1]))
        nr_part.append(float(line[2]))
    index = Group
    dataframe = pd.DataFrame(np.vstack((Group, Mass, nr_part)).transpose(), index=index,
                             columns=['Group', 'Mass', 'nr_part'])
    # save to disk
    dataframe.to_csv(savepath + name + '.csv', sep='\t', float_format='%.10f')
    print('Finished conversion!')

"""
create_catalog(loadpath='simT/wmap5_l100n512.00100',
               savepath='catalogT/',
               name='catalogT',
               cos_par={'current_redshift': 0.0,
                        'omega_lambda':     0.721,
                        'omega_matter':     0.279,
                        'hubble_constant':  0.701})
"""
"""
create_catalog(loadpath='simA/wmap5_l100n512r2.00100',
               savepath='catalogA/',
               name='catalogA',
               cos_par={'current_redshift': 0.0,
                        'omega_lambda':     0.721,
                        'omega_matter':     0.279,
                        'hubble_constant':  0.701})
"""