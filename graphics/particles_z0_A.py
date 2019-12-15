import yt, os, numpy as np
from analysis.helpers.plotting_help import *
"""
homedir     = os.path.dirname(os.getcwd()) + '/'
# I FIND THAT TAKING THE Z=9 SNAPSHOT CONVEYS THE CONCEPT BETTER
loadpath    = homedir + 'source/simA/wmap5_l100n512r2.00100'

bbox = [[-0.50000001, 0.50000001], [-0.50000001, 0.50000001], [-0.50000001, 0.50000001]]

cos_par={'current_redshift': 0.0,
         'omega_lambda':     0.721,
         'omega_matter':     0.279,
         'hubble_constant':  0.701}

ds      = yt.load(loadpath, cosmology_parameters=cos_par, bounding_box=bbox)
dd      = ds.all_data()

coordinates_IC = dd['all', 'Coordinates']
np.save('coordinates_z0_A.npy', coordinates_IC)

"""
coordinates_IC = np.load('coordinates_z0_A.npy')
x = coordinates_IC[:, 0]
y = coordinates_IC[:, 1]
z = coordinates_IC[:, 2]

f, ax = plt.subplots()
mask = np.logical_and(x>8*0.002, x<12*0.002)
print(mask.sum())
ax.scatter(y[mask][::50], z[mask][::50], c='k', marker='.', s=0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.savefig('particles_z0_A.png', dpi=250)
plt.show()

"""
from tqdm import tqdm
masses = np.load(homedir + 'source/halosT/halo_nr_partsT.npy')

labelS = np.load(homedir + 'source/halosT/halo_particle_idsT.npy')
labels = np.zeros_like(x).astype('bool')
for idx in tqdm(labelS):
    labels[idx] = True

x_filtered = x[labels]
y_filtered = y[labels]
z_filtered = z[labels]
label_mask = np.logical_and(z_filtered>-0.02, z_filtered<0.02)

f, ax = plt.subplots()
print(label_mask.sum(), len(label_mask))
ax.scatter(x_filtered[label_mask], y_filtered[label_mask], c='k', marker='.', s=0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.show()
"""