import yt, os, numpy as np
from analysis.helpers.plotting_help import *
"""
homedir     = os.path.dirname(os.getcwd()) + '/'
# I FIND THAT TAKING THE Z=9 SNAPSHOT CONVEYS THE CONCEPT BETTER
loadpath    = homedir + 'source/simT/wmap5_l100n512.00004'

bbox = [[-0.50000001, 0.50000001], [-0.50000001, 0.50000001], [-0.50000001, 0.50000001]]

cos_par={'current_redshift': 8.9,
         'omega_lambda':     0.721,
         'omega_matter':     0.279,
         'hubble_constant':  0.701}

ds      = yt.load(loadpath, cosmology_parameters=cos_par, bounding_box=bbox)
dd      = ds.all_data()

coordinates_IC = dd['all', 'Coordinates']
np.save('coordinates_IC_T.npy', coordinates_IC)
"""

coordinates_IC = np.load('coordinates_IC_T.npy')
x = coordinates_IC[:, 0]
y = coordinates_IC[:, 1]
z = coordinates_IC[:, 2]

f, ax = plt.subplots()
mask = np.logical_and(z>-0.02, z<0.02)
print(mask.sum())

ax.scatter(x[mask][::1000], y[mask][::1000], c='k', marker='.', s=0.5)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_aspect('equal')
plt.savefig('particles_IC_T.png', dpi=250)
plt.show()
