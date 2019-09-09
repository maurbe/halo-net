import numpy as np, os, yt
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from matplotlib.colors import colorConverter
from matplotlib.patches import Circle
from analysis.helpers.hmf_functions_revised import overplot_helper

def sselector(z):
    return (z > 100) & (z < 150)
def linear_kernel(Rs):
    r2 = np.arange(-round(Rs), round(Rs) + 1) ** 2
    dist2 = r2[:, None] + r2[:]
    r = np.sqrt(np.where(np.sqrt(dist2) <= Rs, dist2, np.inf))

    lin = np.zeros_like(r)
    for i in range(lin.shape[0]):
        for j in range(lin.shape[1]):
            if r[i, j] != np.inf:
                lin[i, j] = (1.0 - r[i, j]/Rs)
            else:
                continue
    return lin

homedir = os.path.dirname(os.getcwd()) + '/'
projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'

halo_particle_ids_grouped = np.load(projectdir + 'source/halosT/halo_particle_ids_groupedT.npy', encoding='latin1', allow_pickle=True)
halo_sizes_grouped = np.load(projectdir + 'source/halosT/halo_nr_parts_groupedT.npy')

mask = halo_sizes_grouped >= 16*285  # PLOT LOOKS BETTER, BUT NOT SURE IF THIS IS VALID...
halo_sizes_grouped = halo_sizes_grouped[mask]
halo_particle_ids_grouped = halo_particle_ids_grouped[mask]

# HAVE TO CONVERT TO COM OF THE MULTIPLE HALO SYSTEMS accounting for periodicity!!
"""
dsIC     = yt.load(projectdir + 'source/simT/wmap5almostlucie512.std')
ddIC     = dsIC.all_data()
coords   = np.asarray(ddIC[('all', 'Coordinates')])
np.save('coords.npy', coords)
"""
coords = np.load('coords.npy')


X_COM, Y_COM, Z_COM = [], [], []
for i, hid in enumerate(halo_particle_ids_grouped):
    #print len(hid), hid

    # this implementation follows the article about com in periodic boundary conditons from
    # https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

    # first, shift everything from [-0.5, 0.5] onto [0, 1]
    xmax, ymax, zmax = 1.0, 1.0, 1.0
    halo_coords = coords[hid]
    # print halo_coords.shape

    theta_x = (0.5 + halo_coords[:, 0]) * 2 * np.pi / xmax
    theta_y = (0.5 + halo_coords[:, 1]) * 2 * np.pi / ymax
    theta_z = (0.5 + halo_coords[:, 2]) * 2 * np.pi / zmax

    xi_x = np.cos(theta_x)
    xi_y = np.cos(theta_y)
    xi_z = np.cos(theta_z)

    zeta_x = np.sin(theta_x)
    zeta_y = np.sin(theta_y)
    zeta_z = np.sin(theta_z)

    # defining m_i = 1.0
    M = np.sum(len(halo_coords) * 1.0) + 1e-10
    xi_bar_x = 1.0 / M * np.sum(1.0 * xi_x)
    xi_bar_y = 1.0 / M * np.sum(1.0 * xi_y)
    xi_bar_z = 1.0 / M * np.sum(1.0 * xi_z)

    zeta_bar_x = 1.0 / M * np.sum(1.0 * zeta_x)
    zeta_bar_y = 1.0 / M * np.sum(1.0 * zeta_y)
    zeta_bar_z = 1.0 / M * np.sum(1.0 * zeta_z)

    theta_bar_x = np.arctan2(-zeta_bar_x, -xi_bar_x) + np.pi
    theta_bar_y = np.arctan2(-zeta_bar_y, -xi_bar_y) + np.pi
    theta_bar_z = np.arctan2(-zeta_bar_z, -xi_bar_z) + np.pi

    # convert back to [-0.5, 0.5]
    x_com = xmax * theta_bar_x / (2.0 * np.pi) -0.5
    y_com = ymax * theta_bar_y / (2.0 * np.pi) -0.5
    z_com = zmax * theta_bar_z / (2.0 * np.pi) -0.5

    X_COM.append(x_com)
    Y_COM.append(y_com)
    Z_COM.append(z_com)

    # ----- visual check -----
    #f, ax = plt.subplots()
    #ax.plot(halo_coords[:, 0], halo_coords[:, 1], 'b.')
    #ax.plot(x_com, y_com, 'g.')
    ##circles = [plt.Circle((xi, yi), radius=mxr, linewidth=0.1) for xi, yi, mxr in zip(com_halo_coords[:, 0], com_halo_coords[:, 1], max_r[hid])]
    ##c = matplotlib.collections.PatchCollection(circles)
    ##ax.add_collection(c)
    #ax.set_xlim((-0.5, 0.5))    #ax.set_xlim((0.0 - 32 * cellSize, 0.0 + 32 * cellSize))
    #ax.set_ylim((-0.5, 0.5))    #ax.set_ylim((0.0 - 32 * cellSize, 0.0 + 32 * cellSize))
    #plt.show()

X_COM = ((np.asarray(X_COM) + 0.5) * 512).astype(int)
Y_COM = ((np.asarray(Y_COM) + 0.5) * 512).astype(int)
Z_COM = ((np.asarray(Z_COM) + 0.5) * 512).astype(int)
RADII = (3.0 / (4.0*np.pi) * halo_sizes_grouped)**(1.0/3.0) / 1.0    # 1.5 solely for visually keeping the circles small
print('No. of gt proto halos:', len(RADII))
print('No. of gt proto halos in selected slice:', len(RADII[sselector(Z_COM)]), '\n')

arr = []
for xx, yy, rr in zip(X_COM[sselector(Z_COM)],
                      Y_COM[sselector(Z_COM)],
                      RADII[sselector(Z_COM)]):
    temp_arr = np.zeros((512, 512))
    temp_arr[
    (256 - int(rr)): (256 + int(rr) + 1),
    (256 - int(rr)): (256 + int(rr) + 1)] = linear_kernel(int(rr))
    arr.append(
        np.roll(np.roll(temp_arr, xx - 256, axis=0), yy - 256, axis=1))
heatmap_gt = np.max(arr, axis=0)
"""
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(heatmap_gt, zorder=0, cmap='bone_r')
plt.show()
"""




dist_p  = np.load(homedir + '/boxesT/predictionT.npy')
"""
labels_pred = overplot_helper(distance=dist_p)
np.save('predicted_labels_for_overplot.npy', labels_pred)
"""

dist_p = np.pad(dist_p, pad_width=64, mode='wrap')
labels_pred = np.load('predicted_labels_for_overplot.npy')
un, inv, counts = np.unique(labels_pred, return_inverse=True, return_counts=True)
print(len(un))
print((counts>=8*285).sum()) # BACKGROUND IS STILL INSIDE!


mass_labels = counts[inv].reshape(labels_pred.shape)    # the mass label of every pixel
mass_labels[mass_labels==mass_labels.max()] = 0.0   # set bg to 0

# CAUTION: If a number is missing, None is returned instead of a slice.
isolated_regions = nd.find_objects(labels_pred)
isolated_regions = [x for x in isolated_regions if x is not None]
print(len(isolated_regions))


peak_inds = []
assigned_radii = []
for u, slice3d in zip(un[1:], isolated_regions):
    cleaned_mask = (labels_pred[slice3d] == u)
    # the "None" in the last entry seems to be the stepsize!
    # good assertion check
    #assert np.unique(labels_pred[slice3d][cleaned_mask]) == u

    d = np.where(cleaned_mask > 0, dist_p[slice3d], 0)

    indices_of_peak_value = np.unravel_index(np.argmax(d), cleaned_mask.shape)
    #print indices_of_peak_value
    indices_of_peak_value = (indices_of_peak_value[0] + slice3d[0].start,
                             indices_of_peak_value[1] + slice3d[1].start,
                             indices_of_peak_value[2] + slice3d[2].start)
    assert labels_pred[indices_of_peak_value] == u
    # now perform the filtering
    if mass_labels[indices_of_peak_value] >= 16 * 285:
        peak_inds.append(indices_of_peak_value)
        assigned_radii.append((3.0 / (4.0 * np.pi) * mass_labels[indices_of_peak_value]) ** (
                    1.0 / 3.0) / 1.0)  # 1.5 solely for visually keeping the circles small


peak_inds = np.asarray(peak_inds)
assigned_radii = np.asarray(assigned_radii)

max_X, max_Y, max_Z = peak_inds[:, 0], peak_inds[:, 1], peak_inds[:, 2]
print('No. of pred. proto halos:', len(assigned_radii))
print('No. of pred. proto halos in selected slice:', len(assigned_radii[sselector(max_Z-64)]))
# REMOVE THE PAD-WIDTH OF 64 -> -64
# .....................................................................


# Plotting ............................................................
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.imshow(np.pad(heatmap_gt, pad_width=64, mode='wrap'), zorder=0, cmap='bone_r') # heatmap_gt

#my_blue = colorConverter.to_rgba('#1C6CAB', alpha=0.2)
#for xx, yy, rr in zip(X_COM[sselector(Z_COM)],
#                      Y_COM[sselector(Z_COM)],
#                      RADII[sselector(Z_COM)]):
#    circ = Circle((yy, xx), radius=rr, edgecolor='none', alpha=0.35, linewidth=1.5, facecolor=my_blue)
#    ax.add_patch(circ)

my_orange = colorConverter.to_rgba('#F8512C', alpha=0.2)
for xx, yy, rr in zip(max_X[sselector(max_Z-64)],
                      max_Y[sselector(max_Z-64)],
                      assigned_radii[sselector(max_Z-64)]):
    circ = Circle((yy, xx), radius=rr, edgecolor='#F8512C', linewidth=1.5, facecolor=my_orange)
    ax.add_patch(circ)

ax.set_xlabel('x [cells]')
ax.set_ylabel('y [cells]')
plt.savefig('overlay_new.png', dpi=300)
plt.show()

"""
fig, ax = plt.subplots(1, figsize=(12, 12))
cic_y = np.load(homedir + 'stitched_boxes/cic_y.npy')[:, :, :10]
ax.imshow(np.log10(np.max(cic_y, axis=-1)), alpha=0.5)
print len(np.unique(cic_y))
plt.show()
"""

