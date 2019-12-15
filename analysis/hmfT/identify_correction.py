"""
    Purpose:    Identify the best fit by calibrating with the ground truth of (T)
    Comment:    This is the main script for this computation. The one for (A) is just a mere quality check of the fit.
"""

from __future__ import division
import pandas as pd, os
from analysis.helpers.plotting_help import *
import scipy
from tqdm import trange
from analysis.helpers.hmf_functions_revised import find_correction, fit

def colorbar(mappable, colorbar_label):
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax = mappable.axes
    fig = ax.figure
    cax = inset_axes(ax,
                   width="5%",  # width = % of parent_bbox width
                   height="85%",  # height : %
                   loc='right',
                   bbox_to_anchor=(0.0, 0.0, 1.1, 1.04),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
    cb = fig.colorbar(mappable, cax=cax, label=colorbar_label, orientation='vertical')
    cb.ax.tick_params(labelsize=15)
    return cb

projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data       = pd.read_csv(projectdir + 'source/catalogT/catalogT.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
print('gt no. of halos', len(halo_sizes))
halo_sizes = np.log10(halo_sizes)
bins_hmf   = 25 # should be same as final hmf
true_counts, bin_edges_Y = plt.hist(halo_sizes, bins=bins_hmf)[0:2]
plt.close()

homedir = os.path.dirname(os.getcwd()) + '/'
predicted_distances = np.load(homedir + 'boxesT/predictionT.npy')

#raw_masses, dmean_vals = find_correction(distance=predicted_distances, sim='T', homedir=homedir)
#np.save('correction_src/raw_masses_matrix.npy', raw_masses)
#np.save('correction_src/dmean_matrix.npy', dmean_vals)

raw_masses = np.load('correction_src/raw_masses_matrix.npy')
log_masses = np.log10(raw_masses)

dmean_vals = np.load('correction_src/dmean_matrix.npy')
bin_edges_X = np.linspace(0, dmean_vals.max(), dmean_vals.shape[1]+1, endpoint=True)

#plt.figure()
#for id in [10, 15, 20]:
#    plt.plot(log_masses[id], dmean_vals[id], linewidth=0.5)


hist = scipy.stats.binned_statistic_2d(x=dmean_vals.flatten(), y=log_masses.flatten(),
                                       values=None, statistic='count',
                                       bins=[bin_edges_X, bin_edges_Y])[0]
hist_to_manipulate = hist.copy()


# for loop to determine the index/contour_thresh/mass so that the number of intersections in this given 2D bin matches best
# the number of halos in the ground truth halo mass histogram (normal one, not the accumulated one)
Dv                  = []
Mv                  = []
diffs               = []
markers             = []
index_pairs_of_dots = []
difference_map      = np.zeros_like(hist)

for j in trange(hist_to_manipulate.shape[1]):

    line = hist_to_manipulate[:, j]
    tc = true_counts[j]
    difference_map[:, j] = 1.0-(line-tc)/tc
    indices_of_best_fit = np.argsort(abs(line-tc))[:1]  # retain the best N indices (N=1)
    """
    plt.figure()
    plt.imshow(hist_to_manipulate, cmap='bone', vmin=0, vmax=1)
    plt.scatter(j, indices_of_best_fit[0], color='green')
    plt.figure()
    plt.imshow(line[:, np.newaxis], cmap='bone', vmin=0, vmax=1)
    plt.show()
    """

    #if len(indices_of_best_fit)==0:
    #    continue
    for id in indices_of_best_fit:

        diffs.append(np.min(abs(line - tc)))
        Dv.append(bin_edges_X[id])
        Mv.append(bin_edges_Y[j])
        index_pairs_of_dots.append([j, id])
        #if line[id] < 10:
        markers.append('x')
        #else:
        #    markers.append('o')

    # now remove the lines that were used and redo the histogramming
    c=0
    for k in range(len(dmean_vals)):
        trajectory = dmean_vals[k]
        masses = log_masses[k]

        H = scipy.stats.binned_statistic_2d(x=trajectory, y=masses, values=None, statistic='count',
                                                     bins=[bin_edges_X, bin_edges_Y])[0]
        H[H>0] = 1
        if H[indices_of_best_fit[0], j]==1:
            dmean_vals[k] = np.nan
            c+=1

    # now get the relevant statistics
    hist_to_manipulate = scipy.stats.binned_statistic_2d(x=dmean_vals.flatten(), y=log_masses.flatten(),
                                                         values=None, statistic='count',
                                                         bins=[bin_edges_X, bin_edges_Y])[0]

"""
# Plot the fit vs the data points
plt.figure(figsize=(6, 3))
for dv, mv, mar in zip(Dv, Mv, markers):
    #if mar=='D':
    #    plt.scatter(mv, dv, marker=mar, color='darkred', edgecolors='k', s=25)
    #else:
    plt.scatter(mv, dv, marker=mar, color='orange', edgecolors='k', s=50)
plt.plot(bin_edges_Y, [fit(be) for be in bin_edges_Y],
         color='orange', linestyle='-', linewidth=5, alpha=0.8, zorder=2)
plt.xlabel(r'$\log_{10}(V)$', size=15)
plt.ylabel(r'$\bar{d}$', size=15)
plt.tight_layout()
plt.savefig('correction_src/fitT.png', dpi=300)
#plt.show()
"""


# Plot 4: x = contour_thresholds, y = proto halo masses, z = intersection counts
import matplotlib
matplotlib.rcParams.update({'font.size': 15})
xx, yy = np.meshgrid(np.arange(hist.shape[1]), np.arange(hist.shape[0]))
f, ax = plt.subplots(figsize=(7, 3.5))
#import scipy.ndimage as nd
#hist = nd.gaussian_filter(hist, sigma=2)
im = ax.imshow(hist, cmap='binary', origin='lower', alpha=0.8)
#cb.ax.plot([0, 1], [25 * 1.0/hist.max(), 25 * 1.0/hist.max()], color='green', linestyle='-', linewidth=2)
ax.contour(xx, yy, hist, levels=[10], linestyles=['dashed'], colors=['grey'])

ax.set(xticks       = np.arange(-0.0, len(bin_edges_Y))[::4],
       yticks       = np.arange(-0.0, len(bin_edges_X))[::2],
       yticklabels  = np.round(bin_edges_X[::2], 1),
       xlabel       = r'$\log_{10}(V[\mathrm{cells}])$',
       ylabel       = r'$\bar{d}$')
ax.set_xticklabels( np.round(bin_edges_Y[::3], 1), rotation = 0, ha="center")

train = ax.scatter(np.asarray(index_pairs_of_dots)[:, 0], np.asarray(index_pairs_of_dots)[:, 1],
           marker='o', color='cornflowerblue', edgecolors='none', s=50, zorder=3, label='training')

# include the validation scatters to show that fit is fine
points_val = np.load(homedir+'hmfA/correction_src/scatter_points_val.npy')
val = ax.scatter(np.asarray(points_val)[:, 0], np.asarray(points_val)[:, 1],
           marker='o', facecolor='none',#'#64DD17',
                 edgecolors='darkorange',
                 linewidth=2,
                 s=80, zorder=3, label='validation')
plt.legend((train, val),
           ('training', 'validation'),
           scatterpoints=1,
           loc='upper left',
           frameon=False)
cb = colorbar(mappable=im, colorbar_label='Occurences')
# transform the values from mass fit to the imshow indices [0, 1, ...], really painful.
trans_X = len(bin_edges_X) * ([fit(y) for y in bin_edges_Y] - bin_edges_X.min()) / abs(bin_edges_X.min() - bin_edges_X.max())
trans_Y = len(bin_edges_Y) * (bin_edges_Y - bin_edges_Y.min()) / abs(bin_edges_Y[0] - bin_edges_Y[-1])
ax.plot(trans_Y, trans_X, color='darkred', linestyle='-', linewidth=3, alpha=0.85, zorder=10)   # 1565C0
ax.plot(trans_Y, trans_X, color='white', linestyle='-', linewidth=4, alpha=1.0, zorder=9)


raw_masses = np.load('correction_src/raw_masses_matrix.npy')
log_masses = np.log10(raw_masses)
dmean_vals = np.load('correction_src/dmean_matrix.npy')
for id in [0, 10]:
    print(id)
    lm = log_masses[id]
    dm = dmean_vals[id]
    lm[lm == -np.inf] = 0
    dm[dm == -np.inf] = 0
    tx = len(bin_edges_Y) * (lm - bin_edges_Y.min()) / abs(bin_edges_Y.min() - bin_edges_Y.max())
    ty = len(bin_edges_X) * (dm - bin_edges_X.min()) / abs(bin_edges_X.min() - bin_edges_X.max())
    ax.plot(tx, ty, linewidth=2, color='k', alpha=0.75)
ax.set_xlim(trans_Y.min()-0.5, trans_Y.max()+0.5)
ax.set_ylim(0.5, trans_X.max()+3.75)
f.tight_layout(rect=[0, 0, 1, 1])
plt.savefig('correction_src/fitT_and_A.png', dpi=300)
plt.show()
