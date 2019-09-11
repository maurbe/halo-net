from __future__ import division
import numpy as np, os, pandas as pd
import matplotlib.pyplot as plt
from analysis.helpers.plotting_help import *
import scipy
from tqdm import trange
import scipy.ndimage as nd
from matplotlib.colors import LogNorm
from analysis.helpers.hmf_functions_revised import find_peak_to_thresh_relation

def colorbar(mappable, colorbar_label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    return fig.colorbar(mappable, cax=cax, label=colorbar_label)





sim = 'T'
"""
homedir = os.path.dirname(os.getcwd()) + '/'
predicted_distances = np.load(homedir + 'boxes'+sim+'/prediction'+sim+'.npy')
raw_masses, peak_vals, contour_fs = find_peak_to_thresh_relation(distance=predicted_distances, sim=sim, homedir=homedir)

np.save('raw_masses.npy', raw_masses)
np.save('peak_vals.npy', peak_vals)
np.save('contour_fs.npy', contour_fs)
"""
raw_masses = np.load('raw_masses.npy')
peak_vals = np.load('peak_vals.npy')

peak_vals[np.isnan(peak_vals)] = 0
log_masses = np.log10(raw_masses)
bin_edges_X = np.linspace(0, peak_vals.max(), 150,#peak_vals.shape[1]+1,
                          endpoint=True)


projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data       = pd.read_csv(projectdir + 'source/catalog'+sim+'/catalog'+sim+'.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
print('gt no. of halos', len(halo_sizes))
halo_sizes = np.log10(halo_sizes)
bins_hmf   = log_masses.shape[1] # should be same as final hmf
true_counts, bin_edges_Y = plt.hist(halo_sizes, bins=bins_hmf)[0:2]

print(log_masses.shape)
hist = scipy.stats.binned_statistic_2d(x=peak_vals.flatten(), y=log_masses.flatten(),
                                       values=None, statistic='count',
                                       bins=[bin_edges_X, bin_edges_Y])[0]
hist = nd.gaussian_filter(hist, sigma=1)
hist_to_manipulate = hist.copy()

#f, ax = plt.subplots()
#ax.imshow(hist)
#ax.set_xticks(np.arange(0, len(bin_edges_X)+1))
#ax.set_xticklabels(bin_edges_X, rotation=45, ha='right')
#plt.show()

# for loop to determine the index/contour_thresh so that the number of intersections in this given 2D bin matches best
# the number of halos in the ground truth halo mass histogram (normal one, not the accumulated one)
index_pairs_of_dots = []
Pv = []
Mv = []
diffs = []
markers = []


print(hist_to_manipulate.shape)
print(true_counts)
for j in trange(hist_to_manipulate.shape[1]):

    line = hist_to_manipulate[:, j]
    tc = true_counts[j]
    indices_of_best_fit = np.argsort(abs(line-tc))[:3] # max() to get the one more on the right...
    """
    plt.figure()
    plt.imshow(hist_to_manipulate, cmap='bone', vmin=0, vmax=1)
    plt.scatter(j, indices_of_best_fit[0], color='green')
    plt.figure()
    plt.imshow(line[:, np.newaxis], cmap='bone', vmin=0, vmax=1)
    plt.show()
    """

    #index_of_best_fit = np.argmin(abs(line - tc) )
    #indices_of_best_fit = [max(np.argsort(abs(line-tc))[:2])] # max() to get the one more on the right...


    #if len(indices_of_best_fit)==0:
    #    continue

    for id in indices_of_best_fit:

        diffs.append(np.min(abs(line - tc)))
        Pv.append(bin_edges_X[id])
        Mv.append(bin_edges_Y[j])
        index_pairs_of_dots.append([j, id])
        if line[id] < 10:
            markers.append('d')
        else:
            markers.append('o')

    # now remove the lines that were used and redo the histogramming
    #print(peak_vals.shape)
    c=0
    print(peak_vals.shape)
    for k in range(len(peak_vals)):
        trajectory = peak_vals[k]
        masses = log_masses[k]

        H = scipy.stats.binned_statistic_2d(x=trajectory, y=masses, values=None, statistic='count',
                                                     bins=[bin_edges_X, bin_edges_Y])[0]
        H[H>0] = 1

        if H[indices_of_best_fit[0], j]==1:
            peak_vals[k] = np.nan
            c+=1
    print(c)


    # now get the relevant statistics
    hist_to_manipulate = scipy.stats.binned_statistic_2d(x=peak_vals.flatten(), y=log_masses.flatten(),
                                                         values=None, statistic='count',
                                                         bins=[bin_edges_X, bin_edges_Y])[0]

plt.figure()
plt.imshow(hist, cmap='bone', interpolation='nearest', origin='lower')

plt.figure()
for trajectory, rm in zip(peak_vals, raw_masses):
    plt.semilogy(trajectory, rm, linewidth=0.5)

print(diffs)
print(Pv, markers)







plt.figure(figsize=(6, 5.75))
for pv, mv, mar in zip(Pv, Mv, markers):
    if mar=='d':
        plt.scatter(mv, pv, marker=mar, color='purple', edgecolors='k')
    else:
        plt.scatter(mv, pv, marker=mar, color='cornflowerblue', edgecolors='k')





xx, yy = np.meshgrid(np.arange(hist.shape[1]), np.arange(hist.shape[0]))


# Plot 4: x = contour_thresholds, y = proto halo masses, z = intersection counts
f, ax = plt.subplots(figsize=(7,7))
im = ax.imshow(hist, cmap='twilight_shifted', origin='lower')
cb = colorbar(mappable=im, colorbar_label='Occurences')
cb.ax.plot([0, 1], [25 * 1.0/hist.max(), 25 * 1.0/hist.max()], color='darkred', linestyle='-', linewidth=2)
ax.contour(xx, yy, hist, levels=[10], colors=['darkred'])
ax.set(xticks       = np.arange(-0.5, len(bin_edges_Y))[::3],
       xticklabels  = np.round(bin_edges_Y[::3], 3),
       yticks       = np.arange(-0.5, len(bin_edges_X))[::2],
       yticklabels  = np.round(bin_edges_X[::2], 2),
       xlabel       = r'$\log_{10}(V)$[cells]',
       ylabel       = 'mean dist')
ax.set_xticklabels( np.round(bin_edges_Y[::3], 3), rotation = 45, ha="right")

for points, mar in zip(index_pairs_of_dots, markers):
    if mar=='d':
        ax.scatter(points[0], points[1], marker=mar, color='purple', edgecolors='k', zorder=3)
    else:
        ax.scatter(points[0], points[1], marker=mar, color='cornflowerblue', edgecolors='k', zorder=3)



def fit_M(x):
    m = []
    X = 3.66
    x0 = 4.0
    x1 = 4.75
    x2 = 5.5

    Y = 2.0
    y0 = 1.0
    y1 = 3.0
    y2 = 4.5

    for p in x:
        if p < x0:
            s = (y0-Y) / (x0 - X)
            q = y0 - s * x0
            m.append(s*p+q)
        elif p < x1:
            s = (y1-y0) / (x1 - x0)
            q = y1 - s * x1
            m.append(s*p+q)
        else:
            s = (y2-y1) / (x2 - x1)
            q = y2 - s * x2
            m.append(s*p+q)
    return np.asarray(m)

# transform the values from mass fit to the imshow indices [0, 1, ...], really painful.
trans_X = len(bin_edges_Y) * (fit_M(bin_edges_Y) - bin_edges_Y.min()) / abs(bin_edges_Y.min() - bin_edges_Y.max())
trans_Y = len(bin_edges_Y) * (bin_edges_Y - bin_edges_Y.min()) / abs(bin_edges_Y[0] - bin_edges_Y[-1])
ax.plot(trans_X, trans_Y, color='orange', linestyle='-', linewidth=5, alpha=0.7, zorder=2)

plt.show()