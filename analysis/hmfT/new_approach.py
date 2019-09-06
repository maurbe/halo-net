from __future__ import division
import numpy as np, os, pandas as pd
import matplotlib.pyplot as plt
from analysis.helpers.plotting_help import *
import scipy
import scipy.ndimage as nd
from matplotlib.colors import LogNorm
from analysis.helpers.hmf_functions_revised import find_peak_to_thresh_relation



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

where_are_NaNs = np.isnan(peak_vals)
peak_vals[where_are_NaNs] = 0
"""
plt.figure()
for trajectory, rm in zip(peak_vals, raw_masses):
    plt.semilogy(trajectory, rm, linewidth=0.5)

plt.figure()
print(raw_masses.shape, peak_vals.shape)
"""
log_masses = np.log10(raw_masses)


def colorbar(mappable, colorbar_label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    return fig.colorbar(mappable, cax=cax, label=colorbar_label)


#P = np.asarray([p for sub in peak_vals for p in sub])
X = np.asarray([x for sub in peak_vals for x in sub])
Y = np.asarray([y for sub in log_masses for y in sub])
bin_edges_X = np.linspace(0, peak_vals.max(), peak_vals.shape[1], endpoint=True)


projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data       = pd.read_csv(projectdir + 'source/catalog'+sim+'/catalog'+sim+'.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
print('gt no. of halos', len(halo_sizes))
halo_sizes = np.log10(halo_sizes)
bins_hmf   = 50
true_counts, bin_edges_Y = plt.hist(halo_sizes, bins=bins_hmf)[0:2]


# now get the relevant statistics
hist, bin_edges_X, bin_edges_Y, binnumbers = \
    scipy.stats.binned_statistic_2d(x=X, y=Y, values=None, statistic='count', bins=[bin_edges_X, bin_edges_Y], expand_binnumbers=True)


# prepare for for loop, reverse order...
# smoothing here (before the for loop below) is totally forbidden, since it would change the values inside
hist        = np.rot90(hist)
hist = nd.gaussian_filter(hist, sigma=1)
print(np.asarray([max(x) for x in hist]).sum(), true_counts.sum())   # all good!


# for loop to determine the index/contour_thresh so that the number of intersections in this given 2D bin matches best
# the number of halos in the ground truth halo mass histogram (normal one, not the accumulated one)
index_pairs_of_dots = []
Pv = []
Mv = []
diffs = []
markers = []

print(true_counts[::-1])

for j, (line, tc) in enumerate(zip(hist, true_counts[::-1])):
    # in theory, we should remove the data that has been assigned to a bin!
    # leaving it in is probably a good approximation

    index_of_best_fit = np.argmin(abs(line - tc) )

    diffs.append(np.min(abs(line -tc)))
    Pv.append(peak_vals[j, index_of_best_fit])
    Mv.append(bin_edges_Y[::-1][j])
    index_pairs_of_dots.append([index_of_best_fit, j])
    if hist[j, index_of_best_fit] < 10:
        markers.append('d')
    else:
        markers.append('o')

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
im = ax.imshow(hist, cmap='bone')
cb = colorbar(mappable=im, colorbar_label='Occurences')
cb.ax.plot([0, 1], [25 * 1.0/hist.max(), 25 * 1.0/hist.max()], color='darkred', linestyle='-', linewidth=2)
ax.contour(xx, yy, hist, levels=[25], colors=['darkred'])
ax.set(xticks = np.arange(0, len(bin_edges_X[:-1]))[::3], xticklabels = np.round(bin_edges_X[:-1][::3], 2),
       yticks = np.arange(0, len(bin_edges_Y[::-1][:-1]))[::2], yticklabels = np.round(bin_edges_Y[::-1][:-1][::2] - np.log10(64.0/5.66e10), 1), # second term: conversion from number of particles to real mass
       #xlim = ((0, len(bin_edges_X[:-20]))),
       xlabel = 'mean dist', ylabel = r'$\log_{10}(\mathrm{M/M}_{\odot})$')

for points, mar in zip(index_pairs_of_dots, markers):
    if mar=='d':
        ax.scatter(points[0], points[1], marker=mar, color='purple', edgecolors='k', zorder=3)
    else:
        ax.scatter(points[0], points[1], marker=mar, color='cornflowerblue', edgecolors='k', zorder=3)

def fit_M(x):
    m = []
    X = 3.75
    x0 = 3.9
    x1 = 4.95
    x2 = 5.65

    Y = 1.3
    y0 = 1.8
    y1 = 3.5
    y2 = 5.6

    for p in x:
        if p < x0:
            s = (y0 - Y) / (x0 - X)
            q = y0 - s * x0
            m.append(s * p + q)
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
trans_X = len(bin_edges_X[:-1]) * (fit_M(bin_edges_Y[::-1]) - bin_edges_X.min()) / abs(bin_edges_X.min() - bin_edges_X.max())
trans_Y = len(bin_edges_Y[:-1]) * (bin_edges_Y - bin_edges_Y.min()) / abs(bin_edges_Y[0] - bin_edges_Y[-1])
ax.plot(trans_X, trans_Y, color='orange', linestyle='-', linewidth=5, alpha=0.7, zorder=2)
plt.show()