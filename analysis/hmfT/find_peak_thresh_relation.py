from __future__ import division
import numpy as np, os, pandas as pd
import matplotlib.pyplot as plt
from analysis.helpers.plotting_help import *
import scipy
import scipy.ndimage as nd
from matplotlib.colors import LogNorm
from analysis.helpers.hmf_functions_revised import find_peak_to_thresh_relation



sim = 'T'

homedir = os.path.dirname(os.getcwd()) + '/'
predicted_distances = np.load(homedir + 'boxes'+sim+'/prediction'+sim+'.npy')
raw_masses, peak_vals, contour_fs = find_peak_to_thresh_relation(distance=predicted_distances, sim=sim, homedir=homedir)
np.save('raw_masses.npy', raw_masses)
np.save('peak_vals.npy', peak_vals)
np.save('contour_fs.npy', contour_fs)

raw_masses = np.load('raw_masses.npy')
peak_vals = np.load('peak_vals.npy')
contour_fs = np.load('contour_fs.npy')


"""
plt.figure()
for trajectory, rm in zip(peak_vals, raw_masses):
    plt.plot(rm, trajectory)
plt.show()


plt.figure()
plt.hist(peak_vals[:, 0], 100, log=True, histtype='step',
         color='b')

sim = 'A'
homedir = os.path.dirname(os.getcwd()) + '/'
predicted_distances = np.load(homedir + 'boxes'+sim+'/prediction'+sim+'.npy')
raw_masses, peak_vals, contour_fs = find_peak_to_thresh_relation(distance=predicted_distances, sim=sim, homedir=homedir)
plt.hist(peak_vals[:, 0], 100, log=True, histtype='step',
         color='g')
plt.show()
"""








print(raw_masses.shape, peak_vals.shape, contour_fs.shape)

#peak_vals  = np.load('peak_thresh_src/peak_vals.npy')
#contour_fs = np.load('peak_thresh_src/contour_fs.npy')
#raw_masses = np.load('peak_thresh_src/raw_masses.npy')
log_masses = np.log10(raw_masses)
#raw_masses = np.where(raw_masses>=1, raw_masses, 0)
#log_masses = np.where(raw_masses>=1, log_masses, 0)

#peak_vals = np.asarray([ np.mean(p) * np.ones(100) for p in peak_vals])
print(peak_vals.shape)



def colorbar(mappable, colorbar_label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.02)
    return fig.colorbar(mappable, cax=cax, label=colorbar_label)

def fit(x):
    """
    cf = []
    m = -(0.3-0.1)/80000
    q = 0.28
    for p in x:
            cf.append(m*p+q)
    return np.asarray(cf)
    """
    cf = []
    anchor1 = 2e4
    anchor2 = 4e4
    for p in x:
        if p < anchor1:
            cf.append(0.2)
        elif p <= anchor2:
            m = -(0.2-0.05) / (anchor2 - anchor1)
            q = 0.2 - m*anchor1
            cf.append(m * p + q)
        elif p >= 0.3:
            cf.append(0.05)
    return np.asarray(cf)


def fit_M(x):
    cf = []
    anchor1 = 3.5
    anchor2 = 4.25
    for p in x:
        if p < anchor1:
            cf.append(0.040)
        elif p < anchor2:
            m = (0.075-0.040) / (anchor2 - anchor1)
            q = 0.040 - m*anchor1
            cf.append(m * p + q)
        elif anchor2 >= 0.3:
            cf.append(0.075)
    return np.asarray(cf)









P = np.asarray([p for sub in peak_vals for p in sub])
X = np.asarray([x for sub in contour_fs for x in sub])
Y = np.asarray([y for sub in log_masses for y in sub])
bin_edges_X = np.linspace(0, 1, contour_fs.shape[1], endpoint=True)


projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
data       = pd.read_csv(projectdir + 'source/catalog'+sim+'/catalog'+sim+'.csv', delimiter='\t')
halo_sizes = data['nr_part'].values
halo_sizes = halo_sizes[halo_sizes>=16*285]
print('gt no. of halos', len(halo_sizes))
halo_sizes = np.log10(halo_sizes)
bins_hmf   = 50
true_counts, bin_edges_Y = plt.hist(halo_sizes, bins=bins_hmf)[0:2]

# now get the relevant statistics
peak_means = scipy.stats.binned_statistic_2d(x=X, y=Y, values=P, statistic='mean', bins=[bin_edges_X, bin_edges_Y], expand_binnumbers=True)[0]
hist, bin_edges_X, bin_edges_Y, binnumbers = \
    scipy.stats.binned_statistic_2d(x=X, y=Y, values=None, statistic='count', bins=[bin_edges_X, bin_edges_Y], expand_binnumbers=True)
#print (Y>bin_edges_Y.max()).sum(), (Y<bin_edges_Y.min()).sum()    # everything ok!


# prepare for for loop, reverse order...
# smoothing here (before the for loop below) is totally forbidden, since it would change the values inside
hist        = np.rot90(hist)
peak_means  = np.rot90(peak_means)
print(len(peak_vals))
print(np.asarray([max(x) for x in hist]).sum(), true_counts.sum())   # all good!


# for loop to determine the index/contour_thresh so that the number of intersections in this given 2D bin matches best
# the number of halos in the ground truth halo mass histogram (normal one, not the accumulated one)
index_pairs_of_dots = []
Pv = []
Cv = []
Mv = []
diffs = []
markers = []

print(true_counts[::-1])

for j, (line, tc) in enumerate(zip(hist, true_counts[::-1])):
    # in theory, we should remove the data that has been assigned to a bin!
    # leaving it in is probably a good approximation
    print(line)
    print(tc)
    index_of_best_fit = np.argmin(abs(line - tc) ) # first index with error smaller than 10%

    diffs.append(np.min(abs(line -tc)))
    Pv.append(peak_means[j, index_of_best_fit])
    Cv.append(bin_edges_X[index_of_best_fit])
    Mv.append(bin_edges_Y[::-1][j])
    index_pairs_of_dots.append([index_of_best_fit, j])
    if hist[j, index_of_best_fit] < 10:
        markers.append('d')
    else:
        markers.append('o')

print(diffs)
#raise SystemExit







# Plotting
# Plot 1: difference of true halo number and number in bin of best-fit. Ideally should be all 0.
plt.figure()
plt.plot(bin_edges_Y[:-1][::-1], diffs)
plt.xlabel(r'$\log_{10}$(n)')
plt.ylabel('$|$n$_{\mathrm{true}}$-n$_{\mathrm{best-fit}}|$')


# Plot 2a: scatter points retrieved and linear fit
plt.figure(figsize=(6, 2.75))
for pv, cv, mar in zip(Pv, Cv, markers):
    if mar=='d':
        plt.scatter(np.log10(pv), cv, marker=mar, color='purple', edgecolors='k')
    else:
        plt.scatter(np.log10(pv), cv, marker=mar, color='cornflowerblue', edgecolors='k')

if False:
    MvA = np.load('peak_thresh_src/MvA.npy')
    PvA = np.load('peak_thresh_src/PvA.npy')
    CvA = np.load('peak_thresh_src/CvA.npy')
    markersA = np.load('peak_thresh_src/markersA.npy')
    if len(PvA) != bins_hmf:
        print('Bins in Y must be same between T and A!')
        raise ImportError
    for pv, cv, mar in zip(PvA, CvA, markersA):
        if mar == 'd':
            plt.scatter(np.log10(pv), cv, marker=mar, color='red', edgecolors='k')
        else:
            plt.scatter(np.log10(pv), cv, marker=mar, color='darkorange', edgecolors='k')

#x = np.linspace(0.0, 2e5, 100)
#print(x)
#plt.plot(x, fit(x), zorder=0)#, label='linear fit')
plt.xlabel('Peak value')
plt.ylabel('Contour threshold')
leg = plt.legend(loc='upper left', prop={'size': 15}, frameon=True, framealpha=1.0)
leg.get_frame().set_linewidth(0.0)
plt.tight_layout()
plt.savefig('linear_fit.png', dpi=250)

# Plot 2b: scatter points retrieved and linear fit wrt proto halo mass!
plt.figure()
for mv, cv, mar in zip(Mv, Cv, markers):
    if mar=='d':
        plt.scatter(mv - np.log10(64.0/5.66e10), cv, marker=mar, color='purple', edgecolors='k')
    else:
        plt.scatter(mv - np.log10(64.0/5.66e10), cv, marker=mar, color='cornflowerblue', edgecolors='k')

if False:
    for mv, cv, mar in zip(MvA, CvA, markersA):
        if mar == 'd':
            plt.scatter(mv - np.log10(64.0/5.66e10), cv, marker=mar, color='red', edgecolors='k')
        else:
            plt.scatter(mv - np.log10(64.0/5.66e10), cv, marker=mar, color='darkorange', edgecolors='k')

x = np.linspace(3.0, max(Mv), 100)
plt.plot(x- np.log10(64.0/5.66e10), fit_M(x), zorder=0, label='linear fit')
plt.xlabel(r'$\log_{10}(\mathrm{M/M}_{\odot})$')
plt.ylabel('Contour threshold')
leg = plt.legend(loc='upper left', prop={'size': 15}, frameon=True, framealpha=1.0)
leg.get_frame().set_linewidth(0.0)
plt.tight_layout()


# Plot 3: x = contour_thresholds, y = proto halo masses, z = peak means
f, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(peak_means, zorder=0, cmap='cubehelix')
cb = colorbar(mappable=im, colorbar_label='Mean Peak value')

voids_x, voids_y = np.nonzero(hist == 0)[0], np.nonzero(hist == 0)[1]
for vx, vy in zip(voids_x, voids_y):
    ax.scatter(vy, vx, s=8, linewidth=0.5, c='grey', marker='x')

xx, yy = np.meshgrid(np.arange(hist.shape[1]), np.arange(hist.shape[0]))
ax.contour(xx, yy, hist, levels=[25], colors=['darkorange'], linestyles='-', zorder=1)

# transform the values from mass fit to the imshow indices [0, 1, ...], really painful.
trans_X = len(bin_edges_X[:-1]) * (fit_M(bin_edges_Y[::-1]) - bin_edges_X.min()) / abs(bin_edges_X.min() - bin_edges_X.max())
trans_Y = len(bin_edges_Y[:-1]) * (bin_edges_Y - bin_edges_Y.min()) / abs(bin_edges_Y[0] - bin_edges_Y[-1])
ax.plot(trans_X, trans_Y, color='blue', linestyle='-', linewidth=3, alpha=0.7, zorder=2)
"""
for chosen, colo in zip([4467, 267, 13, 27], ['#D8B75E', '#D8B75E', '#D8B75E', '#D8B75E']):
    lmass = np.log10(raw_masses[chosen]) #- np.log10(64.0 / 5.66e10) ?? no idea why...
    cf = contour_fs[chosen]
    ax.plot((len(bin_edges_X[:-1]) * (cf - bin_edges_X.min()) / abs(bin_edges_X.min() - bin_edges_X.max()))[::-1],
            (len(bin_edges_Y[:-1]) * (bin_edges_Y.max()-lmass) / abs(bin_edges_Y[0] - bin_edges_Y[-1]))[::-1],
            linewidth=1.0, zorder=1, color=colo)
"""
ax.set(xticks = np.arange(0, len(bin_edges_X[:-1]))[::10], xticklabels = np.round(bin_edges_X[:-1][::10], 2),
       yticks = np.arange(0, len(bin_edges_Y[::-1][:-1]))[::5], yticklabels = np.round(bin_edges_Y[::-1][:-1][::5] - np.log10(64.0/5.66e10), 1), # second term: conversion from number of particles to real mass
       #xlim = (-0.5, len(bin_edges_X[:-1])-0.5), ylim = (len(bin_edges_Y[:-1])-0.5, -0.5),  # -0.5 in terms of pixel dimensions
       xlabel = 'Contour threshold', ylabel = r'$\log_{10}(\mathrm{M/M}_{\odot})$')
for points, mar in zip(index_pairs_of_dots, markers):
    if mar=='d':
        ax.scatter(points[0], points[1], marker=mar, color='purple', edgecolors='k', zorder=3)
    else:
        ax.scatter(points[0], points[1], marker=mar, color='cornflowerblue', edgecolors='k', zorder=3)
#ax.set_xlim(0, 75)
plt.savefig('hist2D.png', dpi=250)


# Plot 4: x = contour_thresholds, y = proto halo masses, z = intersection counts
f, ax = plt.subplots()
im = ax.imshow(hist, cmap='bone')
cb = colorbar(mappable=im, colorbar_label='Occurences')
cb.ax.plot([0, 1], [25 * 1.0/hist.max(), 25 * 1.0/hist.max()], color='darkred', linestyle='-', linewidth=2)
ax.contour(xx, yy, hist, levels=[25], colors=['darkred'])
ax.set(xticks = np.arange(0, len(bin_edges_X[:-1]))[::10], xticklabels = np.round(bin_edges_X[:-1][::10], 2),
       yticks = np.arange(0, len(bin_edges_Y[::-1][:-1]))[::5], yticklabels = np.round(bin_edges_Y[::-1][:-1][::5] - np.log10(64.0/5.66e10), 1), # second term: conversion from number of particles to real mass
       #xlim = ((0, len(bin_edges_X[:-20]))),
       xlabel = 'Contour threshold', ylabel = r'$\log_{10}(\mathrm{M/M}_{\odot})$')


plt.figure()
for i, (mass, cf) in enumerate(zip(raw_masses, contour_fs)):
    plt.plot(cf, np.log10(mass) - np.log10(64.0 / 5.66e10), linewidth=0.3)
plt.xlabel('Contour threshold')
plt.ylabel(r'$\log_{10}($M/M$_{\odot})$')
plt.show()
