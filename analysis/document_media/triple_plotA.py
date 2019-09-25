"""
    Purpose:    Triple plot of input, target and prediction without any inset.
    Comment:    Do not (!) correct for the mean, nor do a smoothing!
"""

import os
from analysis.helpers.plotting_help import *

def colorbar(mappable, colorbar_label):
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax = mappable.axes
    fig = ax.figure
    cax = inset_axes(ax,
                   width="85%",  # width = % of parent_bbox width
                   height="4%",  # height : %
                   loc='upper center',
                   bbox_to_anchor=(0.0, 0.0, 1.0, 1.07),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
    cb = fig.colorbar(mappable, cax=cax, label=colorbar_label, orientation='horizontal')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=9)
    return cb

homedir = os.path.dirname(os.getcwd()) + '/'
input = np.load(homedir + 'boxesA/input_density_contrastA.npy')
gt = np.load(homedir + 'boxesA/gt_distancemap_normA.npy')
pred = np.load(homedir + 'boxesA/predictionA.npy')

import scipy.ndimage as nd
pred = nd.gaussian_filter(pred, sigma=2, mode='wrap')

# Merger situation (?): slice 300 middle left
# 250 has some intersting ones as well
n = 375  # I like: 350, 331,
input = input[...,n]
gt = gt[...,n]
pred = pred[...,n]

xl, xr = 250, 400
yb, yt = 320, 460


f, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 8))
for ax, title, field, cmap, ticks in zip(axes,
                                         ['Density Contrast', 'Ground Truth', 'Prediction'],
                                         [input, gt, pred], ['magma', 'twilight_r', 'twilight_r'],
                                         [[-3, -2, -1, 0, 1, 2, 3], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]):
    im = ax.imshow(field, cmap=cmap, vmin=ticks[0], vmax=ticks[-1])
    cb = colorbar(im, title)
    cb.set_ticks(ticks)
    cb.outline.set_linewidth(1.5)
    ax.set(xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512],
           xlabel='x [cells]', ylabel='y [cells]')

plt.tight_layout()
plt.show()
