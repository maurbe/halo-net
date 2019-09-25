"""
    Purpose:    Triple plot of input, target and prediction.
    Comment:    Do not (!) correct for the mean, nor do a smoothing!
"""

import os
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
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

def inset_plot(IN, GT, PRED, n, xl, xr, yb, yt, key):

    input = IN[n]
    gt = GT[n]
    pred = PRED[n]

    # first row
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 8))

    im1 = ax1.imshow(input, cmap='magma', clim=(input.min(), input.max()))
    cb1 = colorbar(im1, r'$\delta$')
    cb1.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cb1.outline.set_linewidth(1.5)
    ax1.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)

    im2 = ax2.imshow(gt, cmap='twilight_r', clim=(0, 1))
    cb2 = colorbar(im2, r'True Distance')
    cb2.set_ticks([0, 0.5, 1.0])
    cb2.outline.set_linewidth(1.5)
    ax2.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax2.spines[axis].set_linewidth(1.5)

    im3 = ax3.imshow(pred, cmap='twilight_r', clim=(0, 1))
    cb3 = colorbar(im3, 'Predicted Distance')
    cb3.set_ticks([0, 0.5, 1.0])
    cb3.outline.set_linewidth(1.5)
    ax3.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
    for axis in ['top', 'bottom', 'left', 'right']:
        ax3.spines[axis].set_linewidth(1.5)

    # second row
    for ax, field, cm, vMax, vMin in zip([ax1, ax2, ax3], [input, gt, pred], ['magma', 'twilight_r', 'twilight_r'],
                                         [input.max(), 1, 1], [input.min(), 0, 0]):
        axins = zoomed_inset_axes(parent_axes=ax, zoom=2.5, loc='lower center',
                                  bbox_to_anchor=(0, -0.92, 1, 1),
                                  bbox_transform=ax.transAxes)
        axins.imshow(field, cmap=cm, clim=(vMin, vMax))
        axins.set(xticks=[], yticks=[], xlim=(xl, xr), ylim=(yt, yb))
        for axis in ['top', 'bottom', 'left', 'right']:
            axins.spines[axis].set_linewidth(1.5)
            axins.spines[axis].set_color('k')

        axins.tick_params(axis=u'both', which=u'both', length=0)
        _patch, pp1, pp2 = mark_inset(ax, axins, 3, 4, fc="none", lw=2.0, ec='k', zorder=0, linestyle=':')
        pp1.loc1, pp1.loc2 = 1, 4  # inset corner 1 to origin corner 4 (would expect 1)
        pp2.loc1, pp2.loc2 = 3, 2  # inset corner 3 to origin corner 2 (would expect 3)

        rect = patches.Rectangle((xl, yb), xr - xl, yt - yb, linewidth=1.5, edgecolor='k', facecolor='none')
        ax.add_patch(rect)
    plt.savefig('triples/triple_' + key + '.png', dpi=300)


homedir = os.path.dirname(os.getcwd()) + '/'
input = np.load(homedir + 'boxesA/input_density_contrastA.npy')
gt = np.load(homedir + 'boxesA/gt_distancemap_normA.npy')
pred = np.load(homedir + 'boxesA/predictionA.npy')

# GOOD example: 275, 150
params_good = {'n': 275,    #150
               'xl': 280,   #10
               'xr': 490,   #220
               'yb': 40,    #170
               'yt': 170,   #300
               'key': 'good'}

# unwanted MERGER example:
params_merg = {'n': 50,
               'xl': 200,
               'xr': 410,
               'yb': 382,
               'yt': 512,
               'key': 'merger'}

# unwanted SEPARATION example:
params_sepa = {'n': 300,
               'xl': 10,
               'xr': 220,
               'yb': 360,
               'yt': 490,
               'key': 'separator'}

for params in [params_good, params_merg, params_sepa]:
    inset_plot(input, gt, pred, **params)

