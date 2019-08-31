"""
    Triple plot of input, target and prediction.
    Use the 8x285 case here, since we are primarily interested in those halos...
    Do not (!) "correct" for the mean, nor do a smoothing!
"""

import numpy as np, os
import scipy.ndimage as nd
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from analysis.helpers.plotting_help import *

from matplotlib.colors import LinearSegmentedColormap
mcmap = LinearSegmentedColormap.from_list('mycmap', ['#3F1F47', '#5C3C9A', '#6067B3',
                                                     #   '#969CCA',
                                                     '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                     '#601A49', '#8C2E50', '#A14250',
                                                     '#B86759',
                                                     '#E0D9E1'][::-1])
mcmap='twilight_r'
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

# Merger situtuation (?): slice 300 middle left
# I like: 350, 331

# GOOD example: 375
#n = 150
#xl, xr = 10, 220
#yb, yt = 170, 300
#key = 'good'

# unwanted MERGER example:
#n = 50
#xl, xr = 200, 410
#yb, yt = 382, 512
#key = 'merger'

# unwanted SEPARATION example:
n = 300
xl, xr = 10, 220
yb, yt = 360, 490
key = 'separator'


input = input[n]
gt = gt[n]
pred = pred[n]


# first row
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 8))

Vmaxp = np.max(input)
Vminp = np.min(input)
Vmaxd = 1.0

im1 = ax1.imshow(input, cmap='magma', vmin=Vminp, vmax=Vmaxp)
cb1 = colorbar(im1, r'$\delta$')
cb1.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cb1.outline.set_linewidth(1.5)
ax1.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
for axis in ['top','bottom','left','right']:
  ax1.spines[axis].set_linewidth(1.5)

im2 = ax2.imshow(gt, cmap=mcmap, vmin=0.0, vmax=Vmaxd)
cb2 = colorbar(im2, r'True Distance')
cb2.set_ticks([0, 0.5, 1.0])
cb2.outline.set_linewidth(1.5)
ax2.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
for axis in ['top','bottom','left','right']:
  ax2.spines[axis].set_linewidth(1.5)

im3 = ax3.imshow(pred, cmap=mcmap, vmin=0.0, vmax=Vmaxd)
cb3 = colorbar(im3, 'Predicted Distance')
cb3.set_ticks([0, 0.5, 1.0])
cb3.outline.set_linewidth(1.5)
ax3.set(xticks=[0, 256, 512], yticks=[0, 256, 512], xlabel='x [cells]', ylabel='y [cells]')
for axis in ['top','bottom','left','right']:
    ax3.spines[axis].set_linewidth(1.5)


# second row
for ax, field, cm, vMax, vMin in zip([ax1, ax2, ax3], [input, gt, pred], ['magma', mcmap, mcmap],
                               [Vmaxp, Vmaxd, Vmaxd],
                               [Vminp, 0.0, 0.0]):
    axins = zoomed_inset_axes(parent_axes=ax, zoom=2.5, loc='lower center',
                              bbox_to_anchor=(0, -0.92, 1, 1),
                              bbox_transform=ax.transAxes)
    axins.imshow(field, cmap=cm, vmin=vMin, vmax=vMax)
    axins.set(xticks = [], yticks = [], xlim = (xl, xr), ylim = (yt, yb))
    for axis in ['top','bottom','left','right']:
      axins.spines[axis].set_linewidth(1.5)
      axins.spines[axis].set_color('#546E7A')

    axins.tick_params(axis=u'both', which=u'both',length=0)
    _patch, pp1, pp2 = mark_inset(ax, axins, 3, 4, fc="none", lw=2.0, ec='#546E7A', zorder=0, linestyle=':')
    pp1.loc1, pp1.loc2 = 1, 4  # inset corner 1 to origin corner 4 (would expect 1)
    pp2.loc1, pp2.loc2 = 3, 2  # inset corner 3 to origin corner 2 (would expect 3)

    rect = patches.Rectangle((xl, yb), xr-xl, yt-yb,
                             linewidth=1.5, edgecolor='#546E7A', facecolor='none')
    ax.add_patch(rect)
plt.savefig('triple_'+key+'.png', dpi=300)
plt.show()
