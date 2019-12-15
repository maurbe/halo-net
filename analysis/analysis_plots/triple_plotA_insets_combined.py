"""
    Purpose:    Triple plot of input, target and prediction.
    Comment:    Do not (!) correct for the mean, nor do a smoothing!
"""

import os
import scipy.ndimage as nd
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from analysis.helpers.plotting_help import *
font = {'size'   : 12}
matplotlib.rc('font', **font)

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

# ------------------------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
viridis = cm.get_cmap('magma')
values = [viridis(x) for x in np.linspace(0, 1, 100)]
#values[0] = (1, 1, 1, 1)    # set the first value to white
yellow = values[-1]
for x in range(50):
    values.append(yellow)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('mycmap', values)
# ------------------------------------------------------------------------------------------
mcmap = cm

homedir = os.path.dirname(os.getcwd()) + '/'
input = np.load(homedir + 'boxesA/input_density_contrastA.npy')
gt = np.load(homedir + 'boxesA/gt_distancemap_normA.npy')
pred = np.load(homedir + 'boxesA/predictionA.npy')


n = 267
input = input[n]
gt = gt[n]
pred = pred[n]


# first row
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=False, sharey=True, figsize=(11, 8))

im1 = ax1.imshow(input, cmap='magma', clim=(input.min(), input.max()))
cb1 = colorbar(im1, r'$\delta_{\mathrm{uvs}}$')
cb1.set_ticks([-3, -2, -1, 0, 1, 2, 3])
cb1.outline.set_linewidth(1.5)
ax1.set(xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512], ylabel='y [cells]')
for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)

im2 = ax2.imshow(nd.gaussian_filter(gt, sigma=1), cmap=mcmap, clim=(0, 1.0))
cb2 = colorbar(im2, r'True Distance')
cb2.set_ticks([0, 0.5, 1.0])
cb2.outline.set_linewidth(1.5)
ax2.set(xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512])
for axis in ['top', 'bottom', 'left', 'right']:
    ax2.spines[axis].set_linewidth(1.5)

im3 = ax3.imshow(nd.gaussian_filter(pred, sigma=1), cmap=mcmap, clim=(0, 1.0))
cb3 = colorbar(im3, 'Predicted Distance')
cb3.set_ticks([0, 0.5, 1.0])
cb3.outline.set_linewidth(1.5)
ax3.set(xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512])
for axis in ['top', 'bottom', 'left', 'right']:
    ax3.spines[axis].set_linewidth(1.5)


# second rows
zoom = 2

xl_11, xr_11, yt_11, yb_11 = 170, 270, 105, 205
for ax, bbox, field, cmap in zip([ax1, ax2, ax3],
                                 [(0, -0.5, 0.5, 0.5), (0, -0.5, 0.5, 0.5), (0, -0.5, 0.5, 0.5)],
                                 [input, gt, pred],
                                 ['magma', mcmap, mcmap]):
    axins_11 = zoomed_inset_axes(parent_axes=ax, zoom=zoom, loc='lower center',
                                 bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins_11.imshow(nd.gaussian_filter(field, sigma=1), cmap=cmap, clim=(field.min(), field.max()))
    axins_11.set(xticks=[], yticks=[], xlim=(xl_11, xr_11), ylim=(yb_11, yt_11))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins_11.spines[axis].set_linewidth(1.5)
        axins_11.spines[axis].set_color('k')
    axins_11.tick_params(axis='both', which='both', length=0)
    ax.text(xl_11 + 5, yt_11 + 5, 'A', fontsize=12, horizontalalignment='left', verticalalignment='top', color='white')
    axins_11.text(0.05, 0.95, 'A', fontsize=14, horizontalalignment='left', verticalalignment='top', transform=axins_11.transAxes, color='white')
    rect = patches.Rectangle((xl_11, yt_11), xr_11-xl_11, yb_11-yt_11, linewidth=1.5, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
"""
xl_12, xr_12, yt_12, yb_12 = 80, 180, 350, 450
for ax, bbox, field, cmap in zip([ax1, ax2, ax3],
                                 [(0, -0.9, 0.5, 0.5), (0, -0.9, 0.5, 0.5), (0, -0.9, 0.5, 0.5)],
                                 [input, gt, pred],
                                 ['magma', mcmap, mcmap]):
    axins_12 = zoomed_inset_axes(parent_axes=ax, zoom=zoom, loc='lower center',
                                 bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins_12.imshow(field, cmap=cmap, clim=(field.min(), field.max()))
    axins_12.set(xticks=[], yticks=[], xlim=(xl_12, xr_12), ylim=(yb_12, yt_12))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins_12.spines[axis].set_linewidth(1.5)
        axins_12.spines[axis].set_color('k')
    axins_12.tick_params(axis='both', which='both', length=0)
    rect = patches.Rectangle((xl_12, yt_12), xr_12-xl_12, yb_12-yt_12, linewidth=1.5, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
"""

xl_21, xr_21, yt_21, yb_21 = 340, 440, 60, 160
for ax, bbox, field, cmap in zip([ax1, ax2, ax3],
                                 [(0.5, -0.5, 0.5, 0.5), (0.5, -0.5, 0.5, 0.5), (0.5, -0.5, 0.5, 0.5)],
                                 [input, gt, pred],
                                 ['magma', mcmap, mcmap]):
    axins_21 = zoomed_inset_axes(parent_axes=ax, zoom=zoom, loc='lower center',
                                 bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins_21.imshow(nd.gaussian_filter(field, sigma=1), cmap=cmap, clim=(field.min(), field.max()))
    axins_21.set(xticks=[], yticks=[], xlim=(xl_21, xr_21), ylim=(yb_21, yt_21))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins_21.spines[axis].set_linewidth(1.5)
        axins_21.spines[axis].set_color('k')
    axins_21.tick_params(axis='both', which='both', length=0)
    ax.text(xl_21+5, yt_21+5, 'B', fontsize=12, horizontalalignment='left', verticalalignment='top', color='white')
    axins_21.text(0.05, 0.95, 'B', fontsize=14, horizontalalignment='left', verticalalignment='top', transform=axins_21.transAxes, color='white')
    rect = patches.Rectangle((xl_21, yt_21), xr_21-xl_21, yb_21-yt_21, linewidth=1.5, edgecolor='white', facecolor='none')
    ax.add_patch(rect)

"""
xl_22, xr_22, yt_22, yb_22 = 400, 500, 350, 450
for ax, bbox, field, cmap in zip([ax1, ax2, ax3],
                                 [(0.5, -0.9, 0.5, 0.5), (0.5, -0.9, 0.5, 0.5), (0.5, -0.9, 0.5, 0.5)],
                                 [input, gt, pred],
                                 ['magma', mcmap, mcmap]):
    axins_22 = zoomed_inset_axes(parent_axes=ax, zoom=zoom, loc='lower center',
                                 bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axins_22.imshow(field, cmap=cmap, clim=(field.min(), field.max()))
    axins_22.set(xticks=[], yticks=[], xlim=(xl_21, xr_21), ylim=(yb_22, yt_22))
    for axis in ['top', 'bottom', 'left', 'right']:
        axins_22.spines[axis].set_linewidth(1.5)
        axins_22.spines[axis].set_color('k')
    axins_22.tick_params(axis='both', which='both', length=0)
    rect = patches.Rectangle((xl_22, yt_22), xr_22-xl_22, yb_22-yt_22, linewidth=1.5, edgecolor='white', facecolor='none')
    ax.add_patch(rect)
"""

plt.tight_layout()
plt.savefig('triples/triple_combined.png', dpi=300)
plt.show()

