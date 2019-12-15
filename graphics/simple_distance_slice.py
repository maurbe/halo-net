import numpy as np, os
import scipy.ndimage as nd
from analysis.helpers.plotting_help import *

def colorbar(mappable, colorbar_label):
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax = mappable.axes
    fig = ax.figure
    cax = inset_axes(ax,
                   width="85%",  # width = % of parent_bbox width
                   height="3%",  # height : %
                   loc='upper center',
                   bbox_to_anchor=(0.0, 0.0, 1.0, 1.04),
                   bbox_transform=ax.transAxes,
                   borderpad=0)
    cb = fig.colorbar(mappable, cax=cax, label=colorbar_label, orientation='horizontal')
    cb.ax.xaxis.set_label_position('top')
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.tick_params(labelsize=11)
    return cb

homedir = os.path.dirname(os.getcwd()) + '/'
distance= np.load(homedir + 'source/cic_fieldsT/distancemap_normT.npy')
cic_y = np.load(homedir + 'source/cic_fieldsT/cic_yT.npy')
# now the filtering of the distance map (directly)
mass_mask = (cic_y < 16*285)
distance[mass_mask] = 0.0

fig, ax = plt.subplots(figsize=(4, 4))
im1 = ax.imshow(distance[50], cmap='twilight_r')#, vmin=0.0, vmax=density.max())
ax.set(xlim=(64, 200),
       ylim=(490, 280),
       xlabel='x [cells]', ylabel='y [cells]')#, xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512])

cb1 = colorbar(im1, r'Normalized euclidean distance')
cb1.outline.set_linewidth(1.5)
plt.savefig('distance_slice.png', dpi=300)
plt.show()