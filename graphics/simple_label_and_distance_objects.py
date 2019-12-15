import numpy as np, os
import scipy.ndimage as nd
from analysis.helpers.plotting_help import *
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as colors
mc = colors.ListedColormap(['#E1D8E1', '#78909C', '#455A64', '#424242', '#616161', '#757575'])
plt.style.use('mystyle.mplstyle')

import matplotlib.cm as cm
viridis = cm.get_cmap('magma')
values = [viridis(x) for x in np.linspace(0, 1, 100)]
#values[0] = (1, 1, 1, 1)    # set the first value to white
last = values[-1]
for x in range(50):
    values.append(last)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('mycmap', values)

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

homedir = os.path.dirname(os.getcwd()) + '/'
labels = np.load(homedir + 'source/cic_fieldsA/cic_yA.npy')[267, 70:150, 340:440]
mass_mask = (labels < 16*285)
labels[mass_mask] = 0
distance= np.load(homedir + 'source/cic_fieldsA/distancemap_normA.npy')[267, 70:150, 340:440]
distance[mass_mask] = 0.0

un = np.unique(labels)
labels[labels==un[1]] = 5
labels[labels==un[2]] = 2
labels[labels==un[3]] = 3
labels[labels==un[4]] = 4
labels[labels==un[5]] = 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
im1 = ax1.imshow(labels, cmap=mc, extent=(340, 440, 150, 70))
ax1.set(xlabel='x [cells]', ylabel='y [cells]')#, xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512])

im2 = ax2.imshow(distance, cmap=cm, extent=(340, 440, 150, 70), clim=(0, 1))
cb2 = colorbar(im2, r'$\hat{d}$')
cb2.outline.set_linewidth(1.5)

ax1.set(xlabel='x [cells]', ylabel='y [cells]')
ax2.set(xlabel='x [cells]')

plt.tight_layout()
plt.savefig('double_figure.png', dpi=250)
plt.show()
