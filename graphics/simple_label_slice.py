import numpy as np, os
import scipy.ndimage as nd
from analysis.helpers.plotting_help import *
import matplotlib.cm as cm

np.random.seed(2019)

viridis = cm.get_cmap('viridis')
values = [viridis(x) for x in np.linspace(0, 1, 100)]
values[0] = (1, 1, 1, 1)    # set the first value to white
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('mycmap', values)


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
labels = np.load(homedir + 'source/cic_fieldsT/cic_yT.npy')[50]
mass_mask = (labels < 16*285)
labels[mass_mask] = 0

un = np.unique(labels)
for u in un:
    where = labels==u
    labels[where] = np.random.randint(1, 1000, (1,))
labels[mass_mask] = 0

fig, ax = plt.subplots(figsize=(8, 8))
im1 = ax.imshow(labels, cmap=cm)
ax.set(xlabel='x [cells]', ylabel='y [cells]', xticks=[0, 128, 256, 384, 512], yticks=[0, 128, 256, 384, 512])

plt.savefig('label_slice.png', dpi=300)
plt.show()