import numpy as np, os
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
    cb.ax.tick_params(labelsize=10)
    return cb

def tiles_plot(sim):
    """
    Simple code to produce an example of two subboxes in the overlapping strategy.
    :param sim:     simulation name
    :return:        save as *.png
    """
    homedir = os.path.dirname(os.getcwd()) + '/'
    input_density_contrast = np.load(homedir + 'boxes' + sim + '/input_density_contrast' + sim + '.npy')

    f, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 256+1, 64), minor=False)
    ax.set_xticks(np.arange(0, 448+1, 64), minor=False)

    im = ax.imshow(input_density_contrast[:256+1, :448+1, 0], cmap='magma')#, vmin=-3, vmax=3)
    cb = colorbar(mappable=im, colorbar_label=r'$\delta_{uvs}$')
    cb.set_ticks([-3, -2, -1, 0, 1, 2, 3, 4])
    cb.outline.set_linewidth(1.5)

    ax.add_patch(matplotlib.patches.Rectangle((32+32, 32+32), 64, 64, linewidth=1.0, linestyle='--', edgecolor='white', facecolor='none'))
    ax.add_patch(matplotlib.patches.Rectangle((0+32, 0+32), 128, 128, linewidth=1.2, edgecolor='white', facecolor='gray', alpha=0.7))
    ax.add_patch(matplotlib.patches.Rectangle((16+32, 16+32), 96, 96, linewidth=1.0, edgecolor='lightgreen', facecolor='none'))

    ax.add_patch(matplotlib.patches.Rectangle((32+64+32, 32+64+32), 64, 64, linewidth=1.0, linestyle='--', edgecolor='white', facecolor='none', alpha=0.7))
    ax.add_patch(matplotlib.patches.Rectangle((0+64+32, 0+64+32), 128, 128, linewidth=1.2, edgecolor='white', facecolor='gray', alpha=0.6))
    #ax.add_patch(matplotlib.patches.Rectangle((16+64, 16+64), 96, 96, linewidth=1.0, edgecolor='lightgreen', facecolor='none'))

    grid_color = 'white'
    for k in [64, 64+64, 64+128]:
        ax.axhline(k, linewidth=0.5, linestyle='--', color=grid_color) # horizontal lines
    for k in [64, 128, 192, 256, 320, 384, 448]:
        ax.axvline(k,  linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    plt.tight_layout()
    plt.savefig('tiles.png', dpi=200)
    plt.show()

tiles_plot(sim='T')
