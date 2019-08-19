import numpy as np
from analysis.helpers.plotting_help import *

def tiles_plot(sim):
    """
    Simple code to produce an example of two subboxes in the overlapping strategy.
    :param sim:     simulation name
    :return:        save as *.png
    """
    cic_density = np.load('boxes' + sim + '/input' + sim + '.npy')

    f, ax = plt.subplots()
    ax.set_yticks(np.arange(0, 256+1, 64), minor=False)
    ax.set_xticks(np.arange(0, 448+1, 64), minor=False)

    ax.imshow(cic_density[:256+1, :448+1, 0], cmap='magma', vmin=-3, vmax=3)

    ax.add_patch(matplotlib.patches.Rectangle((32+32, 32+32), 64, 64, linewidth=1.0, linestyle='--', edgecolor='white', facecolor='none'))
    ax.add_patch(matplotlib.patches.Rectangle((0+32, 0+32), 128, 128, linewidth=1.2, edgecolor='white', facecolor='gray', alpha=0.7))
    ax.add_patch(matplotlib.patches.Rectangle((16+32, 16+32), 96, 96, linewidth=1.0, edgecolor='lightgreen', facecolor='none'))

    ax.add_patch(matplotlib.patches.Rectangle((32+64+32, 32+64+32), 64, 64, linewidth=1.0, linestyle='--', edgecolor='white', facecolor='none', alpha=0.7))
    ax.add_patch(matplotlib.patches.Rectangle((0+64+32, 0+64+32), 128, 128, linewidth=1.2, edgecolor='white', facecolor='gray', alpha=0.6))
    #ax.add_patch(matplotlib.patches.Rectangle((16+64, 16+64), 96, 96, linewidth=1.0, edgecolor='lightgreen', facecolor='none'))

    grid_color = 'white'
    ax.axhline(64, linewidth=0.5, linestyle='--', color=grid_color) # horizontal lines
    ax.axhline(64+64, linewidth=0.5, linestyle='--', color=grid_color) # horizontal lines
    ax.axhline(64+128, linewidth=0.5, linestyle='--', color=grid_color) # horizontal lines

    ax.axvline(64, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(128, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(192, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(256, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(320, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(384, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    ax.axvline(448, linewidth=0.5, linestyle='--', color=grid_color) # vertical lines
    plt.tight_layout()
    plt.savefig('tiles.png', dpi=200)
    plt.show()

tiles_plot(sim='T')
