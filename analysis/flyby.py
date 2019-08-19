"""
    Visualization of slices as seen when "flying" through the data volume.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation
from analysis.helpers.plotting_help import *

from matplotlib.colors import LinearSegmentedColormap
mcmap = LinearSegmentedColormap.from_list('mycmap', ['#969CCA',
                                                     '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                     '#601A49', '#8C2E50', '#A14250',
                                                     '#B86759',
                                                     '#E0D9E1'][::-1])

def fly(sim):
    """
    Main function to call
    :param sim:     simulation/box name
    :return:        save as *.mp4
    """
    final_box = np.load('boxes' + sim + '/prediction' + sim + '.npy')
    input_box = np.load('boxes' + sim + '/input' + sim + '.npy')
    true_box  = np.load('boxes' + sim + '/gt' + sim + '.npy')

    print(input_box.shape)
    print(true_box.shape)
    print(final_box.shape)

    # should smooth the final_box to get rid of edge effects...
    final_box = gaussian_filter(final_box, sigma=2, mode='wrap')

    for box, name, cmp in zip([input_box, true_box, final_box],
                              ['input_density contrast' + sim, 'ground_truth' + sim, 'prediction' + sim],
                              ['magma', mcmap, mcmap]):
        fig = plt.figure()
        plt.axis('off')
        ax_epoch = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        img = plt.imshow(box[0], animated=True, cmap=cmp, vmin=0.0, vmax=1.0)
        title = ax_epoch.text(0.0, 1.01, "", color='k',
                              transform=ax_epoch.transAxes, ha="left")

        def updatefig(i):
            img.set_array(box[i])
            title.set_text('Slice {}'.format(i+1))
            return img, title,

        ani = animation.FuncAnimation(fig, updatefig, frames=box.shape[0],
                                      interval=1, blit=False)
        ani.save('movies' + sim + '/{}.mp4'.format(name), fps=15, dpi=200, writer='ffmpeg')

fly(sim='T')
fly(sim='A')
