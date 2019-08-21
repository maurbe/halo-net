"""
    Code to visualize input, ground truth, prediction and difference for network predictions of the training set.
"""
import numpy as np, glob
import os, json

from unet.helpers.plotting_help import *
from unet.network.u_net import get_model
from unet.metrics.custom_metrics import selective_mse, selective_r2_score, r2_score
from unet.generator.datagen import custom_DataGenerator

from matplotlib.colors import LinearSegmentedColormap
mcmap = LinearSegmentedColormap.from_list('mycmap', ['#3F1F47', '#5C3C9A', '#6067B3',
                                                     #   '#969CCA',
                                                     '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                     '#601A49', '#8C2E50', '#A14250',
                                                     '#B86759',
                                                     '#E0D9E1'][::-1])

projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'

input_shape = (128, 128, 128, 1)
test_ids  = np.arange(150, 170, 3)
datagen_params_test = {'dim': input_shape[0],
                       'mode': 'full',
                       'datapath': projectdir + 'source/setsT/trainingT/',
                       'subsample_size': len(test_ids),
                       'batch_size': 1,
                       'n_channels': input_shape[-1],
                       'shuffle': False,
                       'randomize': False}
testing_generator = custom_DataGenerator(list_IDs=test_ids, **datagen_params_test)
with open(projectdir + 'unet/run_1/hyper_param_dict.json') as json_file:
    param_dict = json.load(json_file)
net = get_model(**param_dict)
latest_weights = max(glob.glob(projectdir + 'unet/run_1/saved_networks/*.hdf5'), key=os.path.getctime)
print('Loaded latest weights:', latest_weights)
net.load_weights(latest_weights)

net.compile('adam', loss=selective_mse, metrics=['mse',
                                                 selective_r2_score,
                                                 r2_score])
#eval = net.evaluate_generator(generator=testing_generator, verbose=True)
#print '\n', eval, '\n'


Prediction = net.predict_generator(generator=testing_generator, verbose=True)

for ti, prediction in zip(test_ids, Prediction):
    x = np.load(projectdir + 'source/setsT/trainingT/x_{}.npy'.format(ti))
    y = np.load(projectdir + 'source/setsT/trainingT/y_{}.npy'.format(ti))
    print('Ground truth min/max:\t', y.min(), y.max(), y.min()/y.mean(), y.max()/y.mean())
    print('Prediction min/max:\t\t', prediction.min(), prediction.max(), \
          prediction.min()/prediction.mean(), prediction.max()/prediction.mean(), '\n')

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.1)

    xi = x[:, :, 50, 0]
    yi = y[:, :, 50, 0]
    pi = prediction[:, :, 50, 0]
    Vmax = max(yi.max(), pi.max())

    im0 = axes[0].imshow(xi, cmap='magma', vmin=-3, vmax=3)
    im1 = axes[1].imshow(yi, cmap=mcmap, vmin=0.0, vmax=Vmax)
    im2 = axes[2].imshow(pi, cmap=mcmap, vmin=0.0, vmax=Vmax)
    im3 = axes[3].imshow(yi-pi, cmap='seismic')

    for ax, im, title in zip(axes, [im0, im1, im2, im3], ['Input density contrast', 'Ground truth', 'Prediction', 'Residual']):
        ax.add_patch(matplotlib.patches.Rectangle((16, 16), 96, 96, linewidth=0.4, edgecolor='lightgreen', facecolor='none'))
        ax.set_title(title)
        colorbar(im)

    plt.tight_layout()
    plt.show()
    plt.close()
