"""
    Under construction...
"""
import numpy as np, glob
import os, json
import matplotlib.pyplot as plt

from keras.models import Model
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

homedir = os.path.dirname(os.getcwd()) + '/'
input_shape = (128, 128, 128, 1)
test_ids  = np.arange(0, 1, 1)
datagen_params_test = {'dim': input_shape[0],
                       'mode': 'full',
                       'subsample_size': len(test_ids),
                       'batch_size': 1,
                       'n_channels': input_shape[-1],
                       'shuffle': False}
testing_generator = custom_DataGenerator(datapath=homedir+'data/training/', list_IDs=test_ids, **datagen_params_test)

with open(homedir+'run_1/hyper_param_dict.json') as json_file:
    param_dict = json.load(json_file)
net = get_model(**param_dict)
latest_weights = max(glob.glob(homedir+'run_1/saved_networks/*.hdf5'), key=os.path.getctime)
print('Loaded latest weights:', latest_weights)
net.load_weights(latest_weights)

net.compile('adam', loss=selective_mse, metrics=[])

# ....................
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
def mosaic(vol, fig=None, title=None, size=[10, 10], vmin=None, vmax=None,
           return_mosaic=False, cbar=True, return_cbar=False, **kwargs):
    if vmin is None:
        vmin = np.nanmin(vol)
    if vmax is None:
        vmax = np.nanmax(vol)

    sq = int(np.ceil(np.sqrt(len(vol))))

    # Take the first one, so that you can assess what shape the rest should be:
    im = np.hstack(vol[0:sq])
    height = im.shape[0]
    width = im.shape[1]

    # If this is a 4D thing and it has 3 as the last dimension
    if len(im.shape) > 2:
        if im.shape[2] == 3 or im.shape[2] == 4:
            mode = 'rgb'
        else:
            e_s = "This array has too many dimensions for this"
            raise ValueError(e_s)
    else:
        mode = 'standard'

    for i in range(1, sq):
        this_im = np.hstack(vol[(len(vol) / sq) * i:(len(vol) / sq) * (i + 1)])
        wid_margin = width - this_im.shape[1]
        if wid_margin:
            if mode == 'standard':
                this_im = np.hstack([this_im,
                                     np.nan * np.ones((height, wid_margin))])
            else:
                this_im = np.hstack([this_im,
                                     np.nan * np.ones((im.shape[2],
                                                       height,
                                                       wid_margin))])
        im = np.concatenate([im, this_im], 0)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
    else:
        # This assumes that the figure was originally created with this
        # function:
        ax = fig.axes[0]

    if mode == 'standard':
        imax = ax.matshow(im.T, vmin=vmin, vmax=vmax, **kwargs)
    else:
        imax = plt.imshow(np.rot90(im), interpolation='nearest')
        cbar = False
    ax.get_axes().get_xaxis().set_visible(False)
    ax.get_axes().get_yaxis().set_visible(False)
    returns = [fig]
    if cbar:
        # The colorbar will refer to the last thing plotted in this figure
        cbar = fig.colorbar(imax, ticks=[np.nanmin([0, vmin]),
                                         vmax - (vmax - vmin) / 2,
                                         np.nanmin([vmax, np.nanmax(im)])],
                            format='%1.2f')
        if return_cbar:
            returns.append(cbar)

    if title is not None:
        ax.set_title(title)
    if size is not None:
        fig.set_size_inches(size)

    if return_mosaic:
        returns.append(im)

    # If you are just returning the fig handle, unpack it:
    if len(returns) == 1:
        returns = returns[0]

    return returns
"""

"""
layer_outputs = [layer.output for layer in net.layers[1:]] # HAVE TO DO [1:] TO LEAVE OUT INPUT LAYER!
activation_model = Model(inputs=net.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

for lay in activation_model.layers:
    print(lay.name)
    if len(lay.get_weights())>0:
        data = lay.get_weights()[0]
        print(lay.get_weights()[0].shape, '\n')
        print(data.shape)
        print(data[:, :, :, 0, 0].shape)
        mosaic(vol=data[:, :, :, 0, 0], cbar=False)
        plt.show()

weight_conv2d_1 = activation_model.layers[1].get_weights()[0]
filters_ = np.reshape(weight_conv2d_1, newshape=(3, 3, 24))
filters_ = np.swapaxes(filters_, axis1=0, axis2=2)
for fislice in filters_:
    plt.figure()
    plt.imshow(fislice, cmap='bone')
plt.show()
"""