"""
    Script to visualize intermediate network outputs, i.e. the input data sample at a given point in the network
    (and NOT the individual filters -> see visualize_filter_volumetric for that)
"""
import numpy as np, glob
import os, json
import matplotlib.pyplot as plt

from keras.models import Model
from unet.network.u_net import get_model
from unet.metrics.custom_metrics import selective_mse, selective_r2_score, r2_score
from unet.generator.datagen import custom_DataGenerator

from matplotlib.colors import LinearSegmentedColormap

import matplotlib.cm as cm
viridis = cm.get_cmap('magma')
values = [viridis(x) for x in np.linspace(0, 1, 100)]
#values[0] = (1, 1, 1, 1)    # set the first value to white
last = values[-1]
for x in range(50):
    values.append(last)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list('mycmap', values)

homedir = os.path.dirname(os.getcwd()) + '/'
projectdir = os.path.dirname(os.path.dirname(os.getcwd())) + '/'
input_shape = (128, 128, 128, 1)
test_ids  = np.arange(0, 1, 1)
datagen_params_test = {'dim': input_shape[0],
                       'mode': 'full',
                       'subsample_size': len(test_ids),
                       'batch_size': 1,
                       'n_channels': input_shape[-1],
                       'shuffle': False}
testing_generator = custom_DataGenerator(datapath=homedir+'data/training/', list_IDs=test_ids, **datagen_params_test)

with open(homedir+'run_1_old/hyper_param_dict.json') as json_file:
    param_dict = json.load(json_file)

net = get_model(**param_dict)
latest_weights = max(glob.glob(homedir+'run_1_old/saved_networks/*.hdf5'), key=os.path.getctime)
print('Loaded latest weights:', latest_weights)
net.load_weights(latest_weights)

net.compile('adam', loss=selective_mse, metrics=['mse',
                                                 selective_r2_score,
                                                 r2_score])

# ....................

layer_outputs = [layer.output for layer in net.layers[1:]] # HAVE TO DO [1:] TO LEAVE OUT INPUT LAYER!
activation_model = Model(inputs=net.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

x = np.load(homedir+'data/training/x_111.npy')
y = np.load(homedir+'data/training/y_111.npy')

activations = activation_model.predict(x[np.newaxis, ...]) # Returns a list of five Numpy arrays: one array per layer activation
print(len(activations))

imgs = []

plt.figure()
plt.imshow(y[:, :, 0, 0], cmap=cm)
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/target_0.png', dpi=150)
plt.figure()
plt.imshow(y[:, -1, :, 0], cmap=cm)
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/target_1.png', dpi=150)
plt.figure()
plt.imshow(np.rot90(y[0, :, :, 0]), cmap=cm)
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/target_2.png', dpi=150)

plt.figure()
plt.imshow(x[:, :, 0, 0], cmap='magma', clim=(-4, 4))
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/input_0.png', dpi=150)
plt.figure()
plt.imshow(x[:, -1, :, 0], cmap='magma', clim=(-4, 4))
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/input_1.png', dpi=150)
plt.figure()
plt.imshow(np.rot90(x[0, :, :, 0]), cmap='magma', clim=(-4, 4))
plt.axis('off')
plt.savefig(projectdir + 'graphics/draw.io/3D/input_2.png', dpi=150)

for i, activation in enumerate(activations):
    print(activation.shape)
    if i in [4, 10, 16, 22, 28,
             33, 41, 49, 57, 65, 74]:
        if i in [33, 41, 49, 57, 65, 74]:
            colormap = cm
        else:
            colormap = 'magma'

        plt.figure()
        plt.imshow(activation[0, :, :, 0, 0], cmap=colormap)
        plt.axis('off')
        plt.savefig(projectdir + 'graphics/draw.io/3D/{}_0.png'.format(i), dpi=150)

        plt.figure()
        plt.imshow(activation[0, :, -1, :, 0], cmap=colormap)
        plt.axis('off')
        plt.savefig(projectdir + 'graphics/draw.io/3D/{}_1.png'.format(i), dpi=150)

        plt.figure()
        plt.imshow(np.rot90(activation[0, 0, :, :, 0]), cmap=colormap)
        plt.axis('off')
        plt.savefig(projectdir + 'graphics/draw.io/3D/{}_2.png'.format(i), dpi=150)







