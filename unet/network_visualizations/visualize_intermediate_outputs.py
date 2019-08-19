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
mcmap = LinearSegmentedColormap.from_list('mycmap', ['#3F1F47', '#5C3C9A', '#6067B3', '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                     '#601A49', '#8C2E50', '#A14250', '#B86759', '#E0D9E1'][::-1])

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

net.compile('adam', loss=selective_mse, metrics=['mse',
                                                 selective_r2_score,
                                                 r2_score])

# ....................

layer_outputs = [layer.output for layer in net.layers[1:]] # HAVE TO DO [1:] TO LEAVE OUT INPUT LAYER!
activation_model = Model(inputs=net.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input

x = np.load(homedir+'data/training/x_0.npy')
y = np.load(homedir+'data/training/y_0.npy')

activations = activation_model.predict(x[np.newaxis, ...]) # Returns a list of five Numpy arrays: one array per layer activation
print(len(activations))
imgs = []
plt.figure()
plt.imshow(x[:, :, 50, 0], cmap='magma')
plt.axis('off')
for i, activation in enumerate(activations):
    print(activation.shape)
    if i in [4, 10, 16, 22, 28,
             33, 41, 49, 57, 65, 74]:
        if i in [33, 41, 49, 57, 65, 74]:
            colormap = mcmap
        else:
            colormap = 'magma'
        plt.figure()
        plt.imshow(activation[0, :, :, np.rint(50.0/128.0*activation.shape[3]).astype(int), 0], cmap=colormap)
        print(np.rint(50/128.0*activation.shape[3]).astype(int))
        plt.axis('off')

plt.figure()
plt.imshow(y[:, :, 50, 0], cmap=mcmap)
plt.axis('off')
plt.show()






