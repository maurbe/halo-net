"""
    Purpose:    This is a code snippet to investigate other techniques of stitiching, e.g. taking maximum values where
                predicted subboxes overlap.
    Comment:    I find that this "max-stitching" technique is more expensive and yields worse results.
"""

import numpy as np, json, glob, os
import scipy.ndimage as nd
from sklearn.preprocessing import scale
from unet.network.u_net import get_model

# re-initialize the trained network
homedir = os.path.dirname(os.getcwd()) + '/'
with open(homedir + 'unet/run_1/hyper_param_dict.json') as json_file:
    param_dict = json.load(json_file)
net = get_model(**param_dict)
latest_weights = max(glob.glob(homedir + 'unet/run_1/saved_networks/*.hdf5'), key=os.path.getctime)
print('Loaded latest weights:', latest_weights)
net.load_weights(latest_weights)

dims = 512
input_full = np.load(homedir + 'source/cic_fieldsA/cic_densityA.npy') * dims**3
input_full = nd.gaussian_filter(input_full, sigma=2.5, mode='wrap')
delta_scaled = scale(input_full.flatten()).reshape(input_full.shape)

pad_width = 64
delta_scaled = np.pad(delta_scaled, pad_width=pad_width, mode='wrap')
print(delta_scaled.shape)

buffer = 16
box = np.zeros_like(delta_scaled)
stepsize = 32
c = 0
for k in range(17):
    for j in range(17):
        for i in range(17):
            current_input = delta_scaled[i * stepsize : i * stepsize + 128,\
                                         j * stepsize : j * stepsize + 128,\
                                         k * stepsize : k * stepsize + 128]
            current_pred  = net.predict(x=current_input[np.newaxis, ..., np.newaxis])[0,...,0]
            box[i * stepsize + buffer : i * stepsize + 128 - buffer,\
                j * stepsize + buffer : j * stepsize + 128 - buffer,\
                k * stepsize + buffer : k * stepsize + 128 - buffer] =\
                np.max(np.stack(
                    [current_pred[buffer:-buffer, buffer:-buffer, buffer:-buffer],
                     box[i * stepsize + buffer : i * stepsize + 128 - buffer, \
                         j * stepsize + buffer : j * stepsize + 128 - buffer, \
                         k * stepsize + buffer : k * stepsize + 128 - buffer]]), axis=0)
            print(c)
            c += 1

box = box[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]
np.save('boxesA/prediction_max_stitchingA.npy', box)
