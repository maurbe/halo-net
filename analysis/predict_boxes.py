import numpy as np
import json, glob, os
from unet.network.u_net import get_model
from unet.generator.datagen import custom_DataGenerator

def predict_and_stitch(datapath):
    """
    Main function to call
    :param datapath:    data path to the sets (x,y)
    :return:            stitched cube of original dimensions (e.g. 512^3)
    Todo:               investigate if there is a better technique of stitching the boxes together,
                        e.g. take mean values where subboxes overlap (opposed to "hard" stitching of innermost subboxes)
    """
    input_shape = (128, 128, 128, 1)
    subbox_ids = np.arange(0, 512, 1)

    # re-initialize the trained network
    homedir = os.path.dirname(os.getcwd()) + '/'
    with open(homedir + 'unet/run_1/hyper_param_dict.json') as json_file:
        param_dict = json.load(json_file)
    net = get_model(**param_dict)
    latest_weights = max(glob.glob(homedir + 'unet/run_1/saved_networks/*.hdf5'), key=os.path.getctime)
    print('Loaded latest weights:', latest_weights)
    net.load_weights(latest_weights)

    datagen_params_test = {'dim': input_shape[0],
                           'mode': 'full',
                           'datapath': homedir + datapath,
                           'subsample_size': len(subbox_ids),
                           'batch_size': 1,
                           'n_channels': input_shape[-1],
                           'shuffle': False,
                           'randomize': False}

    data_generator = custom_DataGenerator(list_IDs=subbox_ids, **datagen_params_test)
    prediction = net.predict_generator(generator=data_generator, verbose=True)[..., 0]

    # since the prediction is a flat array holding subboxes, need to stitch full box back together
    box = np.zeros(shape=(512, 512, 512))
    size = 64
    buffer = 32
    c = 0
    for k in range(8):
        for j in range(8):
            for i in range(8):
                box[i * size: (i + 1) * size, j * size: (j + 1) * size, k * size: (k + 1) * size] = \
                    prediction[c][buffer:-buffer, buffer:-buffer, buffer:-buffer]
                c += 1
                print(c)
    return box

#box_finalT = predict_and_stitch(datapath='source/setsT/trainingT/')
#np.save('boxesT/predictionT.npy', box_finalT)

box_finalA = predict_and_stitch(datapath='source/setsA/validationA/')
np.save('boxesA/predictionA.npy', box_finalA)

