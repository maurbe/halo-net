
PRETRAINED = False

# set up current run-folder
import os, shutil, glob, json
path = '{}/'.format(os.getcwd())
RUN_FOLDER = os.path.basename(__file__).rstrip('.py')

if PRETRAINED==False:
    if os.path.exists(RUN_FOLDER):
        shutil.rmtree(RUN_FOLDER)
    os.makedirs(RUN_FOLDER)
    print('Records will be saved to '+ RUN_FOLDER)
    subfolder_names = ['/log/', '/history/', '/saved_networks/', '/live_output/']
    for subfolder_name in subfolder_names:
        os.makedirs(RUN_FOLDER + subfolder_name)


# set random seed to make runs reproducible
import numpy as np
from tensorflow import set_random_seed
np.random.seed(20)
set_random_seed(20)

from keras.utils import plot_model
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger

from helpers.custom_callbacks import (Paranoia,
                                      Optimizer_cb,
                                      Live_Prediction_cb,
                                      Custom_checkpointer)
from metrics.custom_metrics import (selective_mse,
                                    selective_mae, selective_mae_normalized, selective_mae_normalized_and_gradients3d,
                                    r2_score, selective_r2_score)
from generator.datagen import custom_DataGenerator
from network.u_net import get_model


# Hyper-parameters and flags
input_shape         = (128, 128, 128, 1)
n_filters           = 12
kernel_size         = 3
pool_size           = 2
stride              = 2

dropout_rate_conv   = 0.5
lrelu_alpha         = 0.05
last_activation     ='relu'

hyper_param_dict = {'input_shape':      input_shape,
                    'n_filters':        n_filters,
                    'kernel_size':      kernel_size,
                    'pool_size':        pool_size,
                    'stride':           stride,
                    'dropout_rate_conv':dropout_rate_conv,
                    'lrelu_alpha':      lrelu_alpha,
                    'last_activation':  last_activation}
if PRETRAINED==False:
    with open(RUN_FOLDER + '/hyper_param_dict.json', 'w') as outfile:
        json.dump(hyper_param_dict, outfile, indent=4)



# initialize network
net = get_model(**hyper_param_dict)
net.compile(optimizer=Adam(lr=1e-4, clipvalue=0.5),
            loss=selective_mae_normalized,
            metrics=['mse', selective_mse,
                     'mae', selective_mae, selective_mae_normalized_and_gradients3d,
                      r2_score, selective_r2_score])
net.summary(line_length=150)
#plot_model(net, to_file=RUN_FOLDER+'net.png', show_shapes=True, show_layer_names=True)

callbacks = [Custom_checkpointer(save_folder=RUN_FOLDER + '/saved_networks/', interval=5, mode='weights_only'),
             Optimizer_cb(save_folder=RUN_FOLDER + '/saved_networks/', interval=5),
             CSVLogger(RUN_FOLDER+'/history/history.csv', separator='\t', append=True),
             #ModelCheckpoint(filepath=RUN_FOLDER + '/saved_networks/best_net_{epoch:03d}.hdf5',
             #                save_best_only=False, save_weights_only=True, verbose=1),
             #ReduceLROnPlateau(monitor='val_loss',
             #                  factor=0.1, patience=1000, min_lr=0.000001, verbose=1),
             #EarlyStopping(patience=100, verbose=1),
             Paranoia(savepath=RUN_FOLDER),
             Live_Prediction_cb(savepath=RUN_FOLDER),
             TensorBoard(log_dir=RUN_FOLDER+'/log/')] # we cannot display histograms when using a generator...

train_ids = np.load(path + 'data/ids.npz')['train_ids']
test_ids  = np.load(path + 'data/ids.npz')['test_ids']
datagen_params_train = {'dim': input_shape[0],
                        'mode': 'minibatch',
                        'datapath': os.getcwd() +'/data/training/',
                        'subsample_size': 16,
                        'batch_size': 16,
                        'n_channels': input_shape[-1],
                        'shuffle': True,
                        'randomize': True}
datagen_params_test  = {'dim': input_shape[0],
                        'mode': 'minibatch',
                        'datapath': os.getcwd() +'/data/validation/',
                        'subsample_size': 16,
                        'batch_size': 16,
                        'n_channels': input_shape[-1],
                        'shuffle': False,
                        'randomize': False}
training_generator      = custom_DataGenerator(list_IDs=train_ids, **datagen_params_train)
validation_generator    = custom_DataGenerator(list_IDs=test_ids, **datagen_params_test)



if PRETRAINED:
    # reinitialize the model with the same hyper_params, or change them if needed
    with open('run_1/hyper_param_dict.json') as json_file:
        param_dict = json.load(json_file)
    net = get_model(**hyper_param_dict)
    print('Reinitialized pretrained model.')

    # compile it with the desired loss and metrics
    net.compile(optimizer=Adam(lr=1e-4, clipvalue=0.5),
                loss=selective_mae_normalized,
                metrics=['mse', selective_mse,
                         'mae', selective_mae, selective_mae_normalized_and_gradients3d,
                         r2_score, selective_r2_score])

    # transfer the weights from (latest) pretrained model
    list_of_weights = glob.glob('run_1/saved_networks/*.hdf5')
    latest_weights = max(list_of_weights, key=os.path.getctime)
    print('Loaded latest weights:', latest_weights)
    net.load_weights(latest_weights)
    net._make_train_function()

    # transfer the optimizer state
    # CAUTION: this resets the optimizer to the last state form pretrained model, if e.g. the model diverges due to
    # too high learning_rate, initialize with a fresh optimizer, i.e. do not uncomment the following lines
    #with open('run_1/saved_networks/latest_optimizer_state.pkl', 'rb') as f:
    #    weight_values = pickle.load(f)
    #net.optimizer.set_weights(weight_values)


# train the model
net.fit_generator(generator=training_generator,
                  validation_data=validation_generator,
                  callbacks=callbacks,
                  epochs=10000)
