import numpy as np
import keras
import warnings

class custom_DataGenerator(keras.utils.Sequence):
    """
        A class adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        to feed data to keras models in a highly customizable way without having to pre-load entire data into memory.

        - 'Full' mode allows to train on entire training set in each epoch,
            where the data can be shuffled at the start of each epoch
        - 'Minibatch' mode allows for minibatch training. Through setting subsample_size > (mini-mini-)batchsize
            it is possible to randomly select a minibatch and train on it by dividing it further into mini-mini-batches.
        -  It is possible to perform virtual data-augmentation "on-the-fly" by randomizing the orientation of each cube
            before it is passed to the network (through self.randomize).
    """
    def __init__(self, datapath, list_IDs, mode, subsample_size,
                 batch_size=100, dim=100, n_channels=100, shuffle=True, randomize=False):
        """
        Initialize
        :param datapath:        directory from where the data flows
        :param list_IDs:        valid id's of data samples
        :param mode:            either 'minibatch' or 'full'
        :param subsample_size:  the (mini) batch size of the data for one training iteration
        :param batch_size:      the (mini-mini) batch size to divide the subsample_size in smaller batches (normally should be equal to subsample_size)
        :param dim:             dimension of inputs
        :param n_channels:      number of channels; last entry of dim
        :param shuffle:         whether to shuffle the dataset between iterations
        :param randomize:       whether or not to randomize (randomly rotating, etc.) the individual samples
        """
        self.mode = mode
        if self.mode not in ['minibatch', 'full']:
            print("\nExiting script: 'mode' must be one of 'minibatch' or 'full'.")
            raise AssertionError

        self.subsample_size = subsample_size
        self.datapath = datapath
        print("Data flowing from ", self.datapath)
        self.dim = dim
        self.batch_size = batch_size
        assert self.subsample_size >= self.batch_size
        if self.mode == 'minibatch' and self.subsample_size!=self.batch_size:
            warnings.warn('Are you sure you dont want to train on full minibatch?')

        self.all_ids = list_IDs

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.randomize = randomize
        # first-time shuffling, generating...
        self.on_epoch_end()


    def __len__(self):
        """
        :return:                the number of batches per epoch
        """
        #print(self.mode, int(np.floor(len(self.list_IDs) / self.batch_size)))
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:           randomized indices of data samples
        :return:                data in the (mini-mini) batch
        """
        # Generate indexes of the batch
        idxes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in idxes]
        # uncomment next line for debuggin/printing current indices
        #print(list_IDs_temp)

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Generate new random sample, possible shuffling at epoch/iteration end
        """
        if self.shuffle == True:
            np.random.shuffle(self.all_ids)
        else:
            self.all_ids = self.all_ids

        if self.mode == 'minibatch':
            self.list_IDs = np.random.choice(self.all_ids, self.subsample_size, replace=False)  # list_IDs
        else:   # mode = 'full'
            self.list_IDs = self.all_ids
        # debugging...
        #print(self.list_IDs)
        self.indexes = np.arange(len(self.list_IDs))

    def randomize_cube(self, cube, nx, ny, nz):
        cube = np.rot90(cube, axes=(1, 2), k=nx)
        cube = np.rot90(cube, axes=(0, 2), k=ny)
        cube = np.rot90(cube, axes=(0, 1), k=nz)
        return cube

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
        Randomizes orientation for "on-the-fly" data augmentation if self.randomize==True
        :param list_IDs_temp:       temporary sample ids
        :return:                    data accoring to temporary ids
        """
        # Initialization
        X = np.empty((self.batch_size, self.dim, self.dim, self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim, self.dim, self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample and target
            Xi = np.load(self.datapath + 'x_{}.npy'.format(ID))
            yi = np.load(self.datapath + 'y_{}.npy'.format(ID))

            if self.randomize==True:
                # perform randomization
                nx = np.random.randint(0, 4)
                ny = np.random.randint(0, 4)
                nz = np.random.randint(0, 4)
                Xi = self.randomize_cube(cube=Xi, nx=nx, ny=ny, nz=nz)
                yi = self.randomize_cube(cube=yi, nx=nx, ny=ny, nz=nz)
                # print (nx, ny, nz)
            X[i,] = Xi
            y[i,] = yi

        return X, y