"""
    Custom implementation of various keras callbacks.

    Paranoia:               callback to save history on each epoch end, show live progress of train/validation loss and saving it
    Optimizer_cb:           callback to save the optimizer state at the end of every k-th epoch
    Live_Prediction_cb:     callback to make a couple predictions at the end of epoch on-the-fly and saving them as png images
    Custom_checkpointer:    customized callback for saving the model; frequency and other details are parametrized
"""

import matplotlib.pyplot as plt
import matplotlib
from keras import callbacks
import keras.backend as K
import numpy as np
import pickle, os

# Plotting Parameters that I want changed
matplotlib.rcParams.update(
    {
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    })

plt.rcParams['grid.alpha'] = 1.0
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['grid.linewidth'] = .5
plt.rcParams['legend.edgecolor'] = 'white'
#plt.rcParams['legend.handletextpad'] = 0.0
plt.rcParams['legend.borderpad'] = 0
plt.rcParams['legend.framealpha'] = .8
plt.rcParams['legend.frameon'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class Paranoia(callbacks.Callback, ):
    def __init__(self, savepath):
        self.savepath = savepath

    def on_train_begin(self, logs={}):
        self.i = 1
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure(figsize=(12, 8))

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        plt.close()
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('loss'))
        self.val_acc.append(logs.get('val_loss'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5, 6))
        f.subplots_adjust(hspace=0)

        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, '-', c='#00695C', linewidth=0.3, label="training loss")
        ax1.plot(self.x, self.val_losses, '-', c='#BF360C', linewidth=0.3, label="validation loss")
        ax1.grid(False)

        ax11 = ax1.twinx()
        ratio = np.array(self.losses) / np.array(self.val_losses)
        ax11.plot(self.x, ratio, ':', c='#546E7A', linewidth=0.5, label=r'L$_{train}$ / L$_{val}$')
        ax11.grid(False)
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)
        ax11.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        h1, l1 = ax1.get_legend_handles_labels()
        h11, l11 = ax11.get_legend_handles_labels()
        ax1.legend(h1+h11, l1+l11, loc='best')

        ax2.plot(self.x, self.acc, '-', c='#4527A0', linewidth=0.5, label="training loss")
        ax2.plot(self.x, self.val_acc, '--', c='#4527A0', linewidth=0.5, label="validation loss")
        ax2.legend(loc='best')
        ax2.set_ylim((1e-5, 1e-1))
        ax2.set_yscale('log')
        #ax2.xaxis.set_ticks(self.x)
        #ax2.yaxis.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #ax2.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
        #ax2.set_xlabel('Number of epochs')
        ax2.grid(False)
        ax2.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(0.5)
            ax11.spines[axis].set_linewidth(0.5)
            ax2.spines[axis].set_linewidth(0.5)
        ax2.spines['top'].set_linewidth(0.0)
        ax1.set_title('Epoch {}'.format(epoch + 1))
        plt.savefig(self.savepath+'/live_output/e_{}'.format(epoch + 1), dpi=200)
        plt.show(block=False)
        plt.pause(0.01)

class Optimizer_cb(callbacks.Callback):
    def __init__(self, interval, save_folder):
        self.interval = interval
        self.save_folder = save_folder

    def on_train_begin(self, logs=None):
        self.i = 1
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))

        symbolic_weights = getattr(self.model.optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)

        if epoch % self.interval == 0:  # save every k-th epoch
            with open(self.save_folder + 'latest_optimizer_state.pkl', 'wb') as f:
                pickle.dump(weight_values, f, protocol=2)
            print('-> current learning_rate:', K.eval(lr_with_decay))
            print('-> current decay:', K.eval(decay))
            print('-> saved optimizer state')
        self.i += 1

class Live_Prediction_cb(callbacks.Callback):
    def __init__(self, savepath):
        self.savepath = savepath

    def colorbar(self, mappable, colorbar_label):
        # from mpl_toolkits.axes_grid1 import make_axes_locatable
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax = mappable.axes
        fig = ax.figure
        cax = inset_axes(ax,
                         width="85%",  # width = % of parent_bbox width
                         height="4%",  # height : %
                         loc='upper center',
                         bbox_to_anchor=(0.0, 0.0, 1.0, 1.07),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        cb = fig.colorbar(mappable, cax=cax, label=colorbar_label, orientation='horizontal')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=9)
        return cb

    def on_epoch_end(self, epoch, logs=None):

        for t in [0, 13, 26, 50, 53, 62]:
            x = np.load(os.getcwd() + '/data/validation/x_{}.npy'.format(t))
            y = np.load(os.getcwd() + '/data/validation/y_{}.npy'.format(t))
            pred = self.model.predict(x[np.newaxis, ...])

            from matplotlib.colors import LinearSegmentedColormap
            mcmap = LinearSegmentedColormap.from_list('mycmap', ['#3F1F47', '#5C3C9A', '#6067B3',
                                                                 #   '#969CCA',
                                                                 '#6067B3', '#5C3C9A', '#45175D', '#2F1435',
                                                                 '#601A49', '#8C2E50', '#A14250',
                                                                 '#B86759',
                                                                 '#E0D9E1'][::-1])

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,5))
            im1 = ax1.imshow(x[:, :, 50, 0], cmap='magma', vmin=-3, vmax=3)
            im2 = ax2.imshow(y[:, :, 50, 0], cmap=mcmap, vmin=0.0, vmax=1.0)
            im3 = ax3.imshow(pred[0, :, :, 50, 0], cmap=mcmap, vmin=0.0, vmax=1.0)

            cb1 = self.colorbar(im1, 'Input density contrast')
            cb2 = self.colorbar(im2, 'Ground truth distance')
            cb3 = self.colorbar(im3, 'Predicted distance at iteration {}'.format(epoch+1))
            cb1.set_ticks([-2, -1, 0, 1, 2])
            cb2.set_ticks([0, 0.5, 1])
            cb3.set_ticks([0, 0.5, 1])
            cb1.outline.set_linewidth(1.5)
            cb2.outline.set_linewidth(1.5)
            cb3.outline.set_linewidth(1.5)

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
            plt.tight_layout()
            plt.savefig(self.savepath+'/live_output/p_{0:}_{1:}'.format(t, epoch+1), dpi=250)
            plt.close()

class Custom_checkpointer(callbacks.Callback):
    def __init__(self, interval, save_folder, mode):
        self.interval = interval
        self.save_folder = save_folder
        self.mode = mode

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:  # save every k-th epoch
            if self.mode == 'weights_only':
                self.model.save_weights(filepath=self.save_folder + 'best_net_{0:04d}.hdf5'.format(epoch))
                print('↓ saved model weights')
            elif self.mode == 'full':
                self.model.save(filepath=self.save_folder + 'best_net_{0:04d}.hdf5'.format(epoch))