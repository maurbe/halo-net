from __future__ import division
import keras.backend as K
import tensorflow as tf
from keras.losses import binary_crossentropy, mean_squared_error, mean_absolute_error, mae


def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])

def selection(tensor, buffer):
    # assumed shape = (B, x, y, z, channels)
    tensor = tensor[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]
    return tensor

# ............................................................................

def selective_bce(y_true, y_pred):
    buffer = 16
    y_true = selection(tensor=y_true, buffer=buffer)
    y_pred = selection(tensor=y_pred, buffer=buffer)
    return binary_crossentropy(y_true, y_pred)

def selective_mse(y_true, y_pred):
    buffer = 16
    y_true = selection(tensor=y_true, buffer=buffer)
    y_pred = selection(tensor=y_pred, buffer=buffer)
    return mean_squared_error(y_true, y_pred)

def selective_mae(y_true, y_pred):
    buffer = 16
    y_true = selection(tensor=y_true, buffer=buffer)
    y_pred = selection(tensor=y_pred, buffer=buffer)
    return mean_absolute_error(y_true, y_pred)

def selective_mae_normalized(y_true, y_pred):
    buffer = 16
    y_true = selection(tensor=y_true, buffer=buffer)
    y_pred = selection(tensor=y_pred, buffer=buffer)
    return mean_absolute_error(y_true, y_pred) / (K.mean(y_true + K.epsilon()) + K.epsilon())

def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1.0 - SS_res/(SS_tot + K.epsilon()))

def selective_r2_score(y_true, y_pred):
    buffer = 16
    y_true_s = selection(tensor=y_true, buffer=buffer)
    y_pred_s = selection(tensor=y_pred, buffer=buffer)

    y_true_s = K.flatten(y_true_s)
    y_pred_s = K.flatten(y_pred_s)

    SS_res = K.sum(K.square(y_true_s-y_pred_s))
    SS_tot = K.sum(K.square(y_true_s - K.mean(y_true_s)))
    return (1.0 - SS_res/(SS_tot + K.epsilon()))

def image_gradient3d(tensor):
    # assumed shape = (B, x, y, z, channels)
    Gx, Gy, Gz = [], [], []
    length = 128
    for i in range(length):    #shape(tensor)[3]):
        xy_slice = tensor[:, :, :, i, :]
        dx, dy = tf.image.image_gradients(xy_slice)

        yz_slice = tensor[:, i, :, :, :]
        dy, dz = tf.image.image_gradients(yz_slice)

        Gx.append(dx)
        Gy.append(dy)
        Gz.append(dz)

    Gx = tf.stack(Gx, axis=1)
    Gy = tf.stack(Gy, axis=1)
    Gz = tf.stack(Gz, axis=1)
    return Gx, Gy, Gz

def selective_mae_of_gradients3d(y_true, y_pred):
    """
    :param y_true:  self explanatory
    :param y_pred:  self explanatory
    :return:        joint loss of selective normalized mae and selective mae of gradients
    :Todo:          bahviour has not been tested thouroghly
    """
    Gx_true, Gy_true, Gz_true = image_gradient3d(tensor=y_true)
    Gx_pred, Gy_pred, Gz_pred = image_gradient3d(tensor=y_pred)

    buffer = 16

    Gx_true = selection(tensor=Gx_true, buffer=buffer)
    Gy_true = selection(tensor=Gy_true, buffer=buffer)
    Gz_true = selection(tensor=Gz_true, buffer=buffer)
    Gx_pred = selection(tensor=Gx_pred, buffer=buffer)
    Gy_pred = selection(tensor=Gy_pred, buffer=buffer)
    Gz_pred = selection(tensor=Gz_pred, buffer=buffer)

    loss   = mae(y_true=Gx_true, y_pred=Gx_pred) \
           + mae(y_true=Gy_true, y_pred=Gy_pred) \
           + mae(y_true=Gz_true, y_pred=Gz_pred)

    return loss


def loss_DSSIM_tf(y_true, y_pred):
    patches_true = tf.extract_volume_patches(input=y_true,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    patches_pred = tf.extract_volume_patches(input=y_pred,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    u_true, var_true = tf.nn.moments(patches_true + K.epsilon(), axes=[-1])
    u_pred, var_pred = tf.nn.moments(patches_pred + K.epsilon(), axes=[-1])
    std_true = K.sqrt(var_true + K.epsilon())
    std_pred = K.sqrt(var_pred + K.epsilon())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    c3 = c2 / 2.0

    l = ((2 * u_true * u_pred) + c1) / (u_true * u_true + u_pred * u_pred + c1)
    c = ((2 * std_true * std_pred) + c2) / (var_true + var_pred + c2)
    s = (std_true * std_pred + c3) / (std_true * std_pred + c3)
    ssim = l * c * s
    loss = (1.0 - K.mean(ssim + K.epsilon())) / 2.0
    return loss

def total_loss(y_true, y_pred):
    tot_loss = loss_DSSIM_tf(y_true=y_true, y_pred=y_pred) + selective_mae_of_gradients3d(y_true=y_true, y_pred=y_pred)
    return tot_loss

# ............................................................................

"""
def pixel_wise_loss(y_true, y_pred):
    pos_weight = tf.constant([[1.0, 2.0]])
    loss = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, pos_weight, name=None)
    return K.mean(loss,axis=-1)

def smoothL1(y_true, y_pred):
    HUBER_DELTA = 0.5
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return  K.sum(x)
"""

"""
def loss_DSSIM_tf(y_true, y_pred):
    
    Main function to call to compute the loss function DSSIM = (1-SSIM)/2
    :param y_true:  ground truth
    :param y_pred:  prediction
    :return:        DSSIM(y_true, y_pred)
    Todo:           A general note of caution when designing custom loss functions:
                    Backend functions like K.var, K.sqrt, etc. are prone to return NaN,
                    e.g. the variance might become a very small negative number in float32 representation
                    where it should be zero (i.e. K.sqrt(-x) = NaN) --> adding a K.epsilon() seems to fix those issues
    Todo:           Can we just sum losses to form a joint loss function without normalizing them wrt each other?
                    Yes(?), since the algorithm minimizes the gradient of the loss function, and thus the losses can be on
                    different magnitudes.(??)
    Todo:           Also play with the ksizes and strides parameters.
    Todo:           Try the following loss:
                        L = w * l1 + (1-w) * DSSIM where w has to be determined (in theory by cross validation)
    Todo:           Make the loss selective to innermost region.
    
    #y_true = tf.transpose(y_true, [0, 4, 1, 2, 3]) only for theano
    #y_pred = tf.transpose(y_pred, [0, 4, 1, 2, 3]) only for theano

    patches_true = tf.extract_volume_patches(input=y_true,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    patches_pred = tf.extract_volume_patches(input=y_pred,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    # HERE comes the selective element
    # ...

    #u_true = K.mean(patches_true, axis=-1)
    #u_pred = K.mean(patches_pred, axis=-1)
    # Caution: I do not understand why, but computing the variance with
    # keras backend, i.e. var = K.var(patches, axis=-1), always throws a shape mismatch error
    # for now: use tf.nn.moments to compute mean and var = std**2
    u_true, var_true = tf.nn.moments(patches_true + K.epsilon(), axes=[-1])
    u_pred, var_pred = tf.nn.moments(patches_pred + K.epsilon(), axes=[-1])
    std_true = K.sqrt(var_true + K.epsilon())
    std_pred = K.sqrt(var_pred + K.epsilon())
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    c3 = c2 / 2.0

    l = ((2 * u_true * u_pred) + c1) / (u_true * u_true + u_pred * u_pred + c1)
    c = ((2 * std_true * std_pred) + c2) / (var_true + var_pred + c2)
    s = (std_true * std_pred + c3) / (std_true * std_pred + c3)

    #cs = c * s
    #l = (l + 1.0) / 2.0
    #cs = (cs + 1.0) / 2.0
    #cs = tf.reduce_prod(cs)
    ssim = l * c * s
    #ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    return (1.0 - K.mean(ssim + K.epsilon())) / 2.0


    #ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    #denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    #ssim /= denom
    #ssim /= K.clip(denom, K.epsilon(), np.inf) # no division by 0
    #ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    
    #return K.mean((1.0 - ssim) / 2.0)
"""

"""
def selective_mae_DSSIM(y_true, y_pred):
    
    Parametrized, selective joint loss function of mae+DSSIM
    :param y_true:  ground truth
    :param y_pred:  prediction
    :return:        parametrized loss function
    :Todo:          Investigating best alpha value via cross validation.
    
    # First, selective mse, normalization not required
    alpha       = 0.0
    buffer      = 16
    y_true_s    = y_true[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]
    y_pred_s    = y_pred[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]

    loss1 = mean_absolute_error(y_true=y_true_s, y_pred=y_pred_s)


    # Second, selective DSSIM
    patches_true = tf.extract_volume_patches(input=y_true_s,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    patches_pred = tf.extract_volume_patches(input=y_pred_s,
                                             ksizes=[1, 4, 4, 4, 1],
                                             strides=[1, 1, 1, 1, 1],
                                             padding='VALID')
    u_true, var_true    = tf.nn.moments(patches_true + K.epsilon(), axes=[-1])
    u_pred, var_pred    = tf.nn.moments(patches_pred + K.epsilon(), axes=[-1])
    std_true            = K.sqrt(var_true + K.epsilon())
    std_pred            = K.sqrt(var_pred + K.epsilon())
    c1  = 0.01 ** 2
    c2  = 0.03 ** 2
    c3  = c2 / 2.0

    l   = ((2 * u_true * u_pred) + c1) / (u_true * u_true + u_pred * u_pred + c1)
    c   = ((2 * std_true * std_pred) + c2) / (var_true + var_pred + c2)
    s   = (std_true * std_pred + c3) / (std_true * std_pred + c3)

    ssim = l * c * s
    loss2 = (1.0 - K.mean(ssim + K.epsilon())) / 2.0

    # combined loss
    loss = alpha * loss1 + (1-alpha) * loss2
    return loss
"""


"""
# https://en.wikipedia.org/wiki/Root-mean-square_deviation#Normalized_root-mean-square_deviation
mean_P  = K.mean(y_pred_s)
mean_T  = K.mean(y_true_s)
Nrmse    = mean_absolute_error(y_true_s, y_pred_s) / K.mean(y_true_s)#K.sqrt(mean_squared_error(y_true_s, y_pred_s)) #/ K.mean(y_true_s)
return Nrmse*0.5 + loss_DSSIM_tf(y_true, y_pred)
"""


