import tensorflow as tf
import keras.backend as K
from keras.losses import mae

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0, len(s))])


y_t = tf.placeholder(tf.float64, shape=[17, 128, 128, 128, 1])
y_p = tf.placeholder(tf.float64, shape=[17, 128, 128, 128, 1])

full_shape = shape(y_t)

G_x = []#tf.placeholder(tf.float32, shape=[17, 128, 128, 128, 1])
G_y = []#tf.placeholder(tf.float32, shape=[17, 128, 128, 128, 1])
G_z = []#tf.placeholder(tf.float32, shape=[17, 128, 128, 128, 1])
for i in range(shape(y_t)[3]):
    xy_slice = y_t[:,:,:, i,:]
    dx, dy = tf.image.image_gradients(xy_slice)

    yz_slice = y_t[:, i,:,:,:]
    dy, dz = tf.image.image_gradients(yz_slice)

    G_x.append(dx)
    G_y.append(dy)
    G_z.append(dz)

G_x = tf.stack(G_x, axis=1)
G_y = tf.stack(G_y, axis=1)
G_z = tf.stack(G_z, axis=1)
print(shape(G_x), shape(G_y), shape(G_z))

# make the selection here!
buffer = 16
G_x = G_x[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]
G_y = G_y[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]
G_z = G_z[:, buffer:-buffer, buffer:-buffer, buffer:-buffer, :]
print(shape(G_x), shape(G_y), shape(G_z))

#loss = mae(y_true=Gx_true, y_pred=Gx_pred) \
#       + mae(y_true=Gy_true, y_pred=Gy_pred) \
#       + mae(y_true=Gz_true, y_pred=Gz_pred)

raise SystemExit

slice_shape = shape(y_t_2d_xy_slice)
print(slice_shape, slice_shape[2])
dxt, dyt = tf.image.image_gradients(y_t_2d_xy_slice)
print(shape(dxt), shape(dyt))

y_t_2d_yz_slice = y_t[:, 0, :, :, :]
print(shape(y_t_2d_yz_slice))
dyt, dzt = tf.image.image_gradients(y_t_2d_yz_slice)
print(shape(dyt), shape(dzt))

"""
patches_true = tf.extract_volume_patches(input=y_true,
                                             ksizes=[1, 16, 16, 16, 1],
                                             strides=[1, 8, 8, 8, 1],
                                             padding='SAME')
patches_pred = tf.extract_volume_patches(input=y_pred,
                                         ksizes=[1, 16, 16, 16, 1],
                                         strides=[1, 8, 8, 8, 1],
                                         padding='SAME')
print(shape(patches_true), shape(patches_pred), '\n')
u_true = K.mean(patches_true, axis=-1)
u_pred = K.mean(patches_pred, axis=-1)
print(shape(u_true), shape(u_pred), '\n')
var_true = K.var(patches_true, axis=-1)
var_pred = K.var(patches_pred, axis=-1)
print(shape(var_true), shape(var_pred), '\n')
std_true = K.sqrt(var_true)
std_pred = K.sqrt(var_pred)
print(shape(std_true), shape(std_pred), '\n')

c1 = 0.01 ** 2
c2 = 0.03 ** 2
ssim = (2 * u_true * u_pred + c1) * (2* std_pred * std_true + c2)
denom= (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
print(shape(ssim), shape(denom))
#ssim /= K.clip why...?

ssim /= denom
print(shape(ssim))
dssim = K.mean((1.0 - ssim) / 2.0)
print(shape(dssim))
"""
