from keras.models import Model
from keras.layers import Input, Dropout, Conv3D, Conv3DTranspose, concatenate, MaxPooling3D, LeakyReLU

def conv3d_block(input_tensor, n_filters, kernel_size, lrelu_alpha):
    """
    Function wrapper to pruduce a single convolution block used in encoder and decoder of the network.
    :param input_tensor:    input tensor from previous layer
    :param n_filters:       number of convolution filters
    :param kernel_size:     size of convolution filters
    :param lrelu_alpha:     activation parameter
    :return:                convolution block as keras layers
    """
    # first layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size),
               padding='same',
               use_bias=True)(input_tensor)
    x = LeakyReLU(alpha=lrelu_alpha)(x)

    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size),
               padding='same',
               use_bias=True)(x)
    x = LeakyReLU(alpha=lrelu_alpha)(x)
    return x


def get_model(input_shape,
              n_filters,
              kernel_size,
              stride,

              lrelu_alpha,

              pool_size,
              dropout_rate_conv,
              last_activation):

    # Input layer
    I = Input(input_shape)
    print (I._keras_shape)

    # contracting part
    c1      = conv3d_block(input_tensor=I, n_filters=n_filters * 1, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    p1      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c1)
    p1      = Dropout(rate=dropout_rate_conv)(p1)
    print(p1._keras_shape)

    c2      = conv3d_block(input_tensor=p1, n_filters=n_filters * 2, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    p2      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c2)
    p2      = Dropout(rate=dropout_rate_conv)(p2)
    print(p2._keras_shape)

    c3      = conv3d_block(input_tensor=p2, n_filters=n_filters * 4, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    p3      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c3)
    p3      = Dropout(rate=dropout_rate_conv)(p3)
    print(p3._keras_shape)

    c4      = conv3d_block(input_tensor=p3, n_filters=n_filters * 8, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    p4      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c4)
    p4      = Dropout(rate=dropout_rate_conv)(p4)
    print(p4._keras_shape)

    c5      = conv3d_block(input_tensor=p4, n_filters=n_filters * 16, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    p5      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c5)
    p5      = Dropout(rate=dropout_rate_conv)(p5)
    print(c5._keras_shape)

    c6 = conv3d_block(input_tensor=p5, n_filters=n_filters * 32, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(c6._keras_shape)



    # expansive part
    t5 = Conv3DTranspose(filters=n_filters * 16, kernel_size=kernel_size, strides=stride, padding='same')(c6)
    t5 = LeakyReLU(alpha=lrelu_alpha)(t5)
    t5 = concatenate([t5, c5])
    u5 = Dropout(rate=dropout_rate_conv)(t5)
    t5 = conv3d_block(input_tensor=u5, n_filters=n_filters * 16, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(t5._keras_shape)

    u6 = Conv3DTranspose(filters=n_filters * 8, kernel_size=kernel_size, strides=stride, padding='same')(t5)
    u6 = LeakyReLU(alpha=lrelu_alpha)(u6)
    u6 = concatenate([u6, c4])
    u6 = Dropout(rate=dropout_rate_conv)(u6)
    c6 = conv3d_block(input_tensor=u6, n_filters=n_filters * 8, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(c6._keras_shape)
    
    u7 = Conv3DTranspose(filters=n_filters * 4, kernel_size=kernel_size, strides=stride, padding='same')(c6)
    u7 = LeakyReLU(alpha=lrelu_alpha)(u7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(rate=dropout_rate_conv)(u7)
    c7 = conv3d_block(input_tensor=u7, n_filters=n_filters * 4, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(c7._keras_shape)

    u8 = Conv3DTranspose(filters=n_filters * 2, kernel_size=kernel_size, strides=stride, padding='same')(c7)
    u8 = LeakyReLU(alpha=lrelu_alpha)(u8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(rate=dropout_rate_conv)(u8)
    c8 = conv3d_block(input_tensor=u8, n_filters=n_filters * 2, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(c8._keras_shape)

    u9 = Conv3DTranspose(filters=n_filters * 1, kernel_size=kernel_size, strides=stride, padding='same', use_bias=True)(c8)
    u9 = LeakyReLU(alpha=lrelu_alpha)(u9)
    u9 = concatenate([u9, c1], axis=4)
    u9 = Dropout(rate=dropout_rate_conv)(u9)
    c9 = conv3d_block(input_tensor=u9, n_filters=n_filters * 1, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    print(c9._keras_shape)

    outputs = Conv3D(1, (1, 1, 1), activation=last_activation)(c9)
    print(outputs._keras_shape)

    model = Model(inputs=I, outputs=outputs)
    return model

