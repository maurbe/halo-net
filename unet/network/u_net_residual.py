from keras.models import Model
from keras.layers import Input, Dropout, Conv3D, Conv3DTranspose, concatenate, MaxPooling3D, LeakyReLU, Add

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
               use_bias=False)(input_tensor)
    x = LeakyReLU(alpha=lrelu_alpha)(x)

    # second layer
    x = Conv3D(filters=n_filters, kernel_size=(kernel_size, kernel_size, kernel_size),
               padding='same',
               use_bias=False)(x)
    # NO activation!!
    # x = LeakyReLU(alpha=lrelu_alpha)(x)
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
    import numpy as np
    # contracting part
    # shortcut via the Conv(1,1,1) trick
    con1d1  = Conv3D(filters=n_filters, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, 1, n_filters)), trainable=False)
    x1      = con1d1(I)
    c1      = conv3d_block(input_tensor=I, n_filters=n_filters * 1, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c1      = Add()([c1, x1])
    c1      = LeakyReLU(alpha=lrelu_alpha)(c1)
    p1      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c1)
    p1      = Dropout(rate=dropout_rate_conv)(p1)
    print(p1._keras_shape)

    con1d2  = Conv3D(filters=n_filters * 2, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, n_filters, n_filters * 2)), trainable=False)
    x2      = con1d2(p1)
    c2      = conv3d_block(input_tensor=p1, n_filters=n_filters * 2, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c2      = Add()([c2, x2])
    c2      = LeakyReLU(alpha=lrelu_alpha)(c2)
    p2      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c2)
    p2      = Dropout(rate=dropout_rate_conv)(p2)
    print(p2._keras_shape)

    con1d3  = Conv3D(filters=n_filters * 4, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, n_filters * 2, n_filters * 4)), trainable=False)
    x3      = con1d3(p2)
    c3      = conv3d_block(input_tensor=p2, n_filters=n_filters * 4, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c3      = Add()([c3, x3])
    c3      = LeakyReLU(alpha=lrelu_alpha)(c3)
    p3      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c3)
    p3      = Dropout(rate=dropout_rate_conv)(p3)
    print(p3._keras_shape)

    con1d4  = Conv3D(filters=n_filters * 8, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, n_filters * 4, n_filters * 8)), trainable=False)
    x4      = con1d4(p3)
    c4      = conv3d_block(input_tensor=p3, n_filters=n_filters * 8, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c4      = Add()([c4, x4])
    c4      = LeakyReLU(alpha=lrelu_alpha)(c4)
    p4      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c4)
    p4      = Dropout(rate=dropout_rate_conv)(p4)
    print(p4._keras_shape)

    con1d5  = Conv3D(filters=n_filters * 16, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, n_filters * 8, n_filters * 16)), trainable=False)
    x5      = con1d5(p4)
    c5      = conv3d_block(input_tensor=p4, n_filters=n_filters * 16, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c5      = Add()([c5, x5])
    c5      = LeakyReLU(alpha=lrelu_alpha)(c5)
    p5      = MaxPooling3D(pool_size=(pool_size, pool_size, pool_size))(c5)
    p5      = Dropout(rate=dropout_rate_conv)(p5)
    print(c5._keras_shape)

    con1d6  = Conv3D(filters=n_filters * 32, kernel_size=(1, 1, 1), padding='same', use_bias=False,
                     weights=np.ones(shape=(1, 1, 1, 1, n_filters * 16, n_filters * 32)), trainable=False)
    x6      = con1d6(p5)
    c6      = conv3d_block(input_tensor=p5, n_filters=n_filters * 32, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    c6      = Add()([c6, x6])
    c6      = LeakyReLU(alpha=lrelu_alpha)(c6)
    print(c6._keras_shape)



    # expansive part
    y5 = Conv3DTranspose(filters=n_filters * 16, kernel_size=kernel_size, strides=stride, padding='same')(c6)
    u5 = LeakyReLU(alpha=lrelu_alpha)(y5)
    u5 = concatenate([u5, c5])
    # u5 = Dropout(rate=dropout_rate_conv)(t5)
    u5 = conv3d_block(input_tensor=u5, n_filters=n_filters * 16, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    t5 = Add()([u5, y5])
    t5 = LeakyReLU(alpha=lrelu_alpha)(t5)
    print(t5._keras_shape)

    y6 = Conv3DTranspose(filters=n_filters * 8, kernel_size=kernel_size, strides=stride, padding='same')(t5)
    u6 = LeakyReLU(alpha=lrelu_alpha)(y6)
    u6 = concatenate([u6, c4])
    # u6 = Dropout(rate=dropout_rate_conv)(u6)
    u6 = conv3d_block(input_tensor=u6, n_filters=n_filters * 8, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    t6 = Add()([u6, y6])
    t6 = LeakyReLU(alpha=lrelu_alpha)(t6)
    print(t6._keras_shape)
    
    y7 = Conv3DTranspose(filters=n_filters * 4, kernel_size=kernel_size, strides=stride, padding='same')(t6)
    u7 = LeakyReLU(alpha=lrelu_alpha)(y7)
    u7 = concatenate([u7, c3])
    u7 = Dropout(rate=dropout_rate_conv)(u7)
    u7 = conv3d_block(input_tensor=u7, n_filters=n_filters * 4, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    t7 = Add()([u7, y7])
    t7 = LeakyReLU(alpha=lrelu_alpha)(t7)
    print(t7._keras_shape)

    y8 = Conv3DTranspose(filters=n_filters * 2, kernel_size=kernel_size, strides=stride, padding='same')(t7)
    u8 = LeakyReLU(alpha=lrelu_alpha)(y8)
    u8 = concatenate([u8, c2])
    u8 = Dropout(rate=dropout_rate_conv)(u8)
    u8 = conv3d_block(input_tensor=u8, n_filters=n_filters * 2, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    t8 = Add()([u8, y8])
    t8 = LeakyReLU(alpha=lrelu_alpha)(t8)
    print(t8._keras_shape)

    y9 = Conv3DTranspose(filters=n_filters * 1, kernel_size=kernel_size, strides=stride, padding='same', use_bias=True)(t8)
    u9 = LeakyReLU(alpha=lrelu_alpha)(y9)
    u9 = concatenate([u9, c1], axis=4)
    u9 = Dropout(rate=dropout_rate_conv)(u9)
    u9 = conv3d_block(input_tensor=u9, n_filters=n_filters * 1, kernel_size=kernel_size, lrelu_alpha=lrelu_alpha)
    t9 = Add()([u9, y9])
    t9 = LeakyReLU(alpha=lrelu_alpha)(t9)
    print(t9._keras_shape)

    outputs = Conv3D(1, (1, 1, 1), activation=last_activation)(t9)
    print(outputs._keras_shape)

    model = Model(inputs=I, outputs=outputs)
    return model

