from keras.layers import Conv2D, Conv2DTranspose, DepthwiseConv2D, Dense, GlobalAveragePooling2D, Reshape, Concatenate, Multiply
from keras.layers import PReLU, BatchNormalization, Add


def squeeze_excite(inp_layer, ratio):

    '''
    inp_layer : input tensor
    ratio : speaks for itself
    '''

    filters = inp_layer._keras_shape[-1]


    se = GlobalAveragePooling2D()(inp_layer)
    se = Reshape((1, 1, filters))(se)
    se = Dense(filters // ratio, activation = 'relu', kernel_initializer = 'he_normal', use_bias = True, bias_initializer = 'zeros')(se)
    se = Dense(filters, activation = 'sigmoid', kernel_initializer = 'he_normal', use_bias = True, bias_initializer = 'zeros')(se)

    se = Multiply()[inp_layer, se]

    return se



def inception(inp, f1, f13, f3, f133, f33, f5, gamma_init, trainable):

    '''
    f1 : no. of feature maps(channels) from receptive field 1x1

    f13: no. of features maps(channels) from 1x1 input to 3x3
    f3: no. of feature maps(channels) out from receptive field 3x3

    f133: no. of features maps(channels) from 1x1 input to next two 3x3s
    f33: no. of feature maps(channels) from 3x3 to 3x3
    f5: no. of feature maps(channels) out from receptive field 5x5

    inp: input tensor
    '''

    x1 = Conv2D(f1, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    x1 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x1)
    x1 = PReLU(shared_axes=[1,2])(x1)

    x3 = Conv2D(f13, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    x3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3 = PReLU(shared_axes=[1,2])(x3)
    x3 = Conv2D(f3, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x3)
    x3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3 = PReLU(shared_axes=[1,2])(x3)

    x5 = Conv2D(f133, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    x5 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x5)
    x5 = PReLU(shared_axes=[1,2])(x5)
    x5 = Conv2D(f33, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x5)
    x5 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x5)
    x5 = PReLU(shared_axes=[1,2])(x5)
    x5 = Conv2D(f5, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x5)
    x5 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x5)
    x5 = PReLU(shared_axes=[1,2])(x5)

    x5 = Concatenate(axis = -1)([x1, x3, x5])

    return x5


def rise(x, f1, f13, f3, f133, f33, f5, ratio, gamma_init, trainable):
  x1 = inception(x, f1, f13, f3, f133, f33, f5, gamma_init, trainable)
  x1 = squeeze_excite(x1, ratio)
  x1 = Add()([x1, x])
  return x1


def dil_out(x, d, m, gamma_init, trainable):

  '''
  d : no. of feature maps
  m : dilation rate
  x: input tensor
  '''

  #x=ZeroPadding2D(padding=2**m)(x)
  x=Conv2D(filters = d, kernel_size = 3, strides = 1, dilation_rate = m, padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
  x=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x)
  x=PReLU(shared_axes=[1,2])(x)
  return x

def espy(x, d, level, gamma_init, trainable):
    '''
    d : no. of feature maps
    level : no. of levels of dilation rates (1,...,level)
    x: input tensor
    '''

    x0=Conv2D(filters = d, kernel_size = 1, strides = 1, padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x0=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x0)
    x0=PReLU(shared_axes=[1,2])(x0)

    x1 = dil_out(x0, d, 1, gamma_init, trainable)
    x1c = x1

    for m in range(level-1):
      x2 = dil_out(x0, d, m+2, gamma_init, trainable)
      x2 = Add()([x1, x2])
      x1 = x2
      x2 = Concatenate(axis = -1)([x1c, x2])
      x1c = x2

    x_out = Add()([x, x2])

    return x_out

def conv(x, ch, k, s, gamma_init, trainable):
  x = Conv2D(ch, k, strides = s, padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
  x = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x)
  x = PReLU(shared_axes = [1,2])(x)
  return x

def conv_trans(x, ch, k, s, gamma_init, trainable):
  x = Conv2DTranspose(ch, k, strides = s, padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
  x = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x)
  x = PReLU(shared_axes = [1,2])(x)
  return x

def depthwise_conv(x, ch, k, s, gamma_init, trainable):
  x = DepthwiseConv2D(depth_multiplier=ch, kernel_size=k, strides = s, padding = 'same', use_bias = True, depthwise_initializer = 'he_normal', bias_initializer = 'zeros')(x)
  x = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x)
  x = PReLU(shared_axes = [1,2])(x)
  return x