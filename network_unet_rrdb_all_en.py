from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation
from keras.layers import LeakyReLU, BatchNormalization, Lambda, UpSampling2D
from keras.activations import sigmoid
import tensorflow as tf
from modules import *

def resden(x,fil,gr,beta,gamma_init,trainable):
    x1 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    #x1 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Concatenate(axis=-1)([x,x1])

    x2 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x1)
    #x2 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)

    x2 = Concatenate(axis=-1)([x1,x2])

    x3 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x2)
    #x3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3 = LeakyReLU(alpha=0.2)(x3)

    x3 = Concatenate(axis=-1)([x2,x3])

    x4 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x3)
    #x4 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x4)
    x4 = LeakyReLU(alpha=0.2)(x4)

    x4 = Concatenate(axis=-1)([x3,x4])

    x5 = Conv2D(filters=fil,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x4)
    x5 = Lambda(lambda x:x*beta)(x5)
    xout = Add()([x5,x])

    return xout

def resresden(x,fil,gr,betad,betar,gamma_init,trainable):
    x1 = resden(x,fil,gr,betad,gamma_init,trainable)
    x2 = resden(x1,fil,gr,betad,gamma_init,trainable)
    x3 = resden(x2,fil,gr,betad,gamma_init,trainable)
    x3 = Lambda(lambda x:x*betar)(x3)
    xout = Add()([x3,x])

    return xout


def network(inp_shape, trainable = True):
   gamma_init = tf.random_normal_initializer(1., 0.02)

   fd = 64
   gr = 32
   gr2 = 32
   nb = 4
   betad = 0.2
   betar = 0.2

   inp_real_imag = Input(inp_shape)

   #ft448_0 = UpSampling2D(size=(2, 2), interpolation='bilinear')(inp_real_imag)
   #ft448_0 = DepthwiseConv2D(depth_multiplier=4, kernel_size=3, strides = 1, padding = 'same', use_bias = True, depthwise_initializer = 'he_normal', bias_initializer = 'zeros')(ft448_0)
   #ft448_0 = LeakyReLU(alpha = 0.2)(ft448_0)

   #ft448_1 = Conv2DTranspose(16, (3,3), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_real_imag)
   #ft448_1 = LeakyReLU(alpha = 0.2)(ft448_1)

   #ft448 = Concatenate(axis=-1)([ft448_0, ft448_1])
   #ft448 = convl(ft448, 32, 3, 1, gamma_init, trainable)


   ft224 = Conv2D(fd//4, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_real_imag)
   ft224 = LeakyReLU(alpha = 0.2)(ft224)

   ft112dn = convl(ft224, fd//4, 3, 2, gamma_init, trainable)
   ft112dn = resresden(ft112dn, fd//4, gr2//2, betad, betar, gamma_init, trainable)

   ft56dn = convl(ft112dn, fd//2, 3, 2, gamma_init, trainable)
   ft56dn = resresden(ft56dn, fd//2, gr2//2, betad, betar, gamma_init, trainable)

   ft28dn = convl(ft56dn, fd//2, 3, 2, gamma_init, trainable)
   ft28dn = resresden(ft28dn, fd//2, gr2//2, betad, betar, gamma_init, trainable)

   ft14dn = convl(ft28dn, fd//2, 3, 2, gamma_init, trainable)
   ft14dn = resresden(ft14dn, fd//2, gr2, betad, betar, gamma_init, trainable)
   
   ft7dn = convl(ft14dn, fd, 3, 2, gamma_init, trainable)
   ft7dn = resresden(ft7dn, fd, gr, betad, betar, gamma_init, trainable)

   xrrd = ft7dn
   for m in range(nb):
     xrrd = resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

   ft7up = Add()([xrrd,ft7dn])
   
   ft14up = convl_trans(ft7up, fd//2, 3, 2, gamma_init, trainable)
   ft14up = resresden(ft14up, fd//2, gr2, betad, betar, gamma_init, trainable)
   ft14up = Concatenate(axis=-1)([ft14up, ft14dn])
   
   ft28up = convl_trans(ft14up, fd//2, 3, 2, gamma_init, trainable)
   ft28up = resresden(ft28up, fd//2, gr2//2, betad, betar, gamma_init, trainable)
   ft28up = Concatenate(axis = -1)([ft28up,ft28dn])

   ft56up = convl_trans(ft28up, fd//2, 3, 2, gamma_init, trainable)
   #ft56up = Concatenate(axis = -1)([ft56up,ft56dn])
   ft56up = resresden(ft56up, fd//2, gr2//2, betad, betar, gamma_init, trainable)
   ft56up = Concatenate(axis = -1)([ft56up,ft56dn])

   ft112up = convl_trans(ft56up, fd//4, 3, 2, gamma_init, trainable)
   #ft112up = Concatenate(axis = -1)([ft112up,ft112dn])
   ft112up = resresden(ft112up, fd//4, gr2//2, betad, betar, gamma_init, trainable)
   ft112up = Concatenate(axis = -1)([ft112up,ft112dn])

   ft224up = convl_trans(ft112up, fd//4, 3, 2, gamma_init, trainable)
   #ft224up = Concatenate(axis = -1)([ft224up,ft224])
   ft224up = resresden(ft224up, fd//4, gr2//4, betad, betar, gamma_init, trainable)
   ft224up = Concatenate(axis = -1)([ft224up,ft224])

   #ft448up = convl_trans(ft224up, 32, 3, 2, gamma_init, trainable)
   #ft448up = Add()([ft448,ft448up])

   out = Conv2D(3, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(ft224up)
   #out = Add()([out,inp_real_imag])
   out = Activation(sigmoid)(out)
   out = Add()([out,inp_real_imag])
   model = Model(inputs = inp_real_imag, outputs = [out, out, out])

   return model
