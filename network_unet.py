from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation
from keras.layers import LeakyReLU, BatchNormalization, Lambda
from keras.activations import sigmoid
import tensorflow as tf

def resden(x,fil,gr,beta,gamma_init,trainable):
    x1 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x1 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x1)
    x1 = LeakyReLU(alpha=0.2)(x1)

    x1 = Concatenate(axis=-1)([x,x1])

    x2 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x1)
    x2 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x2)
    x2 = LeakyReLU(alpha=0.2)(x2)

    x2 = Concatenate(axis=-1)([x1,x2])

    x3 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x2)
    x3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3 = LeakyReLU(alpha=0.2)(x3)

    x3 = Concatenate(axis=-1)([x2,x3])

    x4 = Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x3)
    x4 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x4)
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

   fd = 256
   gr = 32
   nb = 8
   betad = 0.2
   betar = 0.2

   inp_real_imag = Input(inp_shape)
   lay_256 = Conv2D(32, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_real_imag)

   lay_256 = LeakyReLU(alpha = 0.2)(lay_256)

   lay_128dn = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_256)
   lay_128dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_128dn)
   lay_128dn = LeakyReLU(alpha = 0.2)(lay_128dn)

   lay_64dn = Conv2D(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128dn)
   lay_64dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64dn)
   lay_64dn = LeakyReLU(alpha = 0.2)(lay_64dn)

   lay_32dn = Conv2D(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64dn)
   lay_32dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32dn)
   lay_32dn = LeakyReLU(alpha=0.2)(lay_32dn)

   lay_16dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32dn)
   lay_16dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16dn)
   lay_16dn = LeakyReLU(alpha=0.2)(lay_16dn)  #16x16

   #lay_8dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16dn)
   #lay_8dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_8dn)
   #lay_8dn = LeakyReLU(alpha=0.2)(lay_8dn) #8x8
   lay_8dn = lay_16dn

   xc1 = Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8dn) #8x8
   xrrd = xc1
   for m in range(nb):
     xrrd = resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

   xc2 = Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(xrrd)
   lay_8upc = Add()([xc1,xc2])

   #lay_16up = Conv2DTranspose(1024, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8upc) # confirm size wuth code, my guess is they are increasing size by 2 in every spatial dimension.
   #lay_16up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16up)
   #lay_16up = Activation('relu')(lay_16up) #16x16
   lay_16upc = lay_8upc
   #lay_16upc = Concatenate(axis = -1)([lay_16up,lay_16dn])

   lay_32up = Conv2DTranspose(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16upc)
   lay_32up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32up)
   lay_32up = Activation('relu')(lay_32up) #32x32

   lay_32upc = Concatenate(axis = -1)([lay_32up,lay_32dn])

   lay_64up = Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32upc)
   lay_64up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64up)
   lay_64up = Activation('relu')(lay_64up) #64x64

   lay_64upc = Concatenate(axis = -1)([lay_64up,lay_64dn])

   lay_128up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64upc)
   lay_128up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_128up)
   lay_128up = Activation('relu')(lay_128up) #128x128

   lay_128upc = Concatenate(axis = -1)([lay_128up,lay_128dn])

   lay_256up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128upc)
   lay_256up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_256up)
   lay_256up = Activation('relu')(lay_256up) #256x256
   
   lay_256upc = Concatenate(axis = -1)([lay_256up,lay_256])

   lay_512up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_256upc)
   lay_512up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_512up)
   lay_512up = Activation('relu')(lay_512up) #512x512
   #out =  Conv2D(1, (1,1), strides = (1,1), activation = 'tanh', padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_256up)
   out = Conv2D(3, (4,4), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_512up)
   out = Activation(sigmoid)(out)
   
   model = Model(inputs = inp_real_imag, outputs = [out, out, out, out, out])

   return model
