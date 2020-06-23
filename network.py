from keras.models import Input, Model
from keras.layers import Add, Lambda, Multiply, Concatenate, UpSampling2D
import tensorflow as tf
from modules import *


def network(inp_shape, trainable = True, vgg):
   gamma_init = tf.random_normal_initializer(1., 0.02)

   ise = 8
   esp = 4

   f1 = 16

   f13 = 16
   f3 = 16

   f133 = 16
   f33 = 16
   f5 = 16

   ratio = 4

   level = 4
   d = 16
   n = d*level + 3

   inp = Input(inp_shape)

   x1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
   x2 = depthwise_conv(x1, 2, 3, 1, gamma_init, trainable) #ch=2*in_ch=8
   x3 = depthwise_conv(x1, 2, 5, 1, gamma_init, trainable)
   x1 = Concatenate(axis=-1)([x2,x3])


   x2 = conv_trans(inp, 8, 3, 2, gamma_init, trainable)
   x3 = conv_trans(inp, 8, 5, 2, gamma_init, trainable)

   x1 = Concatenate(axis=-1)([x1,x2,x3])   
   x1 = conv(x1, f1+f3+f5, 3, 1, gamma_init, trainable)
   x2 = x1

   for i in range(ise):
     x1 = rise(x1, f1, f13, f3, f133, f33, f5, ratio, gamma_init, trainable)

   x1 = Add()[x1,x2]
   
   x1 = conv(x1, n, 3, 1, gamma_init, trainable)

   xft = Lambda(lambda x: x[:,:,:,0:n-3], output_shape = tuple(input_shape[1:3]+[n-3]))(x1)
   ximg = Lambda(lambda x: x[:,:,:,n-3:n], output_shape = tuple(input_shape[1:3]+[3]))(x1)

   for i in range(esp):
     xft = espy(xft, d, level, gamma_init, trainable)

   xft = conv(xft, 3, 3, 1, gamma_init, trainable)
   xft = Multiply()([xft,ximg])
   x_out = Add()([xft,ximg])

   model = Model(inputs = inp, outputs = [x_out, x_out, x_out, vgg(x_out)])

   return model
