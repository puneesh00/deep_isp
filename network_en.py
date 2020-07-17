from keras.models import Input, Model
from keras.layers import Add, Lambda, Multiply, Concatenate, UpSampling2D, Activation
import tensorflow as tf
from modules import *
from keras.activations import sigmoid

def network_en(inp_shape, trainable = True, beta_tr=True):
   gamma_init = tf.random_normal_initializer(1., 0.02)

   ise = 4
   esp = 2

   f1 = 24

   f13 = 16
   f3 = 24

   f133 = 16
   f33 = 16
   f5 = 24

   ratio = 4

   level = 4
   d = 16
   n = d*level

   inp = Input(inp_shape)

   #x1 = depthwise_conv(inp, 4, 1, 1, gamma_init, trainable)
   #x2 = depthwise_conv(inp, 4, 3, 1, gamma_init, trainable) #ch=2*in_ch=8
   #x3 = depthwise_conv(inp, 4, 5, 1, gamma_init, trainable)
   #x0 = Concatenate(axis=-1)([x1,x2,x3])
   x1 = conv(inp, f1, 1, 1, gamma_init, trainable)
   x2 = conv(inp, f3, 3, 1, gamma_init, trainable)
   x3 = conv(inp, f5, 5, 1, gamma_init, trainable)
   x0 = Concatenate(axis=-1)([x1,x2,x3])
   x1 = x0

   for i in range(ise):
     x1 = rise(x1, f1, f13, f3, f133, f33, f5, ratio, gamma_init, trainable, beta_tr)

   if beta_tr:
      x0=adapwt()(x0)
   x1 = Add()([x1,x0])

   xft = conv(x1, n, 3, 1, gamma_init, trainable)
   x1 = xft
   
   for i in range(esp):
     xft = espy(xft, d, level, gamma_init, trainable, beta_tr)

   if beta_tr:
      x1=adapwt()(x1)
   xft = Add()([x1,xft])

   xft = conv(xft, 3, 3, 1, gamma_init, trainable)
   
   xin = Lambda(lambda x:2*x-1)(inp)
   x_out = Add()([xft,xin])
   x_out = Activation(sigmoid)(xft)

   model = Model(inputs = inp, outputs = [x_out, x_out])

   return model
