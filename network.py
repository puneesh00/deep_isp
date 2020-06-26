from keras.models import Input, Model
from keras.layers import Add, Lambda, Multiply, Concatenate, UpSampling2D, Activation
import tensorflow as tf
from modules import *
from keras.activations import sigmoid

def network(vgg, inp_shape, trainable = True):
   gamma_init = tf.random_normal_initializer(1., 0.02)

   ise = 8
   esp = 6

   f1 = 16*2

   f13 = 16*2
   f3 = 16*2

   f133 = 16*2
   f33 = 16*2
   f5 = 16*2

   ratio = 4

   level = 4
   d1 = 16*2
   n1 = d1*level + 12

   d2 = 16*4
   n2 = d2*level

   inp = Input(inp_shape)

   '''
   x1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
   x2 = depthwise_conv(x1, 2, 3, 1, gamma_init, trainable) #ch=2*in_ch=8
   x3 = depthwise_conv(x1, 2, 5, 1, gamma_init, trainable)
   x1 = Concatenate(axis=-1)([x2,x3])


   x2 = conv_trans(inp, 8, 3, 2, gamma_init, trainable)
   x3 = conv_trans(inp, 8, 5, 2, gamma_init, trainable)

   x1 = Concatenate(axis=-1)([x1,x2,x3])
   '''
   x1 = conv(inp, f1+f3+f5, 3, 1, gamma_init, trainable)
   x2 = x1

   for i in range(ise):
     x1 = rise(x1, f1, f13, f3, f133, f33, f5, ratio, gamma_init, trainable)

   x1 = Add()([x1,x2])

   x1 = conv(x1, n1, 3, 1, gamma_init, trainable)

   xft = crop(0,n1-12)(x1) #xft = Lambda(lambda x: x[:,:,:,0:n-3], output_shape = (input_shape[1],input_shape[2],)+[n-3] )(x1)
   ximg = crop(n1-12,n1)(x1) #ximg = Lambda(lambda x: x[:,:,:,n-3:n], output_shape = tuple(input_shape[1:3]+[3]))(x1)
   ximg = SubpixelConv2D()(ximg)

   xft = conv(xft, n2, 3, 2, gamma_init, trainable) #reduce size to 112x112
   for i in range(esp):
     xft = espy(xft, d2, level, gamma_init, trainable)

   xft = conv(xft, 96, 3, 1, gamma_init, trainable)
   w1 = crop(0,48)(xft) #112x112x48 for linear terms
   w2 = crop(48,96)(xft) #112x112x48 for quad. terms
   w1 = SubpixelConv2D(scale = 4)(w1) #448x448x3
   w2 = SubpixelConv2D(scale = 4)(w2) #448x448x3
   xft = Multiply()([w1, ximg])
   img_sq = Multiply()([ximg, ximg])
   img_sq = Multiply()([img_sq, w2])
   x_out = Add()([xft, ximg]) # w1*img + img
   x_out = Add()([x_out, img_sq]) # w2*img**2 + w1*img + img
   x_out = Activation(sigmoid)(x_out)

   model = Model(inputs = inp, outputs = [x_out, x_out, x_out, x_out, vgg(x_out)])

   return model
