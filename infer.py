
from network import network
import tensorflow as tf
import numpy as np
import pickle
from metrics import metrics
from keras.optimizers import Adam
import argparse
from keras.utils import multi_gpu_model
from keras.applications.vgg16 import VGG16
import os
from keras.models import Model
from keras import backend as K
from load_data import load_testing_data

parser = argparse.ArgumentParser()

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'number of epochs for testing')
parser.add_argument('-m' ,'--metrics_file', type = str, default = 'metrics', help = 'metrics file name to be saved')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = 'isp_learn', help = 'experiment title is used as a folder name to save respective files')
parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'weight file name to be appended while testing')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/puneesh/isp_learn' , help = 'complete path for the dataset')

args = parser.parse_args()
n_epochs = args.epochs
metrics_file = args.metrics_file
weights_file = args.weights_file
exp_folder = args.experiment_title
dataset_dir = args.dataset_path

current_path = os.getcwd()

in_shape = (224,224,4)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
for layer in vgg.layers:
     layer.trainable = False

d_model = network(vgg, inp_shape = in_shape, trainable = False)

raw_imgs, canon_imgs = load_testing_data(dataset_dir, 224, 224, 2)

f = open(os.path.join(current_path, exp_folder, metrics_file + '.txt'), 'x')
f = open(os.path.join(current_path, exp_folder, metrics_file + '.txt'), 'a')

for i in range(n_epochs):
   filename = os.path.join(current_path, exp_folder, weights_file + '_%04d.h5' % (i+1))   
   d_model.load_weights(filename)
   out,_,_,_ = d_model.predict(raw_imgs)
   psnr, ssim = metrics(canon_imgs, out, 1.0)
   f.write('psnr = %.5f, ssim = %.7f' %(psnr, ssim))
   f.write('\n')
   print(psnr, ssim)