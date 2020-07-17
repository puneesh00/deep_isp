
#from network_unet_rrdb_all_nobn import network
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
from load_data import load_testing_data, load_testing_inp
import time
from scipy import misc

parser = argparse.ArgumentParser()

parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'best weight file name (only prefix while evaluating)')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/puneesh/isp_learn' , help = 'complete path for the dataset')
parser.add_argument('-path' ,'--main_path', type = str, default = '/home/puneesh/deep_isp_exps' , help = 'main path where the result/experiment folders are stored')
parser.add_argument('-res' ,'--results_folder', type = str, default = 'results' , help = 'folder to save inference results')

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'number of epochs for testing for eval mode')
parser.add_argument('-gt' ,'--gt_avail', default = False, action='store_true' , help = 'ground truth images available or not')
parser.add_argument('-eval' ,'--eval_mode', default = False, action='store_true' , help = 'evaluating all epochs or only best epoch')
parser.add_argument('-m' ,'--metrics_file', type = str, default = 'metrics', help = 'metrics file name to be saved')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = 'isp_learn', help = 'experiment folder name to save respective files')


args = parser.parse_args()
n_epochs = args.epochs
metrics_file = args.metrics_file
exp_folder = args.experiment_title
gt = args.gt_avail
eval_mode = args.eval_mode

weights_file = args.weights_file
dataset_dir = args.dataset_path
current_path = args.main_path
res_folder = args.results_folder

os.mkdir(os.path.join(current_path,res_folder))

in_shape = (224,224,4)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
for layer in vgg.layers:
     layer.trainable = False

d_model = network(vgg, inp_shape = in_shape, trainable = False)

if eval_mode:
   raw_imgs, canon_imgs = load_testing_data(dataset_dir, 224, 224, 2)

   f = open(os.path.join(current_path, exp_folder, metrics_file + '.txt'), 'x')
   f = open(os.path.join(current_path, exp_folder, metrics_file + '.txt'), 'a')

   for i in range(n_epochs):
      filename = os.path.join(current_path, exp_folder, weights_file + '_%04d.h5' % (i+1))   
      d_model.load_weights(filename)
      out,_,_,_,_ = d_model.predict(raw_imgs)
      psnr, ssim = metrics(canon_imgs, out, 1.0)
      f.write('%.1f psnr = %.5f, ssim = %.7f' %(i+1, psnr, ssim))
      f.write('\n')
      print(psnr, ssim)
else:
   if gt:
      raw_imgs, canon_imgs = load_testing_data(dataset_dir, 224, 224, 2)
   else:
      raw_imgs = load_testing_inp(dataset_dir, 224, 224)

   filename = os.path.join(current_path, weights_file)
   d_model.load_weights(filename)

   t1=time.time()
   out,_,_,_,_ = d_model.predict(raw_imgs)
   t2=time.time()
   t = (t2-t1)/raw_imgs.shape[0]
   print(t)
   if gt:
      psnr, ssim = metrics(canon_imgs, out, 1.0)
      print(psnr, ssim)

   for i in range(out.shape[0]):
      I = np.uint8(out[i,:,:,:]*255.0)
      #print(I.shape)
      misc.imsave(os.path.join(current_path, res_folder) + '/' +  str(i) + '.png', I)
