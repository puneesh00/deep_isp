
from network import network
import tensorflow as tf
import numpy as np
import pickle
import imageio
from metrics import metrics
import argparse
from keras.utils import multi_gpu_model
from keras.applications.vgg16 import VGG16
import os
from keras.models import Model
from keras import backend as K
from load_data import extract_bayer_channels
from scipy import misc

parser = argparse.ArgumentParser()

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'epoch number for final inference')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = 'isp_learn', help = 'experiment title is used as a folder name to save respective files')
parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'weight file name to be appended while testing')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/sp-lab-2/isp_learn' , help = 'complete path for the dataset')

args = parser.parse_args()
n_epoch = args.epochs
weights_file = args.weights_file
exp_folder = args.experiment_title
dataset_dir = args.dataset_path

current_path = '/home/sp-lab-2/deep_isp_exps'

in_shape = (1488,1984,4)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (2976,3968,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
for layer in vgg.layers:
     layer.trainable = False

d_model = network(vgg, inp_shape = in_shape, trainable = False)
filename = os.path.join(current_path, exp_folder, weights_file + '_%04d.h5' % (n_epoch))   
d_model.load_weights(filename)

img_dir = dataset_dir + '/test/huawei_full_resolution/'
img_nos = [1167,1614,2169,252,2978,3163,3379,4489,5776,6645]

for img in img_nos:
	I = np.asarray(imageio.imread((img_dir + str(img) + '.png')))
	I = extract_bayer_channels(I)
	print(I.shape)
	I = np.reshape(I, [1, 1488, 1984, 4])
	out,_,_,_ = d_model.predict(I)
	out = np.reshape(out, [2976, 3968, 3])
	misc.imsave(current_path + exp_folder + str(img) + "_res.png", out)




