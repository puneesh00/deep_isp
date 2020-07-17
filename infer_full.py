
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
from load_data import load_testing_inp
from scipy import misc
from skimage.transform import resize

parser = argparse.ArgumentParser()

#parser.add_argument('-e' ,'--epoch', type = int, default = 100, help = 'epoch number for final inference')
parser.add_argument('-path' ,'--main_path', type = str, default = '/home/puneesh/deep_isp_exps' , help = 'main path where the result/experiment folders are stored')
parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'best weight file name (only prefix while evaluating)')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/puneesh/isp_learn/' , help = 'complete path for the dataset')
parser.add_argument('-res' ,'--results_folder', type = str, default = 'results' , help = 'folder to save inference results')


args = parser.parse_args()

#n_epoch = args.epoch
current_path = args.main_path
weights_file = args.weights_file
dataset_dir = args.dataset_path
res_folder = args.results_folder

os.mkdir(os.path.join(current_path,res_folder))

in_shape = (1488,1984,4)
in_shape2 = (2976,3968,3)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = in_shape2)
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
for layer in vgg.layers:
     layer.trainable = False

d_model = network(vgg, inp_shape = in_shape, trainable = False)
filename = os.path.join(current_path, weights_file)
d_model.load_weights(filename)
s = 1
raw_imgs = load_testing_inp(dataset_dir, 1488, 1984, s)
n_imgs =  raw_imgs.shape[0]

for img in range(n_imgs):
        I = raw_imgs[img,:,:,:]
        #print(I.shape)
        I = np.reshape(I, [1, 1488, 1984, 4])
        out,_,_,_,_ = d_model.predict(I)
        I = np.uint8(out*255.0)
        I = np.reshape(I, [2976,3968,3])
        misc.imsave(os.path.join(current_path, res_folder) + '/' + str((img+s)) + ".png", I)
