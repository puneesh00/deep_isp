
from network import network as network 
from network_unet_rrdb_all_en import network as network_en
import tensorflow as tf
import numpy as np
import pickle
from keras.optimizers import Adam
import argparse
from keras.utils import multi_gpu_model
from keras.applications.vgg16 import VGG16
import os
from keras.models import Model
from keras import backend as K
from load_data import load_training_batch
from exp_fusion import *


parser = argparse.ArgumentParser()

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'number of epochs for training')
parser.add_argument('-b' ,'--batch_size', type = int, default = 32, help = 'batch size for training')
parser.add_argument('-l' ,'--log_file', type = str, default = 'log', help = 'log file name to be saved')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = 'isp_learn', help = 'experiment title is used as a folder name to save respective files')
parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'weight file name to be appended while saving')
parser.add_argument('-o' ,'--optimizer_weights', type = str, default = 'opt', help = 'optimizer file name to be appended while saving')
parser.add_argument('-lr' ,'--learning_rate', type = float, default = 0.0001, help = 'initial learning rate for the optimizer')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/puneesh/isp_learn' , help = 'complete path for the dataset')
parser.add_argument('-save' ,'--save_path', type = str, default = '/home/puneesh/deep_isp_exps' , help = 'path where weights are to be saved')
parser.add_argument('-resume_weight', '--file_to_resume_training', type = str, help = 'name of weight file to begin training from')
parser.add_argument('--resume_train', action='store_true', default = False,   help='Provide yes or no if resuming training and giving a file to resume training from')
parser.add_argument('--resume_opt', type = str, help = 'Optimizer file to begin training from')

args = parser.parse_args()
n_epochs = args.epochs
n_batch = args.batch_size
log_file = args.log_file
weights_file = args.weights_file
lr = args.learning_rate
opt_file = args.optimizer_weights
exp_folder = args.experiment_title
save_path = args.save_path
dataset_dir = args.dataset_path

resume_weight =  args.file_to_resume_training
resume_train =  args.resume_train
resume_opt = args.resume_opt

current_path = os.getcwd()
#if not resume_train:
if not os.path.exists(os.path.join(save_path, exp_folder)):
    os.mkdir(os.path.join(save_path, exp_folder))

def mssim(y_true, y_pred):
  costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
  return costs

def color(y_true, y_pred):
  #e = tf.math.scalar_mul(1e-7, tf.ones_like(y_true))
  #y_true = tf.math.add(y_true,e)
  #y_pred = tf.math.add(y_true,e)
  ytn = tf.math.l2_normalize(y_true, axis = -1, epsilon=1e-9)
  ypn = tf.math.l2_normalize(y_pred, axis = -1, epsilon=1e-9)
  color_cos = tf.einsum('aijk,aijk->aij', ytn, ypn)
  #color_cos = tf.clip_by_value(color_cos, -1, 1)
  #color_angle = tf.math.acos(color_cos)
  ca_mean = 1.0 - tf.reduce_mean(color_cos)
  return ca_mean

def vgg_loss(y_true, y_pred):
    cost = tf.reduce_mean(tf.math.square(tf.math.subtract(vgg2(y_true), vgg2(y_pred))))
    return cost

def exp_fusion(y_true, y_pred):
    #costs = tf.reduce_mean(tf.keras.losses.MAE(exp_map(y_true, 1, 1, 1), exp_map(y_pred, 1, 1, 1)))
    costs = tf.reduce_mean(tf.math.abs(tf.math.subtract(exp_map(y_true, 1, 1, 1), exp_map(y_pred, 1, 1, 1))))
    return costs

def lr_decay(lr, epoch):
    if epoch%50==0:
       lr = lr*0.8
    #print(lr)
    return lr

def train(d_par, d_model, n_epochs, n_batch, f, current_path, save_path, exp_folder, weights_file, dataset_dir):

    train_size = 5000
    bat_per_epo = int(train_size/n_batch)

    for i in range(n_epochs):

        raw, canon = load_training_batch(dataset_dir, train_size, PATCH_WIDTH = 224, PATCH_HEIGHT = 224, DSLR_SCALE = 2)

        for j in range(bat_per_epo):
            ix = np.random.randint(0, train_size, n_batch)

            X_real = canon[ix]
            X_in  = raw[ix]
            X_in,_,_,_,_ = d_model.predict(X_in)

            d_loss = d_par.train_on_batch(X_in,[X_real, X_real, X_real])#, X_real, X_real, vgg.predict(X_real)])

            #f.write('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, color=%.3f, exp_fus = %.5f, vgg=%.5f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5]))
            f.write('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, vgg=%.5f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3]))
            f.write('\n')
            #print('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, color=%.3f, exp_fus = %.5f, vgg=%.5f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4], d_loss[5]))
            print('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, vgg=%.5f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3]))
        filename = os.path.join(save_path, exp_folder, weights_file + '_%04d.h5' % (i+1))
        d_save = d_par.get_layer('model_4')
        d_save.save_weights(filename)

        K.set_value(d_par.optimizer.lr, lr_decay(K.get_value(d_par.optimizer.lr),(i+1)))

        if (i+1)%5==0:
            symbolic_weights = getattr(d_par.optimizer, 'weights')
            weight_values = K.batch_get_value(symbolic_weights)
            with open(os.path.join(save_path, exp_folder, opt_file + '_%04d.pkl' %(i+1)), 'wb') as f2:
           	 pickle.dump(weight_values, f2)
           	 f2.close()
            del symbolic_weights
            del weight_values
        del raw
        del canon
        # if (i+1) % 10 == 0:
        # summarize_performance (i, g_model, d_model, dataset)
    f.close()


in_shape = (224,224,4)
in_shape2 = (448,448,3)

base_vgg1 = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg1 = Model(inputs = base_vgg1.input, outputs = base_vgg1.get_layer('block4_pool').output)
for layer in vgg1.layers:
     layer.trainable = False

base_vgg2 = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg2 = Model(inputs=base_vgg2.input, outputs=base_vgg2.get_layer('block4_pool').output)
for layer in vgg2.layers:
     layer.trainable=False
#vgg = multi_gpu_model(vgg1, gpus = 4, cpu_relocation = True)
#vgg.summary()
file_dis = os.path.join(save_path, exp_folder, resume_weight)
d_model2 = network(vgg1, inp_shape = in_shape, trainable = False)
for layer in d_model2.layers:
     layer.trainable = False
#d_model.summary()
d_model2.load_weights(file_dis)

d_model = network_en(inp_shape = in_shape2, trainable = True)
d_par = multi_gpu_model(d_model, gpus = 4, cpu_relocation = True)
d_par.summary()
#d_par.layers[-6].set_weights(d_model.get_weights())
opt = Adam(lr = lr, beta_1 = 0.5)
#d_par.compile(loss = ['mae', mssim, color, exp_fusion, 'mse'], optimizer = opt, loss_weights = [20.0, 10.0, 5.0, 5.0, 100.0])
d_par.compile(loss = ['mae', mssim, vgg_loss], optimizer = opt, loss_weights = [5.0, 1.0, 0.1])
d_par.summary()
'''
d_par._make_train_function()
file_d_opt = os.path.join(save_path, exp_folder, resume_opt)
with open(file_d_opt, 'rb') as f3:
	weight_values = pickle.load(f3)
d_par.optimizer.set_weights(weight_values)
'''
f = open(os.path.join(save_path, exp_folder, log_file + '.txt'), 'x')
f = open(os.path.join(save_path, exp_folder, log_file + '.txt'), 'a')

train(d_par, d_model2, n_epochs, n_batch, f, current_path, save_path, exp_folder, weights_file, dataset_dir)
