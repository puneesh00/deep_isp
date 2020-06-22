import network as network
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


parser = argparse.ArgumentParser()

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'number of epochs for training')
parser.add_argument('-b' ,'--batch_size', type = int, default = 32, help = 'batch size for training')
parser.add_argument('-l' ,'--log_file', type = str, default = log, help = 'log file name to be saved')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = isp_learn, help = 'experiment title is used as a folder name to save respective files')
parser.add_argument('-w' ,'--weights_file', type = str, default = weights , help = 'weight file name to be appended while saving')
parser.add_argument('-o' ,'--optimizer_weights', type = str, default = opt, help = 'optimizer file name to be appended while saving')
parser.add_argument('-lr' ,'--learning_rate', type = float, default = 0.0001, help = 'initial learning rate for the optimizer')

args = parser.parse_args()
n_epochs = args.epochs
n_batch = args.batch_size
log_file = args.log_file
weights_file = args.weights_file
lr = args.learning_rate
opt_file = args.optimizer_weights
exp_folder = args.experiment_title

current_path = os.getcwd()
os.mkdir(os.path.join(current_path, exp_folder))

def mssim(y_true, y_pred):
  costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
  return costs

def color(y_true, y_pred):
  ytn = K.l2_normalize(y_true, axis = -1)
  ypn = K.l2_normalize(y_pred, axis = -1)
  color_cos = tf.einsum('...i,...i->...', ytn, ypn)
  color_angle = tf.math.acos(color_cos)
  ca_mean = tf.reduce_mean(color_angle)
  return ca_mean

def train(d_par, d_model, vgg, n_epochs, n_batch, f, current_path, exp_folder, weights_file):

    train_size = 5000
    bat_per_epo = int(train_size/n_batch)
    dataset_dir = exp_folder

    for i in range(n_epochs):

        raw, canon = load_training_batch(dataset_dir, train_size, PATCH_WIDTH = 224, PATCH_HEIGHT = 224, DSLR_SCALE = 1)

        for j in range(bat_per_epo):
            ix = np.random.randint(0, train_size, n_batch)

            X_real = canon[ix]
            X_in  = raw[ix]

            d_loss = d_par.train_on_batch(X_in,[X_real, X_real, X_real, vgg.predict(X_real)])

            f.write('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, color=%.3f, vgg=%.3f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4]))
            f.write('\n')
            print('>%d, %d/%d, d=%.3f, mae=%.3f,  mssim=%.3f, color=%.3f, vgg=%.3f' %(i+1, j+1, bat_per_epo, d_loss[0], d_loss[1], d_loss[2], d_loss[3], d_loss[4]))
        filename = os.path.join(current_path, exp_folder, weights_file + '_%04d.h5' % (i+1))
        d_save = d_par.get_layer('model_3')
        d_save.save_weights(filename)
        del raw
        del canon
        # if (i+1) % 10 == 0:
        # summarize_performance (i, g_model, d_model, dataset)
    f.close()


in_shape = (224,224,4)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (224,224,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
vgg.summary()

d_model = network(inp_shape = in_shape, trainable = True, vgg)
d_model.summary()
d_par = multi_gpu_model(d_model, gpus = 4, cpu_relocation = True)
opt = Adam(lr = lr, beta_1 = 0.5)
d_par.compile(loss = ['mae', mssim, color, 'mse'], optimizer = opt, loss_weights = [0.01, 20.0, 1.0, 10.0])

f = open(os.path.join(current_path, exp_folder, log_file + '.txt'), 'x')
f = open(os.path.join(current_path, exp_folder, log_file + '.txt'), 'a')

train(d_par, d_model, vgg, n_epochs, n_batch, f, current_path, exp_folder, weights_file)
