"""
@ Modified by Jeyeon Lee
author: Kolařík, M., Burget, R., Uher, V., Říha, K., & Dutta, M. K. (2019). 
Optimized High Resolution 3D Dense-U-Net Network for Brain and Spine Segmentation. Applied Sciences, 9(3), vol. 9, no. 3.
(See https://github.com/mrkolarik/3D-brain-segmentation)
"""

from __future__ import print_function

import os, h5py
import keras.models as models
from skimage.transform import resize
from skimage.io import imsave
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from keras.utils import plot_model
np.random.seed(1337)
tf.set_random_seed(1337)

K.set_image_data_format('channels_last')

project_name = '3D-Dense-Unet'
img_rows = 128
img_cols = 128
img_depth = 128

def read_X_y_test(datapath,mat_file_name):
    os.chdir(datapath)
    mat_contents = h5py.File(mat_file_name, 'r')
    X_Test = mat_contents['X_Test']
    Y_Test = mat_contents['Y_Test']
    X_Test = np.transpose(X_Test)  # for transpose becauseof HDF matfile v7.3
    Y_Test = np.transpose(Y_Test)

    nanidx = np.isnan(X_Test)
    X_Test[nanidx] = 0

    nanidx = np.isnan(Y_Test)
    Y_Test[nanidx] = 0

    print('\tDatafile: ', datapath, mat_file_name)
    print('\tX_Train shape :', X_Test.shape)
    print('\tY_Train shape :', Y_Test.shape)
    return X_Test,Y_Test

def get_unet(img_depth, img_rows, img_cols):
    inputs = Input((img_depth, img_rows, img_cols, 1))
    conv11 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conc11 = concatenate([inputs, conv11], axis=4)
    conv12 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc11)
    conc12 = concatenate([inputs, conv12], axis=4)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conc12)

    conv21 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conc21 = concatenate([pool1, conv21], axis=4)
    conv22 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc21)
    conc22 = concatenate([pool1, conv22], axis=4)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conc22)

    conv31 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conc31 = concatenate([pool2, conv31], axis=4)
    conv32 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc31)
    conc32 = concatenate([pool2, conv32], axis=4)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conc32)

    conv41 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conc41 = concatenate([pool3, conv41], axis=4)
    conv42 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc41)
    conc42 = concatenate([pool3, conv42], axis=4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conc42)

    conv51 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conc51 = concatenate([pool4, conv51], axis=4)
    conv52 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conc51)
    conc52 = concatenate([pool4, conv52], axis=4)

    up6 = concatenate([Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc52), conc42], axis=4)
    conv61 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    conc61 = concatenate([up6, conv61], axis=4)
    conv62 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conc61)
    conc62 = concatenate([up6, conv62], axis=4)

    up7 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc62), conv32], axis=4)
    conv71 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    conc71 = concatenate([up7, conv71], axis=4)
    conv72 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conc71)
    conc72 = concatenate([up7, conv72], axis=4)

    up8 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc72), conv22], axis=4)
    conv81 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    conc81 = concatenate([up8, conv81], axis=4)
    conv82 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conc81)
    conc82 = concatenate([up8, conv82], axis=4)

    up9 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(conc82), conv12], axis=4)
    conv91 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    conc91 = concatenate([up9, conv91], axis=4)
    conv92 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conc91)
    conc92 = concatenate([up9, conv92], axis=4)

    conv10 = Conv3D(1, (1, 1, 1), activation='linear')(conc92) # sigmoid

    model = Model(inputs=[inputs], outputs=[conv10])

    model.summary()
    #plot_model(model, to_file='model.png')
    return model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
j = 1 # fold
lr = 0.0001
#filename = 'mri_to_tau_optb4_fold' + str(j) + '.mat'
filename = 'fdg_to_tau_optb4_fold'+str(j)+'.mat'
#filename = 'mri_to_tau_ADNI_tflearning_fold'+str(j)+'.mat'
# filename = 'classification_test.mat'
codepath = '/home/m186870/data_j6/dl_data/dl_code'
path = '/mnt/j6/m186870/dl_data/Tau_masked/decoding/'
savepath = '/mnt/j6/m186870/dl_data/FDG_masked/decoding/fdg_to_tau_seed4_'+str(lr)+'/'
outname = 'test_fold' + str(j)
#outname = 'test_fold' + str(j)

lru, bs, opt = 0.5, 2, 'Adam'
fit_iter, fit_ep = 10, 15

try:
    os.stat(savepath)
except:
    os.mkdir(savepath)

print('-'*30)
print('Loading and preprocessing train data...')
print('-'*30)

X_Test,Y_Test = read_X_y_test(path,filename)
X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], X_Test.shape[2], X_Test.shape[3], 1))
Y_Test = np.reshape(Y_Test, (Y_Test.shape[0], Y_Test.shape[1], Y_Test.shape[2], Y_Test.shape[3], 1))

X_Test = X_Test.astype('float32')
Y_Test = Y_Test.astype('float32')

print('-'*30)
print('Loading saved weights...')
print('-'*30)

model = get_unet(img_depth, img_rows, img_cols)
os.chdir(savepath)
model.load_weights('weights.best_'+str(j)+'fold.h5')

print('-'*30)
print('Predicting masks on test data...')
print('-'*30)

Y_Pred = model.predict(X_Test, batch_size=2, verbose=1)
os.chdir(savepath)
outname2 = outname + 'ypred'
np.save(outname2, Y_Pred)
outname2 = outname + 'yorig'
np.save(outname2, Y_Test)
outname2 = outname + 'xtrue'
np.save(outname2, X_Test)
