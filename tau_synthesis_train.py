#from __future__ import print_function

import os, h5py
import numpy as np
import tensorflow as tf


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, MaxPooling3D, Conv3DTranspose, AveragePooling3D, ZeroPadding3D
from tensorflow.keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import backend as K
from keras.regularizers import l2
from tensorflow.keras.utils import plot_model

np.random.seed(102)
tf.set_random_seed(102)

K.set_image_data_format('channels_last')

project_name = '3D-Dense-Unet'
img_rows = 128
img_cols = 128
img_depth = 128

def read_X_y_train(datapath,mat_file_name):
    os.chdir(datapath)
    mat_contents = h5py.File(mat_file_name, 'r')
    X_Train = mat_contents['X_Train']
    Y_Train = mat_contents['Y_Train']
    X_Train = np.transpose(X_Train)  # for transpose becauseof HDF matfile v7.3
    Y_Train = np.transpose(Y_Train)
    X_Val = mat_contents['X_Val']
    Y_Val = mat_contents['Y_Val']
    X_Val = np.transpose(X_Val)  # for transpose becauseof HDF matfile v7.3
    Y_Val = np.transpose(Y_Val)

    nanidx = np.isnan(X_Train)
    X_Train[nanidx] = 0
    nanidx = np.isnan(X_Val)
    X_Val[nanidx] = 0
    nanidx = np.isnan(Y_Train)
    Y_Train[nanidx] = 0
    nanidx = np.isnan(Y_Val)
    Y_Val[nanidx] = 0

    print('\tDatafile: ', datapath, mat_file_name)
    print('\tX_Train shape :', X_Train.shape)
    print('\tY_Train shape :', Y_Train.shape)
    print('\tX_Val shape :', X_Val.shape)
    print('\tY_Val shape :', Y_Val.shape)
    return X_Train,X_Val, Y_Train,Y_Val

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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    j = 1 # fold
    lru, bs, opt = 0.5, 2, 'Adam'
    lr = 0.0001
    filename = 'fdg_to_tau_optb4_fold' + str(j) + '.mat'
    codepath = '/home/m186870/data_j6/dl_data/dl_code'
    path = '/mnt/j6/m186870/dl_data/Tau_masked/decoding/'
    savepath = '/mnt/j6/m186870/dl_data/FDG_masked/decoding/fdg_to_tau_seed4_'+str(lr)+'/'
    outname = 'test_fold' + str(j) + '.mat'
    fit_iter, fit_ep = 10, 15
    try:
        os.stat(savepath)
    except:
        os.mkdir(savepath)

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    X_Train,X_Val, Y_Train,Y_Val = read_X_y_train(path,filename)
    X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], X_Train.shape[2], X_Train.shape[3], 1))
    X_Val = np.reshape(X_Val, (X_Val.shape[0], X_Val.shape[1], X_Val.shape[2], X_Val.shape[3], 1))
    Y_Train = np.reshape(Y_Train, (Y_Train.shape[0], Y_Train.shape[1], Y_Train.shape[2], Y_Train.shape[3], 1))
    Y_Val = np.reshape(Y_Val, (Y_Val.shape[0], Y_Val.shape[1], Y_Val.shape[2], Y_Val.shape[3], 1))

    X_Train = X_Train.astype('float32')
    X_Val = X_Val.astype('float32')
    Y_Train = Y_Train.astype('float32')
    Y_Val = Y_Val.astype('float32')

    os.chdir(savepath)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(img_depth, img_rows, img_cols)
    plot_model(model, to_file='model.pdf', show_shapes=True)

    for i in range(fit_iter):
        print("\t Validating setnum:", str(j), "-Training iter:", str(i + 1))
        weights_best = './weights.best_' + str(j) + 'fold' + '.h5'
        if i > 0:
            model.load_weights(weights_best)
            lr *= lru
            print('\tcurrent learning_rate : ', lr)
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                      loss='mse', metrics=['accuracy']) #'mse'

        model_checkpoint = ModelCheckpoint(weights_best, monitor='val_loss', save_best_only=True, mode='min')
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1, mode='auto')
        csv_logger = CSVLogger(os.path.join(savepath,  project_name +'_fold'+str(j)+'_iter' + str(i)+ '.txt'), separator=',', append=False)
        callbacks_list = [csv_logger, model_checkpoint, earlystop]
        print('-'*30)
        print('Fitting model...')
        print('-'*30)
        model.fit(X_Train, Y_Train, batch_size=bs, epochs=fit_ep, verbose=1, shuffle=True, callbacks=callbacks_list, validation_data=(X_Val, Y_Val))
        print('-'*30)
        print('Training finished')
        print('-'*30)
