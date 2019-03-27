from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Cropping2D

from keras import backend as K

import keras
import h5py

from keras.layers.normalization import BatchNormalization


from keras.optimizers import Nadam
from keras.callbacks import History
import pandas as pd
from keras.backend import binary_crossentropy

import datetime
import os

import random
import threading

from keras.models import model_from_json

from tqdm import tqdm
import tifffile as tiff
import cv2
import extra_line2strfunctions
######
import pickle
data_path = '/scratch/spacenetProject/data/train'

walkDirs2 = ['AOI_2_Vegas_Roads_Train', 'AOI_3_Paris_Roads_Train', 'AOI_4_Shanghai_Roads_Train', 'AOI_5_Khartoum_Roads_Train']
walkDirs = ['AOI_2_Vegas_Roads_Train']



output = open('tr_dataID.pkl', 'wb')
flist = []
for currDir in walkDirs:

        print('Queuing sequences in: ' + currDir)
        for root, dirs, files in tqdm(os.walk(os.path.join(data_path, currDir, 'MUL-PanSharpen'))):

                for file in files:
                        if(file[0]=='M'):
                        	#print('image path: ' + file)
                        	flist.append(file)
random.shuffle(flist)
pickle.dump(flist, output)
output.close()

pkl_file = open('tr_dataID.pkl', 'rb')
trID = pickle.load(pkl_file)
pkl_file.close()
print('number of whole training samples=', len(trID))

#######



#######

img_rows = 1300
img_cols = 1300

img_unet = 256
img_unet = 256

smooth = 1e-12

num_channels = 8
num_mask_channels = 1


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


def get_unet0():
    inputs = Input((num_channels, img_unet, img_unet))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = keras.layers.advanced_activations.ELU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = keras.layers.advanced_activations.ELU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = keras.layers.advanced_activations.ELU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = keras.layers.advanced_activations.ELU()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = keras.layers.advanced_activations.ELU()(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = keras.layers.advanced_activations.ELU()(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = keras.layers.advanced_activations.ELU()(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = keras.layers.advanced_activations.ELU()(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    #crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    #conv9 = BatchNormalization(mode=0, axis=1)(crop9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = keras.layers.advanced_activations.ELU()(conv9)
    conv10 = Convolution2D(num_mask_channels, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def form_batch(X, y, batch_size):
    X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
    y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
    X_height = X.shape[2]
    X_width = X.shape[3]

    for i in range(batch_size):
        random_width = random.randint(0, X_width - img_cols - 1)
        random_height = random.randint(0, X_height - img_rows - 1)

        random_image = random.randint(0, X.shape[0] - 1)

        y_batch[i] = y[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols]
        X_batch[i] = np.array(X[random_image, :, random_height: random_height + img_rows, random_width: random_width + img_cols])
    return X_batch, y_batch


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def batch_generator(batch_size):
    #print('inside batch generator')
    while True:
        #print('inside while')
        X_batch = np.zeros((batch_size, num_channels, img_rows, img_cols))
        y_batch = np.zeros((batch_size, num_mask_channels, img_rows, img_cols))
        
	for i in range(batch_size):
                random_image = random.randint(0, len(trID) - 1)
		ID = trID[random_image]
		#print(ID)
		cityNum = ID[19]

                ###loading images
		pathimg = os.path.join(data_path, walkDirs2[int(cityNum)-2], 'MUL-PanSharpen', ID)
		img = (tiff.imread(pathimg).astype(np.float32))/65536
		#print(img.shape)
                imgt = np.transpose(img, (2, 0, 1))
                #print(imgt.shape)
                X_batch[i] = np.array(imgt)

                #####calling line2string function
                df = pd.read_csv(os.path.join(data_path, walkDirs2[int(cityNum)-2], 'summaryData', walkDirs2[int(cityNum)-2]+'.csv'))
                mask = extra_line2strfunctions.linestring2mask(ID, df, img_rows, img_cols, 20)
                #print(mask.shape)
                #print(np.max(mask))
                y_batch[i] = np.array(mask)

	#print(X_batch.shape)
        #print(y_batch.shape)

        X_batch2 = X_batch[:, :, 10:10 + img_rows - 20, 10:10 + img_cols - 20]
        y_batch2 = y_batch[:, :, 10:10 + img_rows - 20, 10:10 + img_cols - 20]

        j = random.randint(0, 4)
        k = random.randint(0, 4)
        X_batch3 = X_batch2[:, :, j*img_unet:j*img_unet+img_unet, k*img_unet:k*img_unet+img_unet]
        y_batch3 = y_batch2[:, :, j*img_unet:j*img_unet+img_unet, k*img_unet:k*img_unet+img_unet]


        #X_batch5 = np.zeros((batch_size*25, num_channels, img_unet, img_unet))
        
        #for j in range(4):
	#	for k in range(4):
	#		X_batch4 = X_batch2[:, :, j*img_unet:j*img_unet+img_unet, k*img_unet:k*img_unet+img_unet]
        #                y_batch4 = y_batch2[:, :, j*img_unet:j*img_unet+img_unet, k*img_unet:k*img_unet+img_unet]
        #                if j+k==0:
	#			X_batch5 = X_batch4
	#			y_batch5 = y_batch4
	#		else:
        #                	X_batch5 = np.concatenate([X_batch5, X_batch4], axis=0)
	#			y_batch5 = np.concatenate([y_batch5, y_batch4], axis=0)
                        #print(X_batch5.shape)
			#print(y_batch5.shape)
        #print(X_batch3.shape)
        #print(y_batch5.shape)
	yield X_batch3, y_batch3



def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model


if __name__ == '__main__':
#    data_path = '../data'
    now = datetime.datetime.now()

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))
#    f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'r')

#    X_train = f['train']

#    y_train = np.array(f['train_mask'])[:, 2]
#    y_train = np.expand_dims(y_train, 1)
#    print(y_train.shape)

#    train_ids = np.array(f['train_ids'])

    batch_size = 32
    nb_epoch = 50

    history = History()
    callbacks = [
        history,
    ]

    suffix = 'road_3_'
    model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
    model.fit_generator(batch_generator(batch_size),
                        nb_epoch=nb_epoch,
                        verbose=1,
                        samples_per_epoch=batch_size,
                        callbacks=callbacks,
                        nb_worker=8
                        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)

#    suffix = 'road_4_'
#    model.compile(optimizer=Nadam(lr=1e-4), loss=jaccard_coef_loss, metrics=['binary_crossentropy', jaccard_coef_int])
#    model.fit_generator(
#        batch_generator(X_train, y_train, batch_size, horizontal_flip=True, vertical_flip=True, swap_axis=True),
#        nb_epoch=nb_epoch,
#        verbose=1,
#        samples_per_epoch=batch_size * 400,
#        callbacks=callbacks,
#        )

#    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
#    save_history(history, suffix)
#    f.close()
