from __future__ import division

import os
from tqdm import tqdm
import pandas as pd

import shapely.geometry
from numba import jit

from keras.models import model_from_json
import numpy as np

from tqdm import tqdm
import tifffile as tiff
import cv2
import extra_line2strfunctions
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#########
import pickle
data_path = '/scratch/spacenetProject/data/test'

walkDirs2 = ['AOI_2_Vegas_Roads_Test_Public', 'AOI_3_Paris_Roads_Test_Public', 'AOI_4_Shanghai_Roads_Test_Public', 'AOI_5_Khartoum_Roads_Test_Public']
walkDirs = ['AOI_2_Vegas_Roads_Test_Public']



output = open('te_dataID.pkl', 'wb')
flist = []
for currDir in walkDirs:

        print('Queuing sequences in: ' + currDir)
        for root, dirs, files in tqdm(os.walk(os.path.join(data_path, currDir, 'MUL-PanSharpen'))):

                for file in files:
                        if(file[0]=='M'):
                                #print('image path: ' + file)
                                flist.append(file)
pickle.dump(flist, output)
output.close()

pkl_file = open('te_dataID.pkl', 'rb')
teID = pickle.load(pkl_file)
pkl_file.close()
print('number of whole testing samples=', len(teID))

#########
img_rows = 1300
img_cols = 1300


def read_model(cross=''):
    json_name = 'architecture_8_20_road_3_' + cross + '.json'
    weight_name = 'model_weights_8_20_road_3_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model

model = read_model()

#sample = pd.read_csv('../data/sample_submission.csv')

#data_path = '../data'
num_channels = 8
num_mask_channels = 1
threshold = 0.9

#three_band_path = os.path.join(data_path, 'three_band')

#train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
#gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
#shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

#test_ids = shapes.loc[~shapes['image_id'].isin(train_wkt['ImageId'].unique()), 'image_id']

result = []


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


#@jit
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threashold, epsilon=0, min_area=1000)

    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons)


for image_id in teID:
    print(image_id)
    #image = extra_functions.read_image_16(image_id)
    cityNum = image_id[19]
    pathimg = os.path.join(data_path, walkDirs2[int(cityNum)-2], 'MUL-PanSharpen', image_id)
    img = (tiff.imread(pathimg).astype(np.float32))/65536
    imgt = np.transpose(img, (2, 0, 1))
    imgc = imgt[:, 10:10 + img_rows - 20, 10:10 + img_cols - 20]
    #H = image.shape[1]
    #W = image.shape[2]
    H = img_rows
    W = img_cols

    #x_max, y_min = extra_functions._get_xmax_ymin(image_id)

    #predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
    #                                                         final_size=(112-32, 112-32),
    #                                                         num_masks=num_mask_channels, num_channels=num_channels)

    predicted_mask = extra_line2strfunctions.make_prediction_cropped(model, imgc, initial_size=(256, 256),
                                                             final_size=(256, 256),
                                                             num_masks=num_mask_channels, num_channels=num_channels)


    #image_v = flip_axis(image, 1)
    #predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
    #                                                           final_size=(112 - 32, 112 - 32),
    #                                                           num_masks=1,
    #                                                           num_channels=num_channels)

    #image_h = flip_axis(image, 2)
    #predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
    #                                                           final_size=(112 - 32, 112 - 32),
    #                                                           num_masks=1,
    #                                                           num_channels=num_channels)

    #image_s = image.swapaxes(1, 2)
    #predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
    #                                                           final_size=(112 - 32, 112 - 32),
    #                                                           num_masks=1,
    #                                                           num_channels=num_channels)

    #new_mask = np.power(predicted_mask *
    #                    flip_axis(predicted_mask_v, 1) *
    #                    flip_axis(predicted_mask_h, 2) *
    #                    predicted_mask_s.swapaxes(1, 2), 0.25)

    #x_scaler, y_scaler = extra_functions.get_scalers(H, W, x_max, y_min)

    #mask_channel = 2
    #result += [(image_id, mask_channel + 1, mask2poly(new_mask, threashold, x_scaler, y_scaler))]

    pm = predicted_mask
    print(pm)
    pmt = pm[0]>threshold
    print(pmt.shape)
    print(pmt)
    pmt2 = pmt.astype(int)
    print(pmt2)

    np.save('out.npy', pmt2)
    scipy.misc.imsave('out.bmp', pmt2)

    #pmt2.dtype = 'uint8'
    #print(np.sum(pmt2))

    #pmt2 = int(pmt2 == 'True')
    #print(pmt2)
    #print(pmt2.shape)
    #np.save('out.npy', pmt2) 
    #scipy.misc.imsave('out.bmp', pmt2)    

    #pmt2 = pmt2.astype(np.uint8)

    #plt.imsave('image.png', pmt2, cmap=plt.cm.gray)
    #scipy.misc.imsave('image2.bmp', pmt2)

    #out_mask2.dtype = 'uint8'
    #cv2.imwrite('image22.png', pmt2)

#submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])


#sample = sample.drop('MultipolygonWKT', 1)
#submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')

#submission.to_csv('temp_road_4.csv', index=False)
