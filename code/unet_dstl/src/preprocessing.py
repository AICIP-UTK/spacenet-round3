
import os
import numpy as np
from tqdm import tqdm
import pickle
import cv2
import random
import tifffile as tiff


data_path = '/scratch/spacenetProject/data'

walkDirs2 = ['AOI_2_Vegas_Roads_Train', 'AOI_3_Paris_Roads_Train', 'AOI_4_Shanghai_Roads_Train', 'AOI_5_Khartoum_Roads_Train']
walkDirs = ['AOI_2_Vegas_Roads_Train/MUL-PanSharpen']



output = open('tr_dataID.pkl', 'wb')
flist = []
for currDir in walkDirs:
	
	print('Queuing sequences in: ' + currDir)
        for root, dirs, files in tqdm(os.walk(os.path.join(data_path, currDir))):

		for file in files:
                	print('image path: ' + file)
			flist.append(file)
random.shuffle(flist)
pickle.dump(flist, output)
output.close()


pkl_file = open('tr_dataID.pkl', 'rb')
trID = pickle.load(pkl_file)
pkl_file.close()
print('number of whole training samples=', len(trID))


trainID = trID[0:100]
print('samples for training=', trainID)

batch_size = 32
for i in range(batch_size):
        ID = trainID[i]
        print(ID)
        cityNum = ID[19]
        pathimg = os.path.join(data_path, walkDirs2[int(cityNum)-2], 'MUL-PanSharpen', ID)
#        print(pathimg)
	img = tiff.imread(pathimg).astype(np.float32)
        print(img.size)






