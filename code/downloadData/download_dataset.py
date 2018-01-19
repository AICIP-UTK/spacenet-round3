#download all data into the original directory structure
from __future__ import print_function
import json
import pickle
import os
from tqdm import tqdm
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool


file_list = ["SpaceNet_Roads_Sample.tar.gz",
			"AOI_3_Paris_Roads_Test_Public.tar.gz", "AOI_3_Paris_Roads_Train.tar.gz",
			"AOI_4_Shanghai_Roads_Test_Public.tar.gz","AOI_4_Shanghai_Roads_Train.tar.gz",
			"AOI_5_Khartoum_Roads_Test_Public.tar.gz", "AOI_5_Khartoum_Roads_Train.tar.gz"]

file_list = ["AOI_2_Vegas_Roads_Train.tar.gz","AOI_2_Vegas_Roads_Test_Public.tar.gz"]

bucket_path = "SpaceNet_Roads_Competition/"
save_path = '/data/spacenetProject/data/'



def get_object(x):
    try:
        os.makedirs(os.path.dirname(save_path))
    except OSError:
        pass
	key = bucket_path + x
	file_name = save_path + x
    cmd = 'aws s3api get-object --request-payer requester --bucket spacenet-dataset --key %s %s' % (key, file_name)
    print(cmd)
    os.system(cmd)


print('Downloading files...')
for x in file_list:
	get_object(x)
