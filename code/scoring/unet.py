"""
Elliot Greenlee
Started 2018-01-26
University of Tennessee AICIP Spacenet Competition
"""

"""Imports"""
import os
import pandas as pd
import numpy as np
import cv2
import shutil

import mask2linestring

"""Constants"""
number_of_cities = 4
vegas = 0
paris = 1
shanghai = 2
khartoum = 3
image_height = 1300
image_width = 1300

# Directory structure
repo_dir = "/scratch/spacenetProject"
data_dir = "data"
training_dir = "train"
city_training_names = ["AOI_2_Vegas_Roads_Train", "AOI_3_Paris_Roads_Train", "AOI_4_Shanghai_Roads_Train", "AOI_5_Khartoum_Roads_Train"]
city_training_dirs = city_training_names
multispectral_pansharpened_dir = "MUL-PanSharpen"
ground_truth_csv_dir = "summaryData"
ground_truth_csv_files = []
for city in range(0, number_of_cities):
    ground_truth_csv_files.append(city_training_names[city] + ".csv")

testing_dir = "test"
city_testing_dirs = ["AOI_2_Vegas_Roads_Test_Public", "AOI_3_Paris_Roads_Test_Public", "AOI_4_Shanghai_Roads_Test_Public", "AOI_5_Khartoum_Roads_Test_Public"]

ground_truth_dir = "ground_truth"
city_ground_truth_dirs = ["AOI_2_Vegas_Roads_Ground_Truth", "AOI_3_Paris_Roads_Ground_Truth", "AOI_4_Shanghai_Roads_Ground_Truth", "AOI_5_Khartoum_Roads_Ground_Truth"]

def clean_ground_truth_dirs():
    for city in range(0, number_of_cities):
        shutil.rmtree(os.path.join(repo_dir, data_dir, ground_truth_dir, city_ground_truth_dirs[city]))
        os.mkdir(os.path.join(repo_dir, data_dir, ground_truth_dir, city_ground_truth_dirs[city]))
    
def get_image_ids():
    image_ids = []
    for city_training_dir in city_training_dirs:
        city_image_names = os.listdir(os.path.join(repo_dir, data_dir, training_dir, city_training_dir, multispectral_pansharpened_dir))

        city_image_ids = []
        for city_image_name in city_image_names:
            city_image_id = city_image_name[15:-4]
            city_image_ids.append(city_image_id)

        image_ids.append(city_image_ids)
    
    return image_ids

def create_mask(ground_truth, image_id, thickness=10):
    mask = np.zeros((image_height, image_width))

    lines = ground_truth[ground_truth['ImageId'] == image_id]

    for line in lines['WKT_Pix']:
        coordinates = line[12:-1].split(', ')
        if coordinates[0] == 'MPT':
            continue
			
        for i, coordinate in enumerate(coordinates):
            point = coordinate.split(' ')
            x = int(float(point[0]))
            y = int(float(point[1]))
            point = (x, y)
            if i is not 0:
                # Draw a line from last point to point
                cv2.line(mask, last_point, point, 1, thickness=thickness)
            last_point = point

    return mask

def write_masks(image_ids, thickness=10):
    for city in range(0, number_of_cities):
        ground_truth = pd.read_csv(os.path.join(repo_dir, data_dir, training_dir, city_training_dirs[city], ground_truth_csv_dir, ground_truth_csv_files[city]))
        for image_id in image_ids[city]:
            mask = create_mask(ground_truth, image_id)
            cv2.imwrite(os.path.join(repo_dir, data_dir, ground_truth_dir, city_ground_truth_dirs[city], "{}_{}.tif".format(image_id, thickness)), mask)

def read_mask(city, image_id, thickness=10):
    mask = cv2.imread(os.path.join(repo_dir, data_dir, ground_truth_dir, city_ground_truth_dirs[city], "{}_{}.tif".format(image_id, thickness)), -1)
    return mask

def read_masks(image_ids):
    masks = {}

    for city in range(0, number_of_cities):
        for image_id in image_ids[city]:
            mask = read_mask(city, image_id)
            masks[image_id] = mask
    return masks

def store_masks(image_ids):
    masks = {}

    for city in range(0, number_of_cities):
        ground_truth = pd.read_csv(os.path.join(repo_dir, data_dir, training_dir, city_training_dirs[city], ground_truth_csv_dir, ground_truth_csv_files[city]))
        for image_id in image_ids[city]:
            mask = create_mask(image_id)
            masks[image_id] = mask
    return masks

def read_training_image(city, image_id):
    image = cv2.imread(os.path.join(repo_dir, data_dir, training_dir, city_training_dirs[city], multispectral_pansharpened_dir, "MUL-PanSharpen_{}".format(image_id)), -1)
    return image

def read_training_images(image_ids):
    images = {}

    for city in range(0, number_of_cities):
        for image_id in image_ids[city]:
            image = read_training_image(city, image_id)
            images[image_id] = image
 
    return images

image_ids = get_image_ids()

masks = read_masks(image_ids)
training_images = read_training_images(image_ids)
