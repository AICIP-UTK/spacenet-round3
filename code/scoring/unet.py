"""
Elliot Greenlee
Started 2018-01-26
University of Tennessee AICIP Spacenet Competition
"""

"""Imports"""
import os

"""Constants"""
number_of_cities = 4
vegas = 0
paris = 1
shanghai = 2
khartoum = 3

# Directory structure
repo_dir = "/scratch/spacenetProject"
data_dir = "data"
training_dir = "train"
city_training_names = ["AOI_2_Vegas_Roads_Train", "AOI_3_Paris_Roads_Train", "AOI_4_Shanghai_Roads_Train", "AOI_5_Khartoum_Roads_Train"]
city_training_dirs = city_training_names
multispectral_pansharpened_dir = "MUL-PanSharpen"
ground_truth_dir = "summaryData"
ground_truth_files = []
for i in range(0, number_of_cities):
    ground_truth_files.append(city_training_names[i] + ".csv")

def training_image_filename_id(image_id):
    return "MUL-PanSharpen_{}.tif".format(image_id)

def training_image_filename(city_const, city, image_id_num):
    return "MUL-PanSharpen_AOI_{}_{}_img{}.tif".format(city_const, city, image_id_num)
    

testing_dir = "test"
city_testing_dirs = ["AOI_2_Vegas_Roads_Test_Public", "AOI_3_Paris_Roads_Test_Public", "AOI_4_Shanghai_Roads_Test_Public", "AOI_5_Khartoum_Roads_Test_Public"]

