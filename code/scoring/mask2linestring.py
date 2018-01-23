"""
Elliot Greenlee
2018-01-23
UTK EECS AICIP
"""

# Imports
import os
import numpy as np
import pandas as pd
import argparse
import cv2

# Open opencv thing based on file name

# function that takes a opencv thing and returns a list of all the new linestrings

# print linestrings

# Constants
height = 1300
width = 1300

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_directory', type=str, default="../../data/AOI_2_Vegas_Roads_Train/MUL-PanSharpen", help="directory where the image files are")
parser.add_argument('--csv_file', type=str, default="../../data/AOI_2_Vegas_Roads_Train/summaryData/AOI_2_Vegas_Roads_Train.csv", help="file with the linestring information")
parser.add_argument('--ground_directory', type=str, default="../../data/AOI_2_Vegas_Roads_Train/MUL-PanSharpen/ground_truth", help="directory to which ground truth files are written")
args = parser.parse_args()

# Get all image file names from the image directory
image_files = os.listdir(args.image_directory)

# Read in the csv file values
df = pd.read_csv(args.csv_file)

# Iterate through all images
for image_file in image_files:
	# Extract the linestrings for this image
        lines = df[df['ImageId'] == image_file[15:-4]]
	
	# Create a black background image
	mask = np.zeros((height, width))
	
	# Iterate over each linestring
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
				cv2.line(mask, last_point, point, (255, 255, 255), thickness=args.width)
			last_point = point
				

	# Write the mask file
	cv2.imwrite(os.path.join(args.ground_directory, "ground_truth_{}_{}.tif".format(image_file[15:-4], args.width)), mask)





