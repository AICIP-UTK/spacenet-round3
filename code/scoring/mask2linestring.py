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
import Queue
from shapely.geometry import LineString, Point
import sys


# Constants
width = 1300
height = 1300
white = 255
black = 0
spacing = 1


def grid(spacing):
    search = []

    for r in range(-spacing, spacing+1, 1):
        for c in range(-spacing, spacing+1, 1):
            if r == -spacing or r == spacing or c == -spacing or c == spacing:
                search.append((r, c))
    return search

# TODO: make this a list of all the searches
search = grid(spacing)

"""
spacing = 2

search = grid(spacing)

for i, line in enumerate(search):
    sys.stdout.write("{}".format(line))
"""


def follow(jobs, image, old_location):
    # Create "linestring"
    linestring = []

    old_direction = -1
    
    # Follow the line while there are still pixels 
    done = False
    while not done:
        found = 0

        row = old_location[0]
        column = old_location[1]

        # Add the current pixel to the linestring
        linestring.append(old_location)

        # Look at the 8 pixels around the center
        for direction, pixel in enumerate(search):
            r1 = pixel[0]
            c1 = pixel[1]

            # If the search goes outside the bounds
            if row+r1 < 0 or row+r1 > len(image)-1 or column+c1 < 0 or column+c1 > len(image[0])-1:
                continue          

            # If the pixel looked at is white
            if image[row+r1][column+c1] == white:
                found += 1
                new_direction = direction

                # If this is the first pixel found
                if found == 1:
                    # If the direction of travel has not changed
                    if old_direction == new_direction:
                        # Replace the last pixel in the linestring
                        linestring.pop()
   
                    old_direction = new_direction

                    # Select new pixel to investigate
                    new_location = (row+r1, column+c1)

                    # Black out the current pixel
                    row = old_location[0]
                    column = old_location[1]
                    image[row][column] = black
                    
                # If another pixel is found
                else:
                    # Add another job to search for that linestring
                    job = (image, old_location)
                    jobs.put(job)

        # If no surrounding pixels were white
        if found == 0:
            done = True

            # Black out the current pixel
            row = old_location[0]
            column = old_location[1]
            image[row][column] = black

        # Update the pixel to investigate if there is one
        if not done:
            old_location = new_location

    return linestring

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image_file', type=str, default="ground_truth_AOI_2_Vegas_img369_1.tif", help="image file to convert")
args = parser.parse_args()

# Open opencv thing based on file name
image = cv2.imread(args.image_file, -1)

# TODO: Skeletonize the image

linestrings = []

# Iterate over all pixels
for row in range(len(image)):
    for column in range(len(image[0])):        
        # If the pixel is white
        if image[row][column] == white:
            # Create empty queue of jobs
            jobs = Queue.Queue()

            # Add this pixel to the jobs queue
            old_location = (row, column)
            job = (image, old_location)
            jobs.put(job)

            while not jobs.empty():
                job = jobs.get()
                linestring = follow(jobs, job[0], job[1])
                
                # Remove all the linestrings that are just a single point
                if len(linestring) > 1:
                    linestrings.append(linestring)

# Write the linestrings to a mask to test
mask = np.zeros((height, width))
total_points = 0
for linestring in linestrings:
    total_points += len(linestring)
    for i, coordinate in enumerate(linestring):
        if i is not 0:
            cv2.line(mask,(last_coordinate[1], last_coordinate[0]),(coordinate[1], coordinate[0]),(white, white, white), thickness=1)

        last_coordinate = coordinate

print(len(linestrings))
print(total_points)
cv2.imwrite('test.tif', mask)
