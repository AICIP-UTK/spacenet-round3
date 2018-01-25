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

# Constants
width = 1300
height = 1300
white = 255
black = 0
                
def follow(jobs, image, old_location):
    # Create "linestring"
    linestring = []
    
    # Follow the line while there are still pixels 
    done = False
    while not done:
        found = 0
   
        row = old_location[0]
        column = old_location[1]
        
        # Look at the 8 pixels around the center
        for r1 in range(-1, 2, 1):
            if row == 0 and r1 == -1:
                continue
            if row == (len(image)-1) and r1 == 1:
                continue

            for c1 in range(-1, 2, 1):
                if column == 0 and c1 == -1:
                    continue
                if column == (len(image[0])-1) and c1 == 1:
                    continue
                    
                if r1 == 0 and c1 == 0:
                    continue                

                # If the pixel looked at is white
                if image[row+r1][column+c1] == white:
                    found += 1

                    # If this is the first pixel found
                    if found == 1:
                        # Add the current pixel to the linestring
                        linestring.append(old_location)

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

            # Add the current pixel to the linestring
            linestring.append(old_location)

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
print(len(linestrings))
for linestring in linestrings:
    print(len(linestring))
    for i, coordinate in enumerate(linestring):
        if i is not 0:
            cv2.line(mask,(last_coordinate[1], last_coordinate[0]),(coordinate[1], coordinate[0]),(white, white, white), thickness=1)

        last_coordinate = coordinate

cv2.imwrite('test.tif', mask)

exit(1)
# function that takes a opencv thing and returns a list of all the new linestrings

#edges = cv2.Canny(image, 50, 150, apertureSize=3)


mask = np.zeros((height, width))
lines = cv2.HoughLinesP(image, 1, (1.0/100)*(np.pi/180), 5, 10, 1)
i = 0
for x1,y1,x2,y2 in lines[0]:
    cv2.line(mask,(x1,y1),(x2,y2),(50, 50, 50),thickness=1)
    a = LineString([(x1, y1), (x2, y2)])
    for x3, y3, x4, y4 in lines[0][i+1:]:
        b = LineString([(x3, y3), (x4, y4)])
	
	shape = a.intersection(b)
	if not shape.is_empty:
            if isinstance(shape, LineString):
                #cv2.line(mask, (int(shape.coords[0][0]), int(shape.coords[0][1])), (int(shape.coords[1][0]), int(shape.coords[1][1])), (white, white, white),thickness=1)
		print("hi")
			
            elif isinstance(shape, Point):
                print(shape.coords[0])
                cv2.circle(mask, (int(shape.coords[0][0]), int(shape.coords[0][1])), 3, (white, white, white), thickness=3) 
            else:
                print("no")
    i += 1


cv2.imwrite('test.tif', mask)

exit(1)
# print linestrings
