"""
Elliot Greenlee
2018-01-23
UTK EECS AICIP

This program converts a given binary image into its component linestrings.
The image is read in, skeletonized, analyzed, and then those resultant linestrings are returned.
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
from skimage import morphology


# Constants
width = 1300
height = 1300
white = 255
black = 0
spacing = 1


# Generates the lists of coordinate alterations needed to search around a candidate pixel
# up to a certain spacing away from the candidate pixel
def grid(spacing):
    search = []

    for r in range(-spacing, spacing+1, 1):
        for c in range(-spacing, spacing+1, 1):
            if r == -spacing or r == spacing or c == -spacing or c == spacing:
                search.append((r, c))
    return search


# From the current pixel, follow along the white pixels and add those pixels to a line
# If multiple paths appear, create a new job
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

        # Search 1, 2, etc. pixels away until a pixel is found or the limit is reached
        for reach, search in enumerate(searches):

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

                    # If the search is looking 1 pixel away, check direction
                    if reach == 0:
                        new_direction = direction
                    # Otherwise, assume a direction change
                    else:
                        new_direction = -1

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

            # If a pixel is found at this search level, stop searching
            if found > 0:
                break

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


# Given a skeletonized image, finds the linestrings that define it
# If the linestrings have gaps in them less than or equal to spacing, they still count
def skeleton2linestrings(image, spacing):

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

    return linestrings

def mask2linestrings(image, spacing):
    searches = []

    # Create the constant search grids needed to look for pixels
    for i in range(spacing):
        searches.append(grid(i + 1))

    # Skeletonize the image
    skeletonized = morphology.medial_axis(image)
    skeletonized = skeletonized.astype(np.uint8)
    skeletonized *= 255

    # Find the linestrings from the image
    linestrings = skeleton2linestrings(skeletonized, spacing)
    return linestrings

def write_csv_predict(images, image_ids, spacing):
    with open() as csv_predict:
        csv_predict.write("ImageId,WKT_Pix\n")


        for image, image_id in zip(images, image_ids):
            linestrings = mask2linestrings(image, spacing)

            if len(linestrings) == 0:
                lines = ["{},LINESTRING EMPTY".format(image_id)]
            else:

                lines = []

                for linestring in linestrings:
                    line = "{},\"LINESTRING (".format(image_id)
                    for i, coordinate in enumerate(linestring):
                        line += "{} {}".format(coordinate[1], coordinate[0])
                        if i != (len(linestring-1)):
                            line += ", "
                        else:
                            line += ")\""

            for line in lines:
                csv_predict.write(line+"\n")
