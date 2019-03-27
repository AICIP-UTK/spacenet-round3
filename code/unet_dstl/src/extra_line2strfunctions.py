
import cv2
import numpy as np
import pandas as pd
import tifffile as tiff


def linestring2mask(image_file, df, height, width, line_width):

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
                                cv2.line(mask, last_point, point, 1, thickness=line_width)
                        last_point = point
        return mask



def make_prediction_cropped(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=19, num_masks=10):
    shift = int((initial_size[0] - final_size[0]) / 2)
    height = X_train.shape[1]
    width = X_train.shape[2]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift
    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

    padded[:, :, width + shift:] = right

    h_start = range(0, padded_height+1, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width+1, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i]

    return predicted_mask[:, :height, :width]

