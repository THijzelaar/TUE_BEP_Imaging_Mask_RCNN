"""
Takes a geojson file and creates a distance map from 
the polygon segmentation. The distance map is saved as a png file.
Can be normalized to a range of 0-255 instead of just the value of the distance.

Maindir = directory where the geojson files are located
save_maindir = directory where the distance maps are saved
normalize = boolean to normalize the distance map to a range of 0-255

"""
maindir = "./TUE_tijmen/"
save_maindir = './Distance Maps/'
normalize = False

import matplotlib.pyplot as plt
import simplejson as json
import glob
import os
import numpy as np
from skimage.draw import polygon
import cv2
from scipy.ndimage import distance_transform_edt 

# Loop through all the subdirectories
for subdir, _, _ in os.walk(maindir):
            # Skip the parent directory
     if subdir == maindir:
          continue
     # Loop through all the geojson files
     for file in glob.glob(subdir + '**/*.geojson', recursive=True):
          with open(file, 'r') as filepath:
               geojson_data = json.load(filepath)
               
               grid = np.zeros((1024,1024))
               # Find the coordinates of each object
               for feature in geojson_data["features"]:
                    segmentation = feature["geometry"]["coordinates"]
                    # GEOJSON also holds multipolygon which does not have a classification and holds list[list[int]] segmentation which gives errors
                    if feature["geometry"]["type"] == "Polygon" and len(segmentation[0]) > 8:
                         segmentation_flt_1 = [
                                        [list(map(float, inner_list)) for inner_list in outer_list]
                                        for outer_list in segmentation
                                   ]
                         coords = np.array(segmentation_flt_1[0])
                         # Create a mask of the polygon
                         grid = cv2.fillPoly(grid, pts = np.int32([coords]), color = 1)
               # Transform the mask to a distance map
               dmap = distance_transform_edt(grid)
               # Normalize distance map to 255
               if normalize:
                    dmap *= 255.0/np.max(dmap)
               cv2.imwrite(save_maindir+os.path.basename(subdir)+'/'+os.path.splitext(os.path.basename(file))[0]+'.png', dmap)


