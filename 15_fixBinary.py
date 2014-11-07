from PIL import Image
import numpy as np
import scipy.ndimage as ndimage
import math
import os
import joblib
import timeit
import sys

def save_bw( npdata, outfilename ) :
	img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype='uint8'), 'L' )
	img.save( outfilename )

dirWorking = "C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/"
dirSaveImg = 'C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/test_image_result2/'

imgMat = joblib.load(dirSaveImg + '1-1_5_31113_mc.jpg.pkl')

filled = ndimage.morphology.binary_fill_holes(imgMat).astype(int)

labeled_array, num_features = ndimage.measurements.label(filled)

sizes = ndimage.sum(filled,labeled_array,range(1,num_features+1))

map = np.where(sizes==sizes.max())[0] + 1 

# inside the largest, respecitively the smallest labeled patches with values
max_index = np.zeros(num_features + 1, np.uint8)
max_index[map] = 1
max_feature = max_index[labeled_array]

eroded = ndimage.binary_erosion(max_feature).astype(max_feature.dtype)



filled = ndimage.morphology.binary_fill_holes(eroded).astype(int)

labeled_array, num_features = ndimage.measurements.label(filled)

sizes = ndimage.sum(filled,labeled_array,range(1,num_features+1))

map = np.where(sizes==sizes.max())[0] + 1 

# inside the largest, respecitively the smallest labeled patches with values
max_index = np.zeros(num_features + 1, np.uint8)
max_index[map] = 1
max_feature = max_index[labeled_array]


max_feature = max_feature * 255

save_bw( max_feature, dirSaveImg + '1-1_5_31113_mc4.png' )