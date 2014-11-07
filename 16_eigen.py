import Image
import os
import numpy as np
import math
from sklearn import svm
import timeit
import joblib
import sys
import random
import scipy.ndimage as ndimage
from skimage import data
from skimage import measure



def save_bw( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype='uint8'), 'L' )
    img.save( outfilename )

def listDirectory(directory, fileExtList):                                         
    # get list of file info objects for files of particular extensions
    fileList = []
    for f in os.listdir(directory):
        fileList.append(os.path.normcase(f))
    filtedFileList = []
    for f in fileList:
        if os.path.splitext(f)[1] in fileExtList:
            filtedFileList.append(f)
    return filtedFileList


def principalComponents(matrix):
    # Columns of matrix correspond to data points, rows to dimensions.
 
    deviationMatrix = (matrix.T - np.mean(matrix, axis=1)).T
    covarianceMatrix = np.cov(deviationMatrix)
    eigenvalues, principalComponents = np.linalg.eig(covarianceMatrix)
 
    # sort the principal components in decreasing order of corresponding eigenvalue
    indexList = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[indexList]
    principalComponents = principalComponents[:, indexList]
 
    return eigenvalues, principalComponents
    

dirWorking = "C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/"

imgExtList = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF"]

dirNameImg = 'C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/test_image_result2/'
dirSaveImg = 'C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/test_image_result3/'
filesImg = listDirectory(dirNameImg, imgExtList)

first = True

for i in range(len(filesImg)):
    
    print 'processing image: ' + filesImg[i]
    
    imgOrig = Image.open(dirNameImg + filesImg[i])
    imgWidth = imgOrig.size[0]
    imgHeight = imgOrig.size[1]
    
    newSize = (int(math.floor(imgWidth/4)), int(math.floor(imgHeight/4)))
    resizedOrig = imgOrig.resize(newSize, Image.ANTIALIAS)
    
    matOrig = np.asarray( resizedOrig, dtype='int32' )
    
    dimNew = 2600/4
    dim = matOrig.shape
    
    imgAdjusted = np.zeros([dimNew,dimNew])
    
    imgAdjusted[(dimNew-dim[0])/2: matOrig.shape[0] + (dimNew-dim[0])/2, (dimNew-dim[1])/2:matOrig.shape[1] + (dimNew-dim[1])/2] = matOrig
    
    imgLine = imgAdjusted.ravel()
    
    if first:
        first = False
        imgArray = imgLine
    else:
        imgArray = np.vstack((imgArray,imgLine))
    

U, s, V = np.linalg.svd(np.transpose(imgArray), full_matrices=0)

for i in range(10):
    eig = U[:,i]
    eig = ((eig-np.amin(eig)) / (np.amax(eig) - np.amin(eig))) * 255

    save_bw( eig.reshape((2600/4,2600/4)), dirSaveImg + "eig" + str(i) + ".jpg" )