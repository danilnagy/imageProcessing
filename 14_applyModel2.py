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

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype='int32' )
    return data

def save_bw( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype='uint8'), 'L' )
    img.save( outfilename )

def save_bw_map( npdata, outfilename ) :
    matMax = np.amax(npdata)
    matMin = np.amin(npdata)
    
    npdata = (npdata - matMin) * (255/matMax)
    
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype='uint8'), 'L' )
    img.save( outfilename )

def save_rgb( npdata, outfilename ) :
    img = Image.fromarray( np.uint8(npdata), 'RGB' )
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


dirWorking = "C:/Users/Danil/Documents/ADVP/Lars Dietrich Lab/pythonProcessFinal/"

imgExtList = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF"]

dirNameImg = dirWorking + 'test_image/'
dirSaveImg = dirWorking + 'test_image_result4/'
filesImg = listDirectory(dirNameImg, imgExtList)

wSize = 10

# load training model
print 'load model...'
startTime = timeit.default_timer()
clf = joblib.load(dirWorking + 'model.pkl')
print 'clf loaded (' + str((timeit.default_timer() - startTime)) + ' seconds)'


count = 0
for i in range(len(filesImg[:2])):
    
    print 'rebuilding training image ' + filesImg[i]
    
    imgOrig = Image.open(dirNameImg + filesImg[i])
    matOrig = np.asarray( imgOrig, dtype='int32' )
    matOrig = matOrig[:,:,1]
    
    imgWidth = imgOrig.size[0]
    imgHeight = imgOrig.size[1]
    
    print imgWidth
    print imgHeight
    
    newSize = (int(math.floor(imgWidth/4)), int(math.floor(imgHeight/4)))
    resizedOrig = imgOrig.resize(newSize, Image.ANTIALIAS)
    
    matrixOrig = np.asarray( resizedOrig, dtype='int32' )
    
    bwOrig = matrixOrig[:,:,1]
    dim = bwOrig.shape
    
    ySize = int(math.floor(dim[0]/wSize))
    xSize = int(math.floor(dim[1]/wSize))
    
    
    print 'xSize = ' + str(xSize)
    print 'ySize = ' + str(ySize)
    
    
    count = 0;
    for y in range(ySize):
        for x in range(xSize):
            imgNew = bwOrig[y*wSize:(y+1)*wSize, x*wSize:(x+1)*wSize]
            
            # convert imgNew to single line
            newLine = imgNew.flatten()
            
            # add a line to dataset (features)
            if count == 0:
                XTest = newLine
            else:
                XTest = np.vstack((XTest,newLine))
                
            #save_bw(imgNew, 'img' + str(count+1) + '.png')
            count += 1
    
    
    imgMat = np.zeros((ySize,xSize))
    
    # rebuild image from predictions
    count = 0;
    for y in range(ySize):
        for x in range(xSize):
            pred = clf.predict([XTest[count,:]])
            pred = int(pred[0])
            
            if pred == 1:
                #imgThresh[x*wSize:(x+1)*wSize, y*wSize:(y+1)*wSize] = 126
                if random.random() < .5:
                    imgMat[y,x] = 1
            elif pred == 2:
                #imgThresh[x*wSize:(x+1)*wSize, y*wSize:(y+1)*wSize] = 255
                imgMat[y,x] = 1
            
            count += 1
            
            
    
    #binary manipulation code
    filled = ndimage.morphology.binary_fill_holes(imgMat).astype(int)

    labeled_array, num_features = ndimage.measurements.label(filled)
    
    sizes = ndimage.sum(filled,labeled_array,range(1,num_features+1))
    
    map = np.where(sizes==sizes.max())[0] + 1 
    
    # inside the largest, respecitively the smallest labeled patches with values
    max_index = np.zeros(num_features + 1, np.uint8)
    max_index[map] = 1
    max_feature = max_index[labeled_array]
    
    
    eroded = ndimage.binary_erosion(max_feature).astype(max_feature.dtype)
    eroded = ndimage.morphology.binary_opening(max_feature).astype(max_feature.dtype)    
    
    
    filled = ndimage.morphology.binary_fill_holes(eroded).astype(int)
    
    labeled_array, num_features = ndimage.measurements.label(filled)
    
    sizes = ndimage.sum(filled,labeled_array,range(1,num_features+1))
    map = np.where(sizes==sizes.max())[0] + 1 
    
    # inside the largest, respecitively the smallest labeled patches with values
    max_index = np.zeros(num_features + 1, np.uint8)
    max_index[map] = 1
    max_feature = max_index[labeled_array]
    
    center = ndimage.measurements.center_of_mass(max_feature)
    center = (int(center[0]*wSize*4), int(center[1]*wSize*4))
    
    print center
    
    imgThresh = np.zeros(dim)
    #imgThresh = np.zeros((imgWidth, imgHeight))
    #wSize = wSize * 4
                   
    for y in range(ySize):
        for x in range(xSize):
            
            pred = max_feature[y,x]
            
            if pred == 1:
                imgThresh[y*wSize:(y+1)*wSize, x*wSize:(x+1)*wSize] = 1
    
    fac = 15
    
    a,b = fac,fac
    n = fac*2+1
    r = fac
    
    y,x = np.ogrid[-a:n-a, -b:n-b]
    mask = x*x + y*y <= r*r
    
    struct = np.zeros((n,n))
    struct[mask] = 1
    
    imgThresh = ndimage.morphology.binary_opening(imgThresh, structure=struct).astype(np.int)
    imgThresh = ndimage.morphology.binary_closing(imgThresh, structure=struct).astype(np.int)
    
    
    imgThresh = np.repeat(np.repeat(imgThresh,4, axis=0), 4, axis=1)
    #imgThresh = imgThresh.resize((imgHeight, imgWidth))
    
    imgMask = np.zeros((imgHeight, imgWidth))
    imgMask[0:imgThresh.shape[0], 0:imgThresh.shape[1]] = imgThresh
    
    #remap original image
    matOrig = ((matOrig-np.amin(matOrig)) / float(np.amax(matOrig)-np.amin(matOrig))) * 255
    matOrig[imgMask==0] = 0
    matOrig = ((matOrig-np.amin(matOrig)) / float(np.amax(matOrig)-np.amin(matOrig))) * 255
    
    
    gradX = ndimage.filters.sobel(matOrig, axis=1)
    gradY = ndimage.filters.sobel(matOrig, axis=0)
    gradMag = np.power((np.power(gradX,2) + np.power(gradY,2)),.5)
    
    gradXTemp = gradX.ravel()
    gradYTemp = gradY.ravel()
    gradTemp = np.transpose(np.vstack((gradXTemp, gradYTemp)))
    
    iMat = np.tile(np.array(range(imgWidth)),[imgHeight,1])
    jMat = np.transpose(np.tile(np.array(range(imgHeight)),[imgWidth,1]))
    
    iMat = iMat.ravel()
    jMat = jMat.ravel()
    
    iMat = iMat - center[0]
    jMat = jMat - center[1]
    distMat = np.transpose(np.vstack((iMat, jMat)))
    
    norm = np.power( np.sum( np.power(distMat,2), axis=1 ), 0.5)
    norm = np.transpose(np.tile(norm,[2,1]))
    distMat = distMat / (norm + .00001)
    
    gradC = np.sum((gradTemp * distMat), axis = 1).reshape(imgHeight, imgWidth)
    
    distMat = np.transpose(np.vstack((distMat[:,1]*-1, distMat[:,0])))
    
    gradR = np.sum((gradTemp * distMat), axis = 1).reshape(imgHeight, imgWidth)
    
    gradC = np.absolute(gradC)
    gradR = np.absolute(gradR)
    
    gradBoth = gradC + gradR
    
    ratioC = np.sum(gradC)/np.sum(gradBoth)
    ratioR = np.sum(gradR)/np.sum(gradBoth)
    gradAve = np.sum(gradBoth)/np.sum(imgMask)
    
    blueMask = np.zeros((dim[0], dim[1]))
    blueMask[center[0]-5:center[0]+5,center[1]-5:center[1]+5] = 255
    
    print gradC.shape
    print gradR.shape
    print blueMask.shape
    
    gradCombined =np.dstack((gradC, gradR, blueMask))
    print gradCombined.shape
    
    gradMag[center[0]-5:center[0]+5,center[1]-5:center[1]+5] = 255
    gradC[center[0]-5:center[0]+5,center[1]-5:center[1]+5] = 255
    gradR[center[0]-5:center[0]+5,center[1]-5:center[1]+5] = 255
    
    save_bw( gradC, dirSaveImg + "gradC_" + filesImg[i] )
    save_bw( gradR, dirSaveImg + "gradR_" + filesImg[i] )
    save_bw( gradBoth, dirSaveImg + "gradBoth_" + filesImg[i] )
    save_rgb( gradCombined, dirSaveImg + "gradCombined_" + filesImg[i] )
    
    save_bw( matOrig, dirSaveImg + filesImg[i] )