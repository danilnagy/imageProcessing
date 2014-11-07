from PIL import Image
import numpy as np
import math
import os
import joblib
import timeit
import LarsTools.py

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


X = ""
Y = ""

wSize = 10

imgExtList = [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".JPG",".JPEG",".PNG",".BMP",".TIF",".TIFF"]

dirWorking = "C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/"

dirNameOrgImg = 'C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/original_image/'
filesOrgImg = listDirectory(dirNameOrgImg, imgExtList)
dirNameMskImg = 'C:/Users/Danil/Documents/ADVP/dietrich/pythonProcessFinal/mask_image/'
filesMskImg = listDirectory(dirNameMskImg, imgExtList)

startTime00 = timeit.default_timer()

#print filesOrgImg

# train images tiles database matrix
count = 0
for i in range(len(filesOrgImg)):
    
    imgOrig = Image.open(dirNameOrgImg + filesOrgImg[i])
    
    try:
        imgMask = Image.open(dirNameMskImg + filesOrgImg[i])
    except Exception as e:
        print 'no mask found for image: ' + filesOrgImg[i]
        continue #go on to next part of loop
    
    imgWidth = imgOrig.size[0]
    imgHeight = imgOrig.size[1]
    
    newSize = (int(math.floor(imgWidth/4)), int(math.floor(imgHeight/4)))
    
    resizedOrig = imgOrig.resize(newSize, Image.ANTIALIAS)
    resizedMask = imgMask.resize(newSize, Image.ANTIALIAS)
    
    matrixOrig = np.asarray( resizedOrig, dtype='int32' )
    matrixMask = np.asarray( resizedMask, dtype='int32' )
    
    bwOrig = matrixOrig[:,:,1]
    bwMask = matrixMask[:,:,1]
    dim = bwOrig.shape
    
    xSize = int(math.floor(dim[0]/wSize))
    ySize = int(math.floor(dim[1]/wSize))
    
    print 'xSize = '+str(xSize)
    print 'ySize = '+str(ySize)
    
    #sys.exit()
    
    print 'extracting data from training image: '+str(i)
    for y in range(ySize):
        for x in range(xSize):
            #print 'x = '+str(x)+'; y = '+str(y)+'; count = '+str(count)
            
            imgNewOrig = bwOrig[x*wSize:(x+1)*wSize, y*wSize:(y+1)*wSize]
            
            imgNewMask = bwMask[x*wSize:(x+1)*wSize, y*wSize:(y+1)*wSize]
            
            # convert imgNew to single line
            newLineOrig = imgNewOrig.flatten()
            
            flatImgTile = imgNewMask.flatten()
            average = int(float(sum(flatImgTile))/float(len(flatImgTile)))
            if average == 0:
                newLineMask = 0
            elif average == 255:
                newLineMask = 2
            else:
                newLineMask = 1
                    
            # add a line to dataset (features)
            if count == 0:
                X = newLineOrig
                Y = newLineMask
            else:
                X = np.vstack((X,newLineOrig))
                Y = np.vstack((Y,newLineMask))
            
            count += 1
  		
    print 'count = '+str(count)


# numpy save csv
#np.savetxt('python/trainingDataImageMatrix.csv', X, fmt='%10.0f', delimiter=',')

# pickle
#output = open('python/trainingDataImageMatrix.pkl','wb')
#pickle.dump(X, output)

# joblib
joblib.dump(X, dirWorking + 'X.pkl')
joblib.dump(Y, dirWorking + 'y.pkl')
#XX = joblib.load('python/trainingDataImageMatrix_2img.pkl')
#print XX
#print XX.shape

# PyTable HDF5 file
#h5file = tables.openFile('python/trainingDataImageMatrix.h5', mode='w', title='Tain Data')
#root = h5file.root
#h5file.createArray(root, 'data', X)
#h5file.close()


print 'DONE' + ' (' + str((timeit.default_timer() - startTime00)) + ' seconds)'