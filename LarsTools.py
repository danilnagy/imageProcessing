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