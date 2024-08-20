from skimage.morphology import dilation, disk, square, opening, closing, label
from skimage.measure import block_reduce, regionprops
from skimage import io
from skimage.transform import resize
import numpy as np
import glob, os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import morpholayers.layers as ml
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
import time
import cv2

PATCH_SIZE = 1024
REDUCE_RATIO = 4
RESOLUTION = int(PATCH_SIZE/REDUCE_RATIO)
input_shape = [RESOLUTION,RESOLUTION,1] 
import argparse
# Functions and algorithm for the detection of cells as h-maxima
# Assuming 8 bits grayscale images


parser = argparse.ArgumentParser()
parser.add_argument("--ROOT_PATH", default="/cluster/CMM/home/xfliu/workspace/JMIV_counting_cells", type=str, help="path to the root dir")
parser.add_argument("--DATA_DIR", default="/cluster/CMM/home/xfliu/workspace/JMIV_counting_cells/data/fluocells_organised_with_zeros_official_testsplit/", type=str, help="path to TRP1 dataset")
args = parser.parse_args()

# Geodesic dilation
def geoDil(imMark, imMask, se):
    imDil = dilation(imMark, se)
    imRes = np.minimum(imDil, imMask)
    return imRes

# Geodesic reconstruction (iterated geodesic dilations until idempotence)
def geoRec(imMark, imMask, se):
    conv = ((imMask-imMark).max()==0)
    imMk = imMark
    while not conv:
        imAux = geoDil(imMk, imMask, se)
        conv = ((imAux-imMk).max()==0)
        imMk = imAux
    return imMk


# h-reconstruction : geoRec(im-h, im)
def hRec(im, h, se):
    im_h = np.maximum(im.astype('int') - h, 0)
    imRec = geoRec(im_h, im, se)
    return imRec

# The h-maxima are the regional maxima of the h-reconstruction
def hMax(im, h, se):
    imRec = hRec(im, h, se)
    imRec2 = hRec(imRec, 1, se)
    imMax = np.zeros(imRec2.shape)
    idxs = np.where(imRec2 < imRec)
    imMax[idxs] = 255
    return imMax, imRec

# Morphological algorithm with parameter h:
# opening, then closing, then h-maxima.
def detectCells(im, h, b, MAX_0XH = True, NORMALISE01 = True):
    #imOpen = opening(im, b)
    #imAF = closing(imOpen, b)
    #imMax, imRec  = hMax(imAF, h, disk(1))
    #se_size = 3
    #opening_2D_layer = ml.Opening2D(num_filters = 1, kernel_size = (se_size,se_size))
    #closing_2D_layer = ml.Closing2D(num_filters = 1, kernel_size = (se_size,se_size))
    
    xin=kl.Input(shape=input_shape)
    #x = opening_2D_layer(xin)
    #x = closing_2D_layer(x)  #ml.closing2d(x, b, strides = (1,1), padding = "same")
    x = xin
    x_after_openin_closing = xin
    
    if MAX_0XH:
        ret = tf.nn.relu(x-h)
        """
        zero_ = tf.zeros_like(x - h, dtype = tf.float32)
        print("maximum with 0 called")
        ret = tf.maximum(x - h,zero_)
        """
        xrec=kl.Lambda(ml.geodesic_dilation, name="reconstruction")([ret,x])
    else:
        xrec=kl.Lambda(ml.geodesic_dilation, name="reconstruction")([x-h,x])
        
    if NORMALISE01:
        delta = tf.constant(1/255., dtype=tf.float32)
    else:
        delta = tf.constant(1., dtype=tf.float32)
    xrec2=kl.Lambda(ml.geodesic_dilation, name="reconstruction2")([xrec-delta,xrec])
    

    #rmax_reg = kl.Lambda(ml.region_maxima_transform)(xrec)
    #xout = kl.Multiply(name="r_max_reg")([xrec, 255.*rmax_reg])
    
    #imMax = tf.zeros_like(xout, dtype = tf.float32)
    #idxs = tf.where(xout > imMax)
    #imMax[idxs] = 1.
    
    modelREC=tf.keras.Model(inputs=xin,outputs=[xrec,xrec2, x_after_openin_closing])
    modelREC.compile()
    
    xrec,xrec2, x_after_openin_closing = modelREC.predict(np.expand_dims(np.expand_dims(im,axis=0),axis=-1),verbose=0)
    xrec2 = np.squeeze(xrec2)
    xrec = np.squeeze(xrec)
    #imMax,imRec = modelREC.predict(np.expand_dims(np.expand_dims(im,axis=0),axis=-1),verbose=0)
    
    imMax = np.zeros(xrec2.shape)

    if xrec2.min() < xrec2.max():
        idxs = np.where(xrec2 < xrec)
        imMax[idxs] = 255
        
        cc, ncells = label(imMax, return_num = True, connectivity = 2)
        props = regionprops(cc)

        imDetec = np.zeros(imMax.shape)
        for k in range(len(props)):
            xc = int(props[k].centroid[0])
            yc = int(props[k].centroid[1])
            imDetec[xc, yc] = 255
        imDetec = dilation(imDetec, square(3))
    else:
        imDetec = np.zeros(imMax.shape)
        ncells = 0

    return ncells, imDetec, xrec, x_after_openin_closing, imMax


def generate_best_h_dataset(input_npy_save_path, output_image_save_path , output_h_file_save_path, dataset_dir, output_npy_save_path, set_name = 'set1', NORMALISE01=False, MAX_0XH = True):
    images_dir = dataset_dir + set_name + '/images/'
    labels_dir = dataset_dir + set_name + '/labels/'
    print("images_dir: {}".format(images_dir))
    print("labels_dir: {}".format(labels_dir))
    best_h_dict ={}
    image_dirs = os.listdir(images_dir) #glob.glob(images_dir + '*.png')
    print("num of datas: {}".format(len(image_dirs)))
    for imname in tqdm(image_dirs):
        #i = 0 #index of image
        #if i == 65:
        print("processing {}".format(imname))
        im_path = os.path.join(images_dir, imname)

        start = time.time()
        imColLarge = io.imread(im_path, as_gray=False)
        imColLarge_gray = io.imread(im_path, as_gray=True)
        h,w = imColLarge.shape[0], imColLarge.shape[1]
        imCol  = resize(imColLarge,(RESOLUTION, RESOLUTION))
        im  = resize(imColLarge_gray,(RESOLUTION, RESOLUTION))
        #im = (imCol[:,:,1])
        im = (255*im).astype('uint8')
        
        imLabelsLarge = io.imread(im_path.replace("images", "labels"))#io.imread(labels_dir+imname+'.png')
        imLabels = imLabelsLarge
        #imLabels = (255*resize(imLabelsLarge,(RESOLUTION, RESOLUTION))).astype('uint8')  #block_reduce(imLabelsLarge, block_size=(REDUCE_RATIO,REDUCE_RATIO), func=np.max)

        _, binary_image = cv2.threshold(imLabels, 128, 255, cv2.THRESH_BINARY)
        label_im = label(binary_image)
        props = regionprops(label_im)
        #imshow(label_im)
        num_connected_components = len(props)

        ntruth = num_connected_components #int(imLabels.sum()/255)
    
        imTruth = np.zeros(imCol.shape)
        
        ###
        imLabels_ = np.zeros(imLabels.shape)
        for k in range(len(props)):
            xc = int(props[k].centroid[0])
            yc = int(props[k].centroid[1])
            imLabels_[xc, yc] = 255

        imLabels_resized  = resize((imLabels_),(RESOLUTION, RESOLUTION))
        imLabels_resized = imLabels_resized/imLabels_resized.max()
        imLabels_resized = (255*imLabels_resized).astype('uint8')
        #imLabels_resized = dilation(imLabels_resized, square(3))

        imLabels = dilation(imLabels_resized, square(3))

        idxs = np.where(imLabels > 0)
        for k in range(3):

            imTruth[:,:,k] = im/255
            if k == 0:
                imTruth[idxs[0], idxs[1], k] = 1
            else :
                imTruth[idxs[0], idxs[1], k] = 0

        #if not ntruth:     
        sz = 3
        b = square(sz) # structuring element
        b = tf.expand_dims(tf.convert_to_tensor(b, dtype=tf.float32), axis=-1)
        
        tmp = np.expand_dims(np.array(im, dtype=np.float32),axis=(0,-1))
        x = ml.opening2d(tmp, b, strides = (1,1), padding = "same")
        im_after_opening_closing = ml.closing2d(x, b, strides = (1,1), padding = "same")[0,:,:,0]
        
        hstar = 0
        besterr = 100
        #h_range = range(0, 150, 1)
        h_range = range(1, 255, 1)
        dfh = np.zeros(len(h_range))
        j = 0
        for h in h_range:
            if NORMALISE01:
                h = h/255
            ndetec, imDetec, imRec, x_after_openin_closing, imMax = detectCells(im_after_opening_closing, h, b, MAX_0XH, NORMALISE01)
            dfh[j] = ndetec
            j = j+1
            if ntruth != 0:
                err = abs(ntruth-ndetec)/ntruth
            else: 
                err = abs(ntruth-ndetec)/100.
            if err < besterr:
                besterr = err
                hstar = h
            elif err > besterr: # early stopping 
                break

        ndetec, imDetec, imRec_best, x_after_openin_closing, imMax_best = detectCells(im_after_opening_closing, hstar, b, MAX_0XH, NORMALISE01)
        print("For h = "+ str(hstar) + ", truth: " + str(ntruth) + " ; Detected: " + str(ndetec))

        end = time.time()
        print("time spent {}".format(end-start))

        impatchname = imname 
        best_h_dict[impatchname] = str(hstar)
        
        imDetecCol = np.zeros(imCol.shape)
        idxs = np.where(imDetec > 0)
        for k in range(3):
            imDetecCol[:,:,k] = im/255
            if k == 0:
                imDetecCol[idxs[0], idxs[1], k] = 1
            else :
                imDetecCol[idxs[0], idxs[1], k] = 0

        plt.figure(figsize = (16, 8))
        plt.subplot(1,5,1)
        plt.plot(h_range, dfh)
        plt.vlines(hstar, ymin = np.min(dfh), ymax = np.max(dfh))
        plt.plot(h_range, ntruth*np.ones(len(h_range)), label = 'Expected num. of cells: '+str(ntruth)+'.')
        plt.xlabel('h')
        plt.ylabel('Detections')
        plt.title(impatchname + ': Num. of detections vs h')
        plt.legend()
        
        plt.subplot(1,5,2)
        plt.imshow(imCol)
        plt.title('Input image (resized)')
        plt.subplot(1,5,3)
        plt.title('Green channel and ground truth ('+str(ntruth)+' cells).')
        plt.imshow(imTruth)
        
        plt.subplot(1,5,4)
        plt.title('Detections for h = ' + str(hstar) + ': '+ str(ndetec)+' cells.')
        plt.imshow(imDetecCol)
        
        plt.subplot(1,5,5)
        plt.title('imRec_best for h = ' + str(hstar) )
        plt.imshow(imRec_best)
        
        ouput_path = output_image_save_path + "/" + set_name 
        if not os.path.exists(ouput_path):
            os.makedirs(ouput_path)
        plt.savefig(ouput_path + "/" + impatchname + "_rec.jpg")
        plt.close()
        #plt.show()set_name = 'set1'
                
        ouput_np_path = output_npy_save_path + "/" + set_name 
        if not os.path.exists(ouput_np_path):
            os.makedirs(ouput_np_path)
        np.save(ouput_np_path + "/" + impatchname + ".npy", imRec_best)
        
        ouput_np_path = output_npy_save_path + "/" + set_name 
        if not os.path.exists(ouput_np_path):
            os.makedirs(ouput_np_path)
        np.save(ouput_np_path + "/" + impatchname + "_imMax_best.npy", imMax_best)
        
        input_np_path = input_npy_save_path + "/" + set_name 
        if not os.path.exists(input_np_path):
            os.makedirs(input_np_path)
        np.save(input_np_path + "/" + impatchname + "_after_opening_closing.npy", x_after_openin_closing)
        

        input_np_path = input_npy_save_path + "/" + set_name 
        if not os.path.exists(input_np_path):
            os.makedirs(input_np_path)
        np.save(input_np_path + "/" + impatchname + ".npy", im)
        #else:
        #    continue
        
    return  best_h_dict


if __name__ == "__main__":
    NORMALISE01 = False
    MAX_0XH = True
    
    if not NORMALISE01:
        dir_name = "best_h_dataset255_yellow_cells_withZeros_forloopearlystop_official_testsplit_bugfixGtMask"
        #dir_name = "best_h_dataset255_se2"
        #dir_name = "best_h_dataset255_se4"
    else:
        dir_name = "best_h_dataset01"
        
    ROOT_PATH = args.ROOT_PATH #"."
    output_image_save_path = ROOT_PATH + "/{}/ouput_images".format(dir_name)
    output_npy_save_path = ROOT_PATH + "/{}/ouput_np".format(dir_name)
    output_h_file_save_path = ROOT_PATH + "/{}/best_h".format(dir_name)
    input_npy_save_path = ROOT_PATH + "/{}/input_np".format(dir_name)
    #images_dir = '../../database_melanocytes_trp1/set1/images/'
    #images_dir = '/home/xiaohu/PythonProject/MINES/segmentation_models_400epochs/examples/data_/database_melanocytes_trp1/set1/images/'
    #labels_dir = '../../database_melanocytes_trp1/set1/labels/'
    #labels_dir = '/home/xiaohu/PythonProject/MINES/segmentation_models_400epochs/examples/data_/database_melanocytes_trp1/set1/labels/'
    dataset_dir = args.DATA_DIR #'/home/xiaohu/PythonProject/MINES/segmentation_models_400epochs/examples/data_/database_melanocytes_trp1/'
    print("DATA_DIR used: {}".format(dataset_dir))
    
    for set_name in ['set1', 'set2']:
        print("processing {}".format(set_name))
        best_h_dict = generate_best_h_dataset(input_npy_save_path = input_npy_save_path, output_image_save_path = output_image_save_path, output_h_file_save_path= output_h_file_save_path, dataset_dir = dataset_dir, output_npy_save_path = output_npy_save_path , set_name = set_name, NORMALISE01 = NORMALISE01, MAX_0XH = MAX_0XH)
        
        print("best_h_dict:", best_h_dict)
        jsObj = json.dumps(best_h_dict)  
        if not os.path.exists(output_h_file_save_path):
            os.makedirs(output_h_file_save_path)
        fileObject = open(output_h_file_save_path + '/best_h_opening_closing_{}.json'.format(set_name)  , 'w')
        fileObject.write(jsObj)  
        fileObject.close()  
        print("best_h_opening_closing_{}.json wrote".format(set_name))