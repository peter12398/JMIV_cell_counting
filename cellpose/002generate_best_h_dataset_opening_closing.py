from skimage.morphology import dilation, disk, square, opening, closing, label
from skimage.measure import block_reduce, regionprops
from skimage import io
from skimage.transform import resize
import numpy as np
import glob, os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
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

import os
dirname = os.getcwd()
#dirname = os.path.dirname(__file__)
parent_dir_name = os.path.dirname(dirname)
print("parent_dir_name = {}.\n".format(parent_dir_name))

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", default = parent_dir_name + "/data/cellpose/CellPose_converted/", type=str, help="path to CellPose dataset")
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
    return ncells, imDetec, xrec, x_after_openin_closing, imMax


def generate_best_h_dataset(input_npy_save_path, output_image_save_path , output_h_file_save_path, dataset_dir, output_npy_save_path, set_name = 'set1', NORMALISE01=False, MAX_0XH = True):
    images_dir = dataset_dir + set_name
    print("images_dir: {}".format(images_dir))

    best_h_dict ={}
    image_dirs = os.listdir(images_dir) #glob.glob(images_dir + '*.png')
    print("num of datas: {}".format(len(image_dirs)//2))
    for imname in tqdm(image_dirs):
        if "img" in imname:
            impath = os.path.join(images_dir, imname)
            maskname = imname.replace("img","masks")
            maskpath = os.path.join(images_dir, maskname)

            print("processing {}".format(imname))
        
            start = time.time()
            imColLarge = io.imread(impath)
            h,w = imColLarge.shape[0], imColLarge.shape[1]
            imCol  = resize(imColLarge,(256, 256))

            im = (255*imCol).astype('uint8')
            
            imLabelsLarge = io.imread(maskpath)#io.imread(labels_dir+imname+'.png')
            #imLabels = block_reduce(imLabelsLarge, block_size=(256, 256), func=np.max)
            imLabels = cv2.resize(imLabelsLarge, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            #imLabels = imLabelsLarge

            ntruth = int(imLabelsLarge.sum()/255)
            imTruth = np.zeros((imCol.shape[0], imCol.shape[1], 3))
            imLabels = dilation(imLabels, square(3))
            idxs = np.where(imLabels > 0)
            for k in range(3):

                imTruth[:,:,k] = im/255
                if k == 0:
                    imTruth[idxs[0], idxs[1], k] = 1
                else :
                    imTruth[idxs[0], idxs[1], k] = 0
            """
            plt.figure(figsize = (16, 10))
            plt.subplot(1,2,1)
            plt.imshow(imCol)
            plt.title('Input image (resized)')
            plt.subplot(1,2,2)
            plt.title('Green channel and ground truth ('+str(ntruth)+' cells).')
            plt.imshow(imTruth)
            plt.show()
            """
            sz = 3
            b = square(sz) # structuring element
            b = tf.expand_dims(tf.convert_to_tensor(b, dtype=tf.float32), axis=-1)
            
            tmp = np.expand_dims(np.array(im, dtype=np.float32),axis=(0,-1))
            x = ml.opening2d(tmp, b, strides = (1,1), padding = "same")
            im_after_opening_closing = ml.closing2d(x, b, strides = (1,1), padding = "same")[0,:,:,0]

            """
            hstar = 0
            besterr = 1
            #h_range = range(0, 150, 1)
            #h_range = range(0, 255, 2)
            H_MIN = 0
            H_MAX = 255
            ERR_THRESHOLD = 0.0001
            MAX_ITER_STEP = 10 # 2**8 = 256
            #dfh = np.zeros(len(h_range))
            ndetec_list = []
            h_tested_list = []
            iter_step = 0
            err = 100
            #for h in h_range:
            h = int(H_MIN+H_MAX)//2 # init h
            while err > ERR_THRESHOLD and iter_step < MAX_ITER_STEP:

                ndetec, imDetec, imRec, x_after_openin_closing, imMax = detectCells(im_after_opening_closing, h, b, MAX_0XH, NORMALISE01)
                ndetec_list.append(ndetec)
                h_tested_list.append(h)

                iter_step += 1
                if ntruth != 0:
                    err = abs(ntruth-ndetec)/ntruth
                else: 
                    err = abs(ntruth-ndetec)/100.
                if err < besterr:
                    besterr = err
                    hstar = h

                if ndetec < ntruth: # h should go left
                    h = (h+H_MIN)//2
                    H_MAX = (H_MAX+H_MIN)//2
                elif ndetec > ntruth: # h should go right
                    h = (h+H_MAX)//2
                    H_MIN = (H_MAX+H_MIN)//2
                else:
                    print("find best h, break")
                    break
            """
            hstar = 0
            besterr = 100
            #h_range = range(0, 150, 1)
            h_range = range(0, 255, 1)
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


            best_h_dict[imname] = str(hstar)
            
            imDetecCol = np.zeros((imCol.shape[0], imCol.shape[1], 3))
            idxs = np.where(imDetec > 0)
            for k in range(3):
                imDetecCol[:,:,k] = im/255
                if k == 0:
                    imDetecCol[idxs[0], idxs[1], k] = 1
                else :
                    imDetecCol[idxs[0], idxs[1], k] = 0
            """
            plt.figure(figsize = (16, 8))
            plt.subplot(1,5,1)
            plt.scatter(h_tested_list, ndetec_list)
            plt.vlines(hstar, ymin = np.min(ndetec_list), ymax = np.max(ndetec_list), colors='r', linestyles='dotted')
            plt.plot(h_tested_list, ntruth*np.ones(len(h_tested_list)), label = 'Expected num. of cells: '+str(ntruth)+'.')
            plt.xlabel('h')
            plt.ylabel('Detections')
            plt.title(imname + ': Num. of detections vs h')
            plt.legend()
            """
            plt.figure(figsize = (16, 8))
            plt.subplot(1,5,1)
            plt.plot(h_range, dfh)
            plt.vlines(hstar, ymin = np.min(dfh), ymax = np.max(dfh), color = "r", linestyles= "dotted")
            plt.plot(h_range, ntruth*np.ones(len(h_range)), label = 'Expected num. of cells: '+str(ntruth)+'.')
            plt.xlabel('h')
            plt.ylabel('Detections')
            plt.title(imname + ': Num. of detections vs h')
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
            plt.savefig(ouput_path + "/" + imname + "_rec.jpg")
            plt.close()
            #plt.show()set_name = 'set1'
                    
            ouput_np_path = output_npy_save_path + "/" + set_name 
            if not os.path.exists(ouput_np_path):
                os.makedirs(ouput_np_path)
            np.save(ouput_np_path + "/" + imname + ".npy", imRec_best)
            
            ouput_np_path = output_npy_save_path + "/" + set_name 
            if not os.path.exists(ouput_np_path):
                os.makedirs(ouput_np_path)
            np.save(ouput_np_path + "/" + imname + "_imMax_best.npy", imMax_best)
            
            input_np_path = input_npy_save_path + "/" + set_name 
            if not os.path.exists(input_np_path):
                os.makedirs(input_np_path)
            np.save(input_np_path + "/" + imname + "_after_opening_closing.npy", x_after_openin_closing)
            

            input_np_path = input_npy_save_path + "/" + set_name 
            if not os.path.exists(input_np_path):
                os.makedirs(input_np_path)
            np.save(input_np_path + "/" + imname + ".npy", im)
            #else:
            #    continue
            plt.close("all")
    
    return  best_h_dict


if __name__ == "__main__":
    NORMALISE01 = False
    MAX_0XH = True
    
    if not NORMALISE01:
        dir_name = "best_h_dataset255_CellPose_converted"
        #dir_name = "best_h_dataset255_se2"
        #dir_name = "best_h_dataset255_se4"
    else:
        raise NotImplementedError
        
    ROOT_PATH = parent_dir_name + "/data" #args.ROOT_PATH #"."
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
    
    for set_name in ['train', 'test']:
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