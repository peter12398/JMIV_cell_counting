# %%
import os
import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt
import scipy
import keras.backend as K
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter
from tensorflow.keras import layers as kl
import tensorflow as tf
import tensorflow.keras.models as km
import tensorflow.keras.layers as kl
from tensorflow.keras.callbacks import *
from tensorflow.python.keras import regularizers
from skimage.morphology import dilation, disk, square, opening, closing, label
import glob
from skimage.measure import block_reduce, regionprops
#sys.path.append("/home/xiaohu/PythonProject/DIMA_comtage_cellule/examples/segmentation_models_/models")
import morpholayers.layers as ml
import tensorflow as tf
import json
from skimage import io
from skimage.transform import resize
import morpholayers.layers as ml
from tensorflow.keras import layers as kl
from skimage.morphology import area_opening
from skimage.morphology import label
import ipdb
import argparse
import random
print(tf.__version__)
print('It should be >= 2.0.0.')

import os
dirname = os.getcwd()
#dirname = os.path.dirname(__file__)
parent_dir_name = os.path.dirname(dirname)
print("parent_dir_name = {}.\n".format(parent_dir_name))

PATCH_SIZE = 1024
REDUCE_RATIO = 4
RESOLUTION = int(PATCH_SIZE/REDUCE_RATIO)
input_shape = [RESOLUTION,RESOLUTION,1] 


parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="test", type=str, help="train or test")
parser.add_argument("--DATA_DIR", default=parent_dir_name+"/data/cellpose/CellPose_converted", type=str, help="path to TRP1 dataset")
parser.add_argument("--exp_name", default="debug", type=str, help="experiment name")
parser.add_argument("--model_weight_path", default="pretrained/model/weight.h5", type=str, help="experiment name")
parser.add_argument("--best_h_dataset_name", default="best_h_dataset255_CellPose_converted", type=str, help="best_h_dataset_name name")
args = parser.parse_args()

NORMALISE01 = False
if not NORMALISE01:
    dir_name = args.best_h_dataset_name #"best_h_dataset255_CellPose_converted_05_06_2024_onlyRed_max3channels_forloopearlystop_bugfixUint8_less200"
    #dir_name = "best_h_dataset255_new"
else:
    dir_name = "best_h_dataset01"
print("dir:{} used".format(dir_name))    

ROOT_PATH = args.ROOT_PATH #"/home/xiaohu/workspace/MINES/DGMM2024_comptage_cellule"
output_npy_save_path = ROOT_PATH + "/{}/ouput_np".format(dir_name)
output_h_file_save_path = ROOT_PATH + "/{}/best_h".format(dir_name)
input_npy_save_path = ROOT_PATH + "/{}/input_np".format(dir_name)

def count(images):
    """PLot images in one row."""
    tmp = np.sum(images)
    #print("sum = :", tmp)
    return tmp
    

class Sampling(tf.keras.layers.Layer):
    """Sampling Random Uniform."""

    def call(self, inputs):
        dim = tf.shape(inputs)
        epsilon = tf.keras.backend.random_uniform(shape=(dim))
        return epsilon
  
# classes for data loading and preprocessing
class Dataset:
    """
    """
    def __init__(
            self, 
            images_dir,  
            ids = None,
    ):  
        if ids:
            self.ids = ids
        else:
            self.ids = os.listdir(images_dir)
        print("len self.ids={}".format(len(self.ids)))
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [i.replace("img","masks") for i in self.images_fps]
        self.h_th_dict = {}
        self.SET = images_dir.split("/")[-1]

        with open(output_h_file_save_path + '/best_h_opening_closing_{}.json'.format(self.SET), 'r') as f:
            self.h_th_dict = json.load(f)

        self.ntruth_cache_dict = {}
        self.imlabels_cache_dict = {}
        self.iminput_cache_dict = {}
        self.imCol_cache_dict = {}
        self.rec_cache_dict = {}
        self.imMax_cache_dict = {}
    
    def __getitem__(self, i):
        
        # read data
        #print(self.images_fps[i])
        image_name = self.images_fps[i].split("/")[-1]
        set_name = self.images_fps[i].split("/")[-2]
        #image = cv2.imread(self.images_fps[i])[:,:,1:2]

        if not image_name in self.imCol_cache_dict.keys():
            print("loading from disk to get imCol.. for {}".format(image_name))
            #ipdb.set_trace()
            converted_path = self.images_fps[i]
            orig_path = converted_path.replace("CellPose_converted_05_06_2024_onlyRed_max3channels_bugfixUint8_less200","CellPose").replace("img_converted","img")
            imColLarge = io.imread(orig_path)
            h,w = imColLarge.shape[0], imColLarge.shape[1]
            imCol = resize(imColLarge,(256, 256))
            if not NORMALISE01:
                imCol = (255*imCol).astype('uint8')
            else:
                raise NotImplementedError
                #imCol = (imCol[:,:,1])
            self.imCol_cache_dict[image_name] = imCol
        else:
            imCol = self.imCol_cache_dict[image_name]
            #print("loading imCol for {} from self.imCol_cache_dict".format(image_name))
        

        if not image_name in self.iminput_cache_dict.keys():
            print("loading from disk to get image.. for {}".format(image_name))
            input_np_path = input_npy_save_path + "/" + set_name 
            image = np.load(input_np_path + "/" + image_name + "_after_opening_closing.npy")[0,:,:,0]
            self.iminput_cache_dict[image_name] = image
        else:
            image = self.iminput_cache_dict[image_name]
        

        gt_h = np.array(float(self.h_th_dict[str(image_name)]))

        # cache
        if not image_name in self.imlabels_cache_dict.keys():
            print("loading from disk to get imLabels.. for {}".format(image_name))
            #mask = cv2.imread(self.masks_fps[i], 0)
            mask = io.imread(self.masks_fps[i]) 
            imLabels = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_AREA)
            self.imlabels_cache_dict[image_name] = imLabels
        else:
            imLabels = self.imlabels_cache_dict[image_name]
            #print("loading imLabels for {} from self.imlabels_cache_dict".format(image_name))
        
        if not image_name in self.ntruth_cache_dict.keys():
            print("using regionprops to get ntruth.. for {}".format(image_name))
            #mask = cv2.imread(self.masks_fps[i], 0)
            mask = io.imread(self.masks_fps[i]) 
            ntruth = int(mask.sum()/255)
            self.ntruth_cache_dict[image_name] = ntruth
        else:
            ntruth = self.ntruth_cache_dict[image_name]
            #print("loading ntruth = {} for {} from self.ntruth_cache_dict".format(ntruth, image_name))

        mask = imLabels
   
        if not image_name in self.rec_cache_dict.keys():
            print("loading rec_ from disk for {}".format(image_name))
            #REC=modelREC_.predict([np.expand_dims(np.expand_dims(image,axis=0),axis=-1),np.expand_dims(np.expand_dims(mask,axis=0),axis=-1)],verbose=0)
            npy_path_to_load = output_npy_save_path + "/" +set_name + "/" + image_name + ".npy"
            #print("npy loaded from:", npy_path_to_load)
            rec_ = np.load(npy_path_to_load)
            self.rec_cache_dict[image_name] = rec_
        else:
            rec_ = self.rec_cache_dict[image_name]
        

        if not image_name in self.imMax_cache_dict.keys():
            print("loading imMax_best from disk for {}".format(image_name))
            npy_path_to_load = output_npy_save_path + "/" +set_name + "/" + image_name + "_imMax_best.npy"
            #print("npy loaded from:", npy_path_to_load)
            imMax_best = np.load(npy_path_to_load)/255.
            self.imMax_cache_dict[image_name] = imMax_best
        else:
            imMax_best = self.imMax_cache_dict[image_name] 

        if mask.max()>0:
            mask = mask / mask.max()
        #mask = np.array(scipy.ndimage.morphology.binary_dilation(mask, iterations = 8),dtype=np.uint8)
        #ipdb.set_trace()

        return image, mask, rec_, image_name ,gt_h, ntruth, imMax_best, imCol #, gt_threshold

        
    def __len__(self):
        return len(self.ids)
  
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, train=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.train = train

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            #data.append(tuple((self.dataset[j][0],self.dataset[j][2])))
            data.append(list((self.dataset[j][0],self.dataset[j][2], self.dataset[j][4], self.dataset[j][5])))
            #data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        #batch = [batch[0], list((batch[1],batch[2]))]
        
        if self.train:
            
            if int(np.random.randint(2, size=1) % 2):
                #print("image and rec flip_up_down used")
                #batch[0] = tf.image.flip_up_down(np.expand_dims(batch[0],axis=-1))[:,:,:,0].numpy()
                #batch[1] = tf.image.flip_up_down(np.expand_dims(batch[1],axis=-1))[:,:,:,0].numpy()
                
                batch[0] = np.flip(batch[0], 2)
                batch[1] = np.flip(batch[1], 2)
                #batch[-1] = np.flip(batch[-1], 2)
                
                
            if int(np.random.randint(2, size=1) % 2):
                #print("image and rec flip_left_right used")
                #batch[0] = tf.image.flip_left_right(np.expand_dims(batch[0],axis=-1))[:,:,:,0].numpy()
                #batch[1] = tf.image.flip_left_right(np.expand_dims(batch[1],axis=-1))[:,:,:,0].numpy()
                
                batch[0] = np.flip(batch[0], 1)
                batch[1] = np.flip(batch[1], 1)
                #batch[-1] = np.flip(batch[-1], 1)
            
            """"
            if int(np.random.randint(2, size=1) % 2):
                x_roll_dis =  np.random.randint(1,batch[0].shape[1])
                batch[0] = np.roll(batch[0],x_roll_dis, axis=1)
                batch[1] = np.roll(batch[1],x_roll_dis, axis=1)
                
            if int(np.random.randint(2, size=1) % 2):
                y_roll_dis =  np.random.randint(1,batch[0].shape[2])
                batch[0] = np.roll(batch[0],y_roll_dis, axis=2)
                batch[1] = np.roll(batch[1],y_roll_dis, axis=2)
            """
                
            if int(np.random.randint(2, size=1) % 2):
                rotate_num =  np.random.randint(1,4)
                batch[0] = np.rot90(batch[0], rotate_num, (1,2))
                batch[1] = np.rot90(batch[1], rotate_num, (1,2))
                #batch[-1] = np.rot90(batch[-1], rotate_num, (1,2))
        else:
            pass


        #batch = [batch[0], batch[1]]
        batch[1] = np.expand_dims(batch[1],axis=-1)
        #batch[-1] = np.expand_dims(batch[-1],axis=-1)
        #batch[2] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[2],axis=-1),axis=-1),axis=-1)
        #batch[3] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[3],axis=-1),axis=-1),axis=-1)
        batch = batch[0], list((batch[1],batch[2],batch[3]))
        #batch = [batch[0], batch[3]]
        
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)   

class Hextrema(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Hextrema, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Hextrema, self).build(input_shape)

    def call(self, x):
        return ml.h_maxima_transform([x[0], x[1]])

class h_extrema_denoising_block2(tf.keras.Model):
    def __init__(self, dropout=0.2, name="h_extrema_denoising_block2"):       
        super(h_extrema_denoising_block2, self).__init__(name=name)
        self.conv1 = kl.Conv2D(filters = 8, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv2 = kl.Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv3 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.conv4 = kl.Conv2D(filters = 32, kernel_size = (3, 3), activation='relu',kernel_initializer='glorot_uniform')
        self.maxpooling = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.maxpooling2 = kl.MaxPool2D(pool_size=(2,2), padding="valid")
        self.dropout = kl.Dropout(dropout)
        self.globalavgpooling = kl.GlobalMaxPooling2D(data_format='channels_last')
        self.dense = kl.Dense(1,kernel_constraint=tf.keras.constraints.NonNeg(),name="h_denoising")
        self.batchnorm1 = kl.BatchNormalization()
        self.batchnorm2 = kl.BatchNormalization()
        self.batchnorm3 = kl.BatchNormalization()
        self.batchnorm4 = kl.BatchNormalization()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        #x = tf.nn.relu(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        #x = tf.nn.relu(x)
        x = self.maxpooling(x)
        
        
        x = self.batchnorm2(x)
        x = self.conv3(x)
        #x = tf.nn.relu(x)
        x = self.batchnorm3(x)
        x = self.conv4(x)
        #x = tf.nn.relu(x)
        x = self.maxpooling2(x)

        x = self.globalavgpooling(x)        
        x = self.dense(x)
        h = tf.expand_dims(tf.expand_dims(x,axis=-1),axis=-1)
        return h

def detectCells(im, NORMALISE01 = False):
    #imOpen = opening(im, b)
    #imAF = closing(imOpen, b)
    #imMax, imRec  = hMax(imAF, h, disk(1))

    xrec=kl.Input(shape=input_shape)
  
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
    
    modelREC=tf.keras.Model(inputs=xrec,outputs=[xrec,xrec2])
    modelREC.compile()
    
    xrec,xrec2 = modelREC.predict(im,verbose=0)
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
    return ncells, imDetec, imMax

class H_maxima_model:
    def __init__(self, input_shape, train_dataloader, valid_dataloader, test_dataloader, test_dataset, EPOCHS=100, BATCH_SIZE = 16, loss = 'binary_crossentropy', loss_weights = None , lr = 0.01,  IMAGE_SAVE_PATH = "/home/xiaohu/PythonProject/DIMA_comtage_cellule/visualize_main", MODE = "post_processor", metrics = None, resume = False) -> None:
        self.input_shape = input_shape
        self.MODE = MODE
        print("MODE = {}".format(self.MODE))
        
        self.h_extrema_denoising_block_ = h_extrema_denoising_block2()
        self.Hextrema = Hextrema()

        #self.opening_2D_layer = ml.Opening2D(num_filters = 1, kernel_size = (3,3))
        #self.closing_2D_layer = ml.Closing2D(num_filters = 1, kernel_size = (3,3))
        
        self.nn, self.nn_h, self.nn_n, self.nn_binaraliszd = self.get_simple_model(self.input_shape)
        self.nn_epochs = EPOCHS
        self.nn_batch_size = EPOCHS
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.metrics = metrics
        #self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics = self.metrics)
        #self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.RMSprop(lr=self.lr, rho=0.9), metrics = self.metrics)
        self.nn.compile(loss=self.loss, loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.RMSprop(lr=self.lr, rho=0.9))
        self.resume = resume
        
        if self.resume:
            best_weight_load_path = args.ROOT_PATH + '/best_model_{}.h5'.format(exp_name)
            self.nn.load_weights(best_weight_load_path)
            print("load weight from :{}".format(best_weight_load_path))
        
        #self.nn.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate = self.lr), metrics = self.metrics)
        print("lr =",self.lr)
        print("metrics =",self.metrics)
        print("self.nn.compile done\n")
        
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.test_dataset = test_dataset
        self.IMAGE_SAVE_PATH = IMAGE_SAVE_PATH
        


        
    def get_simple_model(self, input_shape):
        xin=kl.Input(shape=input_shape)
        
        x = xin   

        h = self.h_extrema_denoising_block_(x)
        h_ret = tf.math.reduce_sum(h,axis=[1,2,3])
        
        rec = self.Hextrema([h, x])
        
        delta = tf.constant(1., dtype=tf.float32)
        rec2 = kl.Lambda(ml.geodesic_dilation, name="reconstruction2")([rec-delta,rec])
        
        size_im = self.input_shape #(256,256,1)
        c=10*size_im[0]*size_im[1]*size_im[2] #Different should be greater that one to be detected in the approximation of =
        InLayer = rec-rec2
        
        
        U=tf.ones_like(InLayer)
        M=tf.keras.layers.Minimum()([U,InLayer*c])
        imMax=tf.nn.relu((1-tf.math.abs((U-M))*c)) #Approximation of =
        
        InLayer = imMax
        U=Sampling()(InLayer)
        U=U*c
        M=tf.keras.layers.Minimum()([U,InLayer*c])
        R= tf.keras.layers.Lambda(ml.geodesic_dilation,name="rec")([M,InLayer*c])
        #NCC=tf.math.equal(U,R) #This is not differentiable.
        Detection=tf.nn.relu(1-tf.math.abs(U-R)) #Approximation of =
        NCC=tf.math.reduce_sum(Detection,axis=[1,2,3])

        detected_diff = NCC #modeltoCount(np.expand_dims(np.expand_dims(imMax,axis=0),axis=-1)).numpy()[0]

        return tf.keras.Model(inputs=xin, outputs=[rec,h_ret,detected_diff]), tf.keras.Model(xin,h), tf.keras.Model(xin,detected_diff), tf.keras.Model(xin,imMax)
        #return tf.keras.Model(xin,final), tf.keras.Model(xin,xout_tmp), tf.keras.Model(xin,h)
    
    def predict_output(self, X):
        return self.nn.predict(X)
    
    def predict_h(self, X):
        return self.nn_h.predict(X)
    
    def predict_n(self, X):
        return self.nn_n.predict(X)
    
    def predict_binaraliszd(self, X):
        return self.nn_binaraliszd(X)

    def train(self, verbose=1):
        if verbose == 1:
            self.nn.summary()

        #Callback definition
        CBs = [
            tf.keras.callbacks.ModelCheckpoint(args.ROOT_PATH + '/best_model_{}.h5'.format(exp_name), monitor='val_loss', verbose=1 ,save_weights_only=True, save_best_only=True, mode='min', period=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=0.0001),
            tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(exp_name), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                              patience=300, verbose=1, mode='auto',
                              baseline=None, restore_best_weights=False)
            #ModelCheckpoint('best_model.h5', monitor='loss',verbose=1, save_best_only=True)
        ]
        #Training the model
        #ipdb.set_trace()
        self.history = self.nn.fit(
                self.train_dataloader, 
                steps_per_epoch=len(self.train_dataloader), 
                epochs=self.nn_epochs, 
                validation_data=self.valid_dataloader, 
                validation_steps=len(self.valid_dataloader),
                callbacks=CBs,
            )
        
        """
        plt.figure()
        plt.plot(self.history.history['loss'],label='loss')
        plt.plot(self.history.history['val_loss'],label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.IMAGE_SAVE_PATH+"/loss.jpg")
        plt.close()
        """
        
    def test(self, test_set):      

        best_weight_load_path = args.model_weight_path #args.ROOT_PATH + '/pretrained_model_weight/best_model_{}.h5'.format(exp_name)
        self.nn.load_weights(best_weight_load_path)
        print("load weight from :{}".format(best_weight_load_path))
        
        ids = np.arange(len(self.test_dataset)) #np.random.choice(np.arange(len(test_dataset)), size=n)

        rat_err_list = []
        rat_err_diff_list = []
        rat_err_nomerator_list = []
        
        count_err_nomerator_list = []
        count_err_diff_list = []
        average_relative_error_list = []
        average_relative_error_list_MPE = []
        
        pr_h_list = []
        gt_h_list = []
        
        n_detec_list = []
        n_gt_list = []
        imname_list = []
        for i in ids:
            
            #image, gt_mask, rec_, image_name ,gt_h , gt_threshold = self.test_dataset[i]
            image, gt_mask, rec_, image_name ,gt_h, ntruth, imMax_best, imCol  = self.test_dataset[i]
            imname = self.test_dataset.images_fps[i].split("/")[-1].split(".")[0]
            image_expanded = np.expand_dims(image, axis=0)
            pr_mask = self.predict_output(image_expanded)[0] #self.predict_output(image_expanded) #self.predict_output(image_expanded)[0]
            
            try:
                pr_h =  self.predict_output(image_expanded)[1].item() #self.predict_h(image_expanded) #self.predict_output(image_expanded)[1].item() #self.predict_h(image)
                pr_h_list.append(pr_h)
                gt_h_list.append(gt_h)

            except:
                pr_h =  self.predict_output(image_expanded)[1][0]
                pr_h_list.append(count(pr_h)/(1024//PATCH_SIZE))
                gt_h_list.append(count(gt_h)/(1024//PATCH_SIZE))
            
            pr_n =  self.predict_n(image_expanded).item()
            pr_binarized_map =  self.predict_binaraliszd(image_expanded).numpy().squeeze()
            
            imTruth = np.zeros((RESOLUTION,RESOLUTION,3),dtype=np.uint8)
            image = imCol
            #imRGB = np.zeros((256,256,3))
            #imLabels = dilation(gt_mask, square(3))
            imLabels = gt_mask
            imLabels = dilation(imLabels, square(3))
            idxs = np.where(imLabels > 0)
            for k in range(3):
                #if not NORMALISE01:
                #    imTruth[:,:,k] = image/255 #im #im/255
                #else:
                imTruth[:,:,k] = I_converted = np.max(image, axis = -1) #image[:,:,k] #im/255
                if k == 0:
                    imTruth[idxs[0], idxs[1], k] = 255
                else :
                    imTruth[idxs[0], idxs[1], k] = 0
            
            ndetec, imDetec, imMax_pred = detectCells(pr_mask, NORMALISE01)
            print("For h = "+ str(pr_h) + ", truth: " + str(ntruth) + " ; Detected: " + str(ndetec))
            
            n_detec_list.append(ndetec)
            n_gt_list.append(ntruth)
            
            count_err_diff_list.append(np.abs(ntruth - ndetec))
            count_err_nomerator_list.append(np.abs(ntruth))
            if np.abs(ntruth)!= 0:
                average_relative_error_list.append(np.abs(ntruth - ndetec)/np.abs(ntruth))
                average_relative_error_list_MPE.append((ntruth - ndetec)/np.abs(ntruth))
            
            imDetecCol = np.zeros((RESOLUTION,RESOLUTION,3),dtype=np.uint8)
            idxs = np.where(imDetec > 0)
            for k in range(3):
                #if not NORMALISE01:
                #    imDetecCol[:,:,k] = image/255
                #else:
                imDetecCol[:,:,k] = I_converted = np.max(image, axis = -1) #image[:,:,k] #im/255
                if k == 0:
                    imDetecCol[idxs[0], idxs[1], k] = 255
                else :
                    imDetecCol[idxs[0], idxs[1], k] = 0

            save_path = args.ROOT_PATH+"/visualize_{}_{}".format(test_set ,exp_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.imsave(save_path+"/{}_Input.png".format(imname), imCol)
            plt.imsave(save_path+"/{}_imTruth_{}.png".format(imname,ntruth), imTruth)
            plt.imsave(save_path+"/{}_imDetecCol_{}.png".format(imname,ndetec), imDetecCol)
            plt.imsave(save_path+"/{}_rec_pred.png".format(imname), pr_mask[0,:,:,0])
            plt.imsave(save_path+"/{}_rec_gt.png".format(imname), rec_)
            plt.imsave(save_path+"/{}_RMAX_pred.png".format(imname), imMax_pred)
            plt.imsave(save_path+"/{}_RMAX_gt.png".format(imname), imMax_best)
            
            
            if count(gt_mask) < 0.001:
                continue
            else:
                print("count(pr_h):",count(pr_h))
                print("count(gt_h):",count(gt_h))
                rat_err_diff_list.append(np.abs(count(pr_h)- count(gt_h)))
                rat_err_nomerator_list.append( (count(gt_h)))
                if (count(gt_h)) > 1e-3:
                    rat_err = np.abs((count(pr_h)- count(gt_h)) ) / (count(gt_h))
                    rat_err_list.append(rat_err)

            imname_list.append(imname.split("_img")[0])



        data = [{"n_detec_list":n_detec_list}, {"n_gt_list":n_gt_list}]
        with open(args.ROOT_PATH+"/visualize_{}_{}/test_result_list.json".format(test_set, exp_name), 'w') as f:
            json.dump(data, f, indent=4)
            
        print("len(pr_h_list):", len(pr_h_list))
        #ipdb.set_trace()
        plt.close()
        plt.figure()
        pr_h_array = np.array(pr_h_list)
        gt_h_array = np.array(gt_h_list)
        plt.plot(pr_h_array, label='pr_h_array')
        plt.plot(gt_h_array, label='gt_h_array')
        plt.legend()
        plt.title("test set predited h")
        plt.savefig(args.ROOT_PATH+"/visualize_{}_{}/{}.jpg".format(test_set, exp_name,'pr_h_array'))
        

        n_detec_array = np.array(n_detec_list)
        n_gt_array = np.array(n_gt_list)
        
        fig = plt.figure()
        
        min_n = np.min([np.min(n_gt_array),np.min(n_detec_array)])
        max_n = np.max([np.max(n_gt_array),np.max(n_detec_array)])
        
        ax = fig.add_subplot(111)
        plt.plot(n_gt_array, n_detec_array, 'o')
        plt.plot(np.linspace(min_n,max_n,100), np.linspace(min_n,max_n,100))
        plt.xlabel('True cell number')
        plt.ylabel('Estimated cell number')
        
        for i in range(len(n_gt_array)):                           
            ax.annotate('%s' % str(imname_list[i]), xy = [n_gt_array[i], n_detec_array[i]] , textcoords='data')
        plt.title('Predicted vs true cell number, {} set'.format(test_set))
        plt.savefig(args.ROOT_PATH+"/visualize_{}_{}/{}.jpg".format(test_set, exp_name,'n_detect_and_n_gt'))
        plt.show()

        
        fig = plt.figure()
        
        min_h = np.min([np.min(gt_h_array),np.min(pr_h_array)])
        max_h = np.max([np.max(gt_h_array),np.max(pr_h_array)])
        
        ax = fig.add_subplot(111)
        plt.plot(gt_h_array, pr_h_array, 'o')
        plt.plot(np.linspace(min_h,max_h,100), np.linspace(min_h,max_h,100))
        plt.xlabel('True h')
        plt.ylabel('Estimated h')
        
        for i in range(len(gt_h_array)):                           
            ax.annotate('%s' % str(imname_list[i]), xy = [gt_h_array[i], pr_h_array[i]] , textcoords='data')
        plt.title('Predicted vs true h, {} set'.format(test_set))
        plt.savefig(args.ROOT_PATH+"/visualize_{}_{}/{}.jpg".format(test_set, exp_name,'pr_h_and_gt_h'))
        plt.show()
                
        #print("rat_err_avg =:", np.sum(rat_err_diff_list)/ np.sum(rat_err_nomerator_list))
        print("rat_err_avg =:", np.average(rat_err_list))
        print("average relative error =:", np.average(average_relative_error_list))
        print("total relative error =:", np.sum(count_err_diff_list)/ np.sum(count_err_nomerator_list))
        print("MAE =:", np.average(count_err_diff_list))
        print("MPE =:", np.average(average_relative_error_list_MPE))

        plt.figure()
        plt.hist(gt_h_list, label="{} set".format(test_set))
        plt.title("histogarm best h CellPose on {} set".format(test_set))
        plt.ylabel("histogarm best h CellPose {} set".format(test_set))
        plt.xlabel("N")
        plt.legend()
        plt.savefig(args.ROOT_PATH+"/visualize_{}_{}/{}.jpg".format(test_set, exp_name,'hist_gt_h'))
        plt.close('all')


if __name__ == "__main__":
    
    IMAGE_SAVE_PATH = ROOT_PATH + "visualise_test_results/visualize_CellPose"
    DATA_DIR = args.DATA_DIR
    #DATA_DIR = '/Users/santiago1/Downloads/database_melanocytes_trp1/'
    # load repo with data if it is not exists

    random_seed = 2024
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    random.seed(random_seed)

    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    valid_dir = train_dir #os.path.join(DATA_DIR, 'val')
    exp_name = args.exp_name #"1hmax_layer_out_conv_sigmoid_jaccardloss" #"1hmax_layer_out_conv_sigmpoid_l1loss_100gaussian"
                    
    BATCH_SIZE=16
    
    ids_images_train = [i for i in os.listdir(train_dir) if "img" in i]
    ids_images_test = [i for i in os.listdir(test_dir) if "img" in i]
    TRAIN_VAL_SPLIT = 0.8
    len_train = len(ids_images_train)
    idx_val = int(TRAIN_VAL_SPLIT*len_train)
    
    np.random.shuffle(ids_images_train)
    # Reserve 10,000 samples for validation
    ids_images_val = ids_images_train[idx_val:]
    ids_images_train = ids_images_train[:idx_val]
    print("train len:{}; val len:{}".format(len(ids_images_train), len(ids_images_val)))
    
    # Dataset for train images
    train_dataset = Dataset(
        train_dir, 
        ids = ids_images_train,
    )
    # Dataset for validation images
    
    valid_dataset = Dataset(
        valid_dir, 
        ids = ids_images_val,
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, train = True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False, train = False)

    
    test_dataset = Dataset(
        test_dir, 
        ids = ids_images_test,
    )
    
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train = False)

    test_set = "test"
    test_set_dict = {"train": train_dataset, "val": valid_dataset, "test": test_dataset}
    test_dataset_to_use = test_set_dict[test_set]
    print("test_set = {}".format(test_set))
    # %%

    EPOCHS = 1600
    LR = .001
    BATCH_SIZE = 16
    NORMALISE01_ = False
    assert NORMALISE01_ == NORMALISE01
    RESUME = False
    
    lambda_h = 0.1 # hyperparameter to be adjusted
    
    h_maxima_model = H_maxima_model(input_shape = input_shape, 
                                    train_dataloader = train_dataloader, 
                                    valid_dataloader = valid_dataloader, 
                                    test_dataloader = test_dataloader, 
                                    test_dataset = test_dataset_to_use, 
                                    EPOCHS=EPOCHS, 
                                    BATCH_SIZE = BATCH_SIZE, 
                                    loss = ["mean_squared_error","mean_absolute_error","mean_absolute_error"], 
                                    loss_weights = [1., 0., 0.001],
                                    lr = LR,
                                    IMAGE_SAVE_PATH = IMAGE_SAVE_PATH,
                                    MODE = "only_hmaximalayer",
                                    resume = RESUME)
    if args.mode == "train":
        h_maxima_model.train()
    elif args.mode == "test":
        h_maxima_model.test(test_set)