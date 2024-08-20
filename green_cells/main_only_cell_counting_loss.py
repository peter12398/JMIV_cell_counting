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
print(tf.__version__)
print('It should be >= 2.0.0.')


PATCH_SIZE = 1024
REDUCE_RATIO = 4
RESOLUTION = int(PATCH_SIZE/REDUCE_RATIO)
input_shape = [RESOLUTION,RESOLUTION,1] 


@tf.function
def condition_equal(last,new,image):
    return tf.math.logical_not(tf.reduce_all(tf.math.equal(last, new)))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="test", type=str, help="train or test")
parser.add_argument("--ROOT_PATH", default="/cluster/CMM/home/xfliu/workspace/JMIV_counting_cells", type=str, help="path to the root dir")
parser.add_argument("--DATA_DIR", default="/cluster/CMM/home/xfliu/workspace/JMIV_counting_cells/data/database_melanocytes_trp1/".format(PATCH_SIZE), type=str, help="path to TRP1 dataset")
parser.add_argument("--exp_name", default="newCCCLayer_RMSProp_dgmm24config_new_def_grad_minus_upstream_interactive_N_2_cache_Noclip_reduceLrPlateau".format(PATCH_SIZE), type=str, help="experiment name")
parser.add_argument("--model_weight_path", default="/cluster/CMM/home/xfliu/workspace/JMIV_counting_cells/best_model_newCCCLayer_RMSProp_dgmm24config_new_def_grad_minus_upstream_interactive_N_2_cache_Noclip_reduceLrPlateau.h5".format(PATCH_SIZE), type=str, help="experiment name")

args = parser.parse_args()

NORMALISE01 = False
#if not NORMALISE01:
dir_name = "best_h_dataset255"
#dir_name = "best_h_dataset255_new"
#else:
#    dir_name = "best_h_dataset01"
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

def update_dilation(last,new,mask):
     return [new, geodesic_dilation_step([new, mask]), mask]

@tf.function
def geodesic_dilation_step(X):
    """
    1 step of reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation_step, name="reconstruction")([Mask,Image])
    """
    # perform a geodesic dilation with X[0] as marker, and X[1] as mask
    return tf.keras.layers.Minimum()([tf.keras.layers.MaxPooling2D(pool_size=(3, 3),strides=(1,1),padding='same')(X[0]),X[1]])

@tf.function
def geodesic_dilation(X,steps=None):
    """
    Full reconstruction by dilation if steps=None, else
    K steps reconstruction by dilation
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(geodesic_dilation, name="reconstruction")([Mask,Image])
    """
    rec = X[0]
    #Full reconstruction is steps==None by dilation, else: partial reconstruction
    rec = geodesic_dilation_step([rec, X[1]])
    _, rec,_=tf.while_loop(condition_equal,
                            update_dilation,
                            [X[0], rec, X[1]],
                            maximum_iterations=steps)
    return rec

def reconstruction_dilation(X):
    """
    Full geodesic reconstruction by dilation, reaching idempotence
    :X tensor: X[0] is the Mask and X[1] is the Image
    :param steps: number of steps (by default NUM_ITER_REC)
    :Example:
    >>>Lambda(reconstruction_dilation, name="reconstruction")([Mask,Image])
    """
    return geodesic_dilation(X, steps=None)


@tf.custom_gradient
def custom_h_rec_and_exact_cc(xinput):
  x=xinput[0]
  #print('x.shape',x.shape)
  h=xinput[1]
  h=tf.expand_dims(tf.expand_dims(h,axis=-1),axis=-1)
  #print('h.shape',h.shape)
  x=tf.stop_gradient(x)

  xh=reconstruction_dilation([x-h,x])
  #X_xh = x-xh
  epsilon = 1e-5
  Rmax=(xh-reconstruction_dilation([xh-epsilon,xh]))>0
  dxh_dh = tf.cast(Rmax, tf.dtypes.float32)  #tape.jacobian(xh, h)

  N = 2
  print("N={} used inside.".format(N))
  def grad(upstream):
    return_grad = -upstream #upstream*tf.reduce_sum(-N*((x-xh)**(N-1)/(h**N))*((x-xh)/(h)+dxh_dh)) #tf.reduce_sum(N*h*((x - xh)/h)**N*(-dxh_dh/h - (x - xh)/h**2)/(x - xh),axis=[1,2])
    print('return_grad:',return_grad)
    return upstream, return_grad  #upstream *N*h*((x - xh)/h)**N*(-Derivative(xh(h), h)/h - (x - xh)/h**2)/(x - xh)
  #Connect Components Counting
  
  xminuxxh = dxh_dh #x-xh
  U=Sampling()(xminuxxh)
  M=tf.keras.layers.Minimum()([U,xminuxxh])
  R=tf.keras.layers.Lambda(geodesic_dilation)([M,xminuxxh])
  Detection=tf.cast(U==R,tf.float32)
  CC=tf.math.reduce_sum(Detection,axis=[1,2])
  print('CC',CC)
  return CC, grad
  """
  CC = tf.math.reduce_sum(((x-xh)/h)**10000,axis=[1,2])
  return CC, grad
  """


class Sampling(tf.keras.layers.Layer):
    """Sampling Random Uniform."""

    def call(self, inputs):
        dim = tf.shape(inputs)
        epsilon = tf.keras.backend.random_uniform(shape=(dim))/100
        return epsilon
  

# classes for data loading and preprocessing
class Dataset:
    """
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            ids = None,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
        
    ):  
        if ids:
            self.ids = ids
        else:
            self.ids = os.listdir(images_dir)
        print("len self.ids={}".format(len(self.ids)))
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.h_th_dict = {}
        self.SET = images_dir.split("/")[-2]

        with open(output_h_file_save_path + '/best_h_opening_closing_{}.json'.format(self.SET), 'r') as f:
            self.h_th_dict = json.load(f)
        
        self.ntruth_cache_dict = {}
        self.imlabels_cache_dict = {}
        self.imCol_cache_dict = {}
        self.rec_cache_dict = {}
        self.imMax_cache_dict = {}
        self.iminput_cache_dict = {}

    
    def __getitem__(self, i):
        
        # read data
        #print(self.images_fps[i])
        image_name = self.images_fps[i].split("/")[-1].split(".")[0]
        set_name = self.images_fps[i].split("/")[-3]
        #image = cv2.imread(self.images_fps[i])[:,:,1:2]
        
        if not image_name in self.imCol_cache_dict.keys():
            print("loading from disk to get imCol.. for {}".format(image_name))
            imColLarge = io.imread(self.images_fps[i])
            h,w = imColLarge.shape[0], imColLarge.shape[1]
            imCol = resize(imColLarge,(h//4, w//4))
            if not NORMALISE01:
                imCol = (255*imCol).astype('uint8')
            else:
                imCol = (imCol[:,:,1])
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
            mask = block_reduce(mask, block_size=(4,4), func=np.max)
            self.imlabels_cache_dict[image_name] = mask
        else:
            mask = self.imlabels_cache_dict[image_name]
            #print("loading imLabels for {} from self.imlabels_cache_dict".format(image_name))

        if not image_name in self.ntruth_cache_dict.keys():
            print("calculating ntruth.. for {}".format(image_name))
            ntruth = int(mask.sum()/255)
            self.ntruth_cache_dict[image_name] = ntruth
        else:
            ntruth = self.ntruth_cache_dict[image_name]
            #print("loading ntruth = {} for {} from self.ntruth_cache_dict".format(ntruth, image_name))

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
        
        #image += np.random.randn(256,256)*0.5

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
            data.append(list((self.dataset[j][0],self.dataset[j][2], self.dataset[j][4], self.dataset[j][5], self.dataset[j][6])))
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
                batch[-1] = np.flip(batch[-1], 2)
                
                
            if int(np.random.randint(2, size=1) % 2):
                #print("image and rec flip_left_right used")
                #batch[0] = tf.image.flip_left_right(np.expand_dims(batch[0],axis=-1))[:,:,:,0].numpy()
                #batch[1] = tf.image.flip_left_right(np.expand_dims(batch[1],axis=-1))[:,:,:,0].numpy()
                
                batch[0] = np.flip(batch[0], 1)
                batch[1] = np.flip(batch[1], 1)
                batch[-1] = np.flip(batch[-1], 1)
            
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
                batch[-1] = np.rot90(batch[-1], rotate_num, (1,2))
        else:
            pass


        #batch = [batch[0], batch[1]]
        batch[1] = np.expand_dims(batch[1],axis=-1)
        batch[-1] = np.expand_dims(batch[-1],axis=-1)
        #batch[2] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[2],axis=-1),axis=-1),axis=-1)
        #batch[3] =  np.expand_dims(np.expand_dims(np.expand_dims(batch[3],axis=-1),axis=-1),axis=-1)
        #batch = [batch[0], list((batch[1],batch[2],batch[3],batch[-1]))]
        batch = batch[0], batch[3]
        
        
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
        #x = tf.nn.relu(x)
        #x = tf.nn.sigmoid(x)
        #h = x #tf.expand_dims(tf.expand_dims(x,axis=-1),axis=-1)
        #x = tf.nn.relu(x)
        #x_max = tf.reduce_max(input_tensor)
        #h = K.clip(x, 0., x_max)
        h = x
        #h = tf.nn.sigmoid(x)*200. # h range 0-200
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
        #self.Hextrema = Hextrema()

        #self.opening_2D_layer = ml.Opening2D(num_filters = 1, kernel_size = (3,3))
        #self.closing_2D_layer = ml.Closing2D(num_filters = 1, kernel_size = (3,3))
        
        self.nn, self.nn_h = self.get_simple_model(self.input_shape)
        self.nn.summary()
        self.nn_epochs = EPOCHS
        self.nn_batch_size = EPOCHS
        self.loss = loss
        self.loss_weights = loss_weights
        self.lr = lr
        self.metrics = metrics
        #self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics = self.metrics)
        self.nn.compile(loss=self.loss,loss_weights = self.loss_weights, optimizer=tf.keras.optimizers.RMSprop(lr=self.lr, rho=0.9))
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
        
        xinput=xin
        xh=self.h_extrema_denoising_block_(xinput)
        NCC=tf.keras.layers.Lambda(custom_h_rec_and_exact_cc)([xinput,xh])
        detected_diff = NCC #modeltoCount(np.expand_dims(np.expand_dims(imMax,axis=0),axis=-1)).numpy()[0]

        return tf.keras.Model(xin,detected_diff), tf.keras.Model(xin,xh)
        #return tf.keras.Model(xin,final), tf.keras.Model(xin,xout_tmp), tf.keras.Model(xin,h)
    
    def predict_output(self, X):
        return self.nn.predict(X)
    
    def predict_h(self, X):
        return self.nn_h.predict(X)

    def train(self, verbose=1):
        if verbose == 1:
            self.nn.summary()

        #Callback definition
        CBs = [
            tf.keras.callbacks.ModelCheckpoint(args.ROOT_PATH + '/best_model_{}.h5'.format(exp_name), monitor='val_loss', verbose=1 ,save_weights_only=True, save_best_only=True, mode='min', period=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, min_lr=0.0001),
            tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(args.exp_name), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch'),
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
        
        plt.figure()
        plt.plot(self.history.history['loss'],label='loss')
        plt.plot(self.history.history['val_loss'],label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.IMAGE_SAVE_PATH+"/loss.jpg")
        plt.close()
        
        
    def test(self,):      

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
        for i in ids:
            #if i!= 18:
                
            #image, gt_mask, rec_, image_name ,gt_h , gt_threshold = self.test_dataset[i]
            image, mask, rec_, image_name ,gt_h, ntruth, imMax_best, imCol  = self.test_dataset[i]
            imname = self.test_dataset.images_fps[i].split("/")[-1].split(".")[0]
            image_expanded = np.expand_dims(image, axis=0)
                    
            try:
                pr_h =  self.predict_h(image_expanded).item() #self.predict_h(image_expanded) #self.predict_output(image_expanded)[1].item() #self.predict_h(image)
                pr_h_list.append(pr_h)
                gt_h_list.append(gt_h)

            except:
                pr_h =  self.predict_h(image_expanded)
                pr_h_list.append(count(pr_h)/(1024//PATCH_SIZE))
                gt_h_list.append(count(gt_h)/(1024//PATCH_SIZE))
            
            ndetec =  self.predict_output(image_expanded).item()
                        
            print("For h = "+ str(pr_h) + ", truth: " + str(ntruth) + " ; Detected: " + str(ndetec))
            
            n_detec_list.append(ndetec)
            n_gt_list.append(ntruth)
            
            count_err_diff_list.append(np.abs(ntruth - ndetec))
            count_err_nomerator_list.append(np.abs(ntruth))
            if np.abs(ntruth)!= 0:
                average_relative_error_list.append(np.abs(ntruth - ndetec)/np.abs(ntruth))
                average_relative_error_list_MPE.append((ntruth - ndetec)/np.abs(ntruth))

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

        save_fig_path = args.ROOT_PATH+"/visualize_test_{}".format(exp_name)
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
            print("{} made.".format(save_fig_path))
        
        for i in range(len(n_gt_array)):                           
            ax.annotate('%s' % str(i), xy = [n_gt_array[i], n_detec_array[i]] , textcoords='data')
        plt.title('Predicted vs true cell number, val set')
        plt.savefig(args.ROOT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,'n_detect_and_n_gt'))
        plt.show()


        pr_h_array = np.array(pr_h_list)
        gt_h_array = np.array(gt_h_list)

        fig = plt.figure()
        
        min_h = np.min([np.min(gt_h_array),np.min(pr_h_array)])
        max_h = np.max([np.max(gt_h_array),np.max(pr_h_array)])
        
        ax = fig.add_subplot(111)
        plt.plot(gt_h_array, pr_h_array, 'o')
        plt.plot(np.linspace(min_h,max_h,100), np.linspace(min_h,max_h,100))
        plt.xlabel('True h')
        plt.ylabel('Estimated h')
        
        for i in range(len(gt_h_array)):                           
            ax.annotate('%s' % str(i), xy = [gt_h_array[i], pr_h_array[i]] , textcoords='data')
        plt.title('Predicted vs true h, val set')
        plt.savefig(args.ROOT_PATH+"/visualize_test_{}/{}.jpg".format(exp_name,'pr_h_and_gt_h'))
        plt.show()

                
        #print("rat_err_avg =:", np.sum(rat_err_diff_list)/ np.sum(rat_err_nomerator_list))
        print("rat_err_avg =:", np.average(rat_err_list))
        print("average relative error =:", np.average(average_relative_error_list))
        print("total relative error =:", np.sum(count_err_diff_list)/ np.sum(count_err_nomerator_list))
        print("MAE =:", np.mean(count_err_diff_list))
        print("MPE =:", np.mean(average_relative_error_list_MPE))





if __name__ == "__main__":
    
    IMAGE_SAVE_PATH = ROOT_PATH + "/visualize_main"
    DATA_DIR = args.DATA_DIR
    #DATA_DIR = '/Users/santiago1/Downloads/database_melanocytes_trp1/'
    # load repo with data if it is not exists

    random_seed = 4
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    x_train_dir = os.path.join(DATA_DIR, 'set1', 'images')
    y_train_dir = os.path.join(DATA_DIR, 'set1', 'labels')

    x_test_dir = os.path.join(DATA_DIR, 'set2', 'images')
    y_test_dir = os.path.join(DATA_DIR, 'set2', 'labels')

    x_valid_dir = x_train_dir #os.path.join(DATA_DIR, 'val')
    y_valid_dir = y_train_dir #os.path.join(DATA_DIR, 'valannot')

    exp_name = args.exp_name #"1hmax_layer_out_conv_sigmoid_jaccardloss" #"1hmax_layer_out_conv_sigmpoid_l1loss_100gaussian"
                    
    # Lets look at data we have
    dataset = Dataset(x_train_dir, y_train_dir)


    BATCH_SIZE=16
    
    ids_images_train = os.listdir(x_train_dir)
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
        x_train_dir, 
        y_train_dir, 
        ids = ids_images_train,
    )
    # Dataset for validation images
    
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        ids = ids_images_val,
    )

    train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True, train = True)
    valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False, train = False)

    
    test_dataset = Dataset(
        x_test_dir, 
        y_test_dir, 
    )
    
    test_dataloader = Dataloder(test_dataset, batch_size=1, shuffle=False, train = False)

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
                                    test_dataset = test_dataset, 
                                    EPOCHS=EPOCHS, 
                                    BATCH_SIZE = BATCH_SIZE, 
                                    loss = "mean_absolute_error", 
                                    lr = LR,
                                    IMAGE_SAVE_PATH = IMAGE_SAVE_PATH,
                                    MODE = "only_hmaximalayer",
                                    resume = RESUME)
    if args.mode == "train":
        h_maxima_model.train()
    elif args.mode == "test":
        h_maxima_model.test()