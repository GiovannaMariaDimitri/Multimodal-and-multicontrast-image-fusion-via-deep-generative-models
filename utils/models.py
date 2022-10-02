

#Based on https://github.com/alexhallam/TensorFlow-Survival-Analysis/blob/master/deepsurv_tf.py

from keras.layers import Input, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Reshape, Dense, ELU, add, MaxPooling2D, Lambda, concatenate, UpSampling3D
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import numpy as np
import math
from keras import regularizers
from utils.sepconv3D import SeparableConv3D
from utils.augmentation import CustomIterator
from keras.models import load_model


class Parameters():
    def __init__ (self, param_dict):
        self.w_regularizer = param_dict['w_regularizer']
        self.batch_size = param_dict['batch_size']
        self.drop_rate = param_dict['drop_rate']
        self.epochs = param_dict['epochs']
        self.gpu = param_dict['gpu']
        self.model_filepath = param_dict['model_filepath'] + '/net.h5'
        self.image_shape = param_dict['image_shape']

class Net ():
    def __init__ (self, params):
        self.params = params
        self.modality1 = Input (shape = (self.params.image_shape))
        self.modality2 = Input (shape = (self.params.image_shape))

        xalex3D = XAlex3D(w_regularizer = self.params.w_regularizer, drop_rate = self.params.drop_rate)
        with tf.device(self.params.gpu):
            reconstructions , self.embedding = xalex3D ([self.modality1, self.modality2])
            self.reconstruction1 = reconstructions[0]
            self.reconstruction2 = reconstructions[1]


    def train (self,modalities):
        modality1, modality2 = modalities
        modality1, modality2 = self.normalize_data ([modality1, modality2])
        train_samples = modality1[0].shape[0]
        data_flow_train = CustomIterator ([modality1, modality2], batch_size = self.params.batch_size,
                                          shuffle = True)
        self.model = Model(inputs = [self.modality1, self.modality2], outputs = [self.reconstruction1, self.reconstruction2])
        #lrate = LearningRateScheduler(step_decay)
        #callback = [lrate]
        self.optimizer = Adam(lr=1e-3)
        self.model.compile(optimizer = self.optimizer, loss = ['binary_crossentropy', 'binary_crossentropy'], metrics = ['accuracy'])
        history = self.model.fit_generator (data_flow_train,
                   steps_per_epoch = train_samples/self.params.batch_size,
                   epochs = self.params.epochs,
                   #callbacks = callback,
                   shuffle = True)
                   #validation_data = data_flow_test,
                   #validation_steps =  test_samples/self.params.batch_size)
        return history.history

    def predict (self, modalities):
        #get model with two outputs and get embeddings
        #test_modalities = []
        modality1, modality2 = modalities
        modality1, modality2 = self.normalize_data ([modality1, modality2])
        test_modalities=[modality1,modality2]
        model = Model(inputs = [self.modality1, self.modality2], outputs = [self.reconstruction1, self.reconstruction2, self.embedding])
        reconstr1, reconstr2, embedding = model.predict (test_modalities, batch_size = self.params.batch_size)
        return reconstr1, reconstr2, embedding

    def normalize_data (self, modalities):
        self.modality_mins = []
        self.modality_maxs = []
        normalized_modalities = []
        for modality in modalities:
            min_im = np.min(modality, axis = 0)
            self.modality_mins.append(min_im)
            max_im = np.max(modality, axis = 0)
            self.modality_maxs.append(max_im)
            modality = (modality - min_im) /(max_im  - min_im + 1e-20)
            normalized_modalities.append(modality)
        return normalized_modalities

    def save_model(self, path):
        self.model.save(path)

    def load_model (self, path):
        self.model= load_model('model.h5', custom_objects={'SeparableConv3D': SeparableConv3D})

    def summary(self):
        return self.model.summary()


def XAlex3D(w_regularizer = None, drop_rate = 0.):
    def f(modalities):
        modality1, modality2 = modalities
        def downstream(mri):
            activations = []
            input = mri
            for i in range(5):
                activation = _conv_bn_relu_pool_drop(2**(i+1), 5, 6, 5,  w_regularizer = w_regularizer,drop_rate = drop_rate, pool=False)(input)
                activation = mid_flow_conv(activation, drop_rate, w_regularizer, filters = 2**(i+1))
                if i==0:
                    activations.append(activation)
                activation = _conv_bn_relu_pool_drop(2**(i+1), 5, 6, 5, w_regularizer = w_regularizer,  drop_rate = drop_rate, pool=True) (activation)
                activations.append(activation)
                input = activation
            return activations
        def upstream(activations):
            input=activations[-1]
            new_activations=activations[:-1][::-1]
            for activation in new_activations:
                up=_upconv_bn_relu_drop(activation.shape[-1], 5, 6, 5, w_regularizer = w_regularizer,drop_rate = drop_rate) (input)
                merge= concatenate([activation,up])
                merge=mid_flow_conv(merge, drop_rate, w_regularizer, 2*activation.shape[-1])
                conv=_conv_bn_relu_pool_drop(2*activation.shape[-1], 5, 6, 5, w_regularizer = w_regularizer,drop_rate = drop_rate, pool=False)(merge)
                input=conv
            reconstruction = Conv3D(1, 1, activation = 'sigmoid')(conv)
            return reconstruction
        activations1 = downstream(modality1)
        activations2 = downstream(modality2)
        merged_embeddings = concatenate([ activations1[-1],  activations2[-1]], axis = -1)
        embedding_shape = activations1[-1].get_shape().as_list()
        embeddings = _conv_bn_relu_pool_drop(embedding_shape[-1], embedding_shape[1], embedding_shape[2], embedding_shape[3],w_regularizer = w_regularizer,drop_rate = drop_rate, pool=False)(merged_embeddings)
        activations1[-1] = embeddings
        activations2[-1] = embeddings
        reconstruction1 = upstream(activations1)
        reconstruction2 = upstream(activations2)
        return [reconstruction1, reconstruction2], embeddings
    return f



def _fc_bn_relu_drop (units, w_regularizer = None, drop_rate = 0., name = None):
    #Defines Fully connected block (see fig. 3 in paper)
    def f(input):
        fc = Dense(units = units, activation = 'linear', kernel_regularizer=w_regularizer, name = name) (input) #was 2048 initially
        fc = BatchNormalization()(fc)
        fc = ELU()(fc)
        fc = Dropout (drop_rate) (fc)
        return fc
    return f

def _conv_bn_relu_pool_drop(filters, height, width, depth, strides=(1, 1, 1), padding = 'same', w_regularizer = None,
                            drop_rate = None, name = None, pool = False):
   #Defines convolutional block (see fig. 3 in paper)
   def f(input):
       conv = Conv3D(filters, (height, width, depth),
                             strides = strides, kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
       norm = BatchNormalization()(conv)
       elu = ELU()(norm)
       if pool == True:
           elu = MaxPooling3D(pool_size=(2, 2, 2)) (elu)
       return Dropout(drop_rate) (elu)
   return f


def _upconv_bn_relu_drop (filters, height, width, depth, strides = (1,1,1), padding = 'same', w_regularizer = None,
                          upsampling = (2,2,2), drop_rate = None, name = None):
    def f(input):
        upsampled = UpSampling3D (size=upsampling)(input)
        conv = Conv3D(filters, (height, width, depth),
                             strides = strides, kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(upsampled)
        norm = BatchNormalization()(conv)
        elu = ELU()(norm)
        return Dropout(drop_rate) (elu)
    return f



def _sepconv_bn_relu_pool_drop (filters, height, width, depth, strides = (1, 1, 1), padding = 'same', depth_multiplier = 1, w_regularizer = None,
                            drop_rate = None, name = None, pool = False):
    #Defines separable convolutional block (see fig. 3 in paper)
    def f (input):
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                             strides = strides, depth_multiplier = depth_multiplier,kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(input)
        sep_conv = BatchNormalization()(sep_conv)
        elu = ELU()(sep_conv)
        if pool == True:
           elu = MaxPooling3D(pool_size=3, strides=2, padding = 'same') (elu)
        return Dropout(drop_rate) (elu)
    return f

def _upsepconv_bn_relu_drop (filters, height, width, depth, strides = (1,1,1), padding = 'same', depth_multiplier = 1, w_regularizer = None,
                          upsampling = (2,2,2), drop_rate = None, name = None):
    def f(input):
        upsampled = UpSampling3D (size=upsampling)(input)
        sep_conv = SeparableConv3D(filters, (height, width, depth),
                             strides = strides, depth_multiplier = depth_multiplier,kernel_initializer="he_normal",
                             padding=padding, kernel_regularizer = w_regularizer, name = name)(upsampled)
        norm = BatchNormalization()(sep_conv)
        elu = ELU()(norm)
        return Dropout(drop_rate) (elu)
    return f



def mid_flow (x, drop_rate, w_regularizer, filters = 48):
    #3 consecutive separable blocks with a residual connection (refer to fig. 4)
    residual = x
    x = _sepconv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _sepconv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _sepconv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', depth_multiplier = 1, drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
    x = add([x, residual])
    return x


def mid_flow_conv (x, drop_rate, w_regularizer, filters = 48):
    #3 consecutive separable blocks with a residual connection (refer to fig. 4)
    residual = x
    x = _conv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _conv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', drop_rate=drop_rate, w_regularizer = w_regularizer )(x)
    x = _conv_bn_relu_pool_drop (filters, 5, 6, 5, padding='same', drop_rate=drop_rate, w_regularizer = w_regularizer)(x)
    x = add([x, residual])
    return x

def step_decay (epoch):
    #Decaying learning rate function
    initial_lrate = 1e-3
    drop = 0.3
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,((1+epoch)/epochs_drop))
    lrate = initial_lrate
    return lrate



def combine_tuples (control, pd):

    images = np.concatenate((control[0], pd[0]), axis = 0)
    meth = np.concatenate((control[1], pd[1]), axis = 0)
    ages = np.concatenate((control[2], pd[2]), axis = 0)

    return images, meth, ages



'''
    def evaluate (self, mris):
        test_mris = (mris - self.min_im) /(self.max_im  - self.min_im + 1e-20)
        self.model.compile(optimizer = self.optimizer, loss = ['binary_crossentropy'], metrics = ['accuracy'])
        metrics = self.model.evaluate (x = test_mris, y = test_mris, batch_size = self.params.batch_size)
        #will return mse error?

        return metrics
'''
