
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import os, sys
import cv2
import os
import math
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import sklearn.metrics as sklm
import keras
import json
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from scipy.ndimage.filters import gaussian_filter as gf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GRU, Lambda
from keras.layers import Input, Dense, Activation, BatchNormalization, add
from keras.layers import Reshape, Lambda
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.layers import Reshape, Lambda, LeakyReLU
from keras.models import Model
import itertools
import json
# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib inline')


# In[4]:


sess = tf.Session()
K.set_session(sess)


# In[5]:


json_data = json.load(open("COCO_Text.json"))


# In[6]:


tf.__version__


# ## Split train and validation set

# In[7]:


from collections import Counter

def get_letters():
    letters = ''
    
    for imgId in json_data['imgs']:
        anns = json_data['imgToAnns'][imgId]
        if(len(anns) != 0):
            for annId in anns:
                if(json_data['anns'][str(annId)]['legibility'] == 'legible' and json_data['anns'][str(annId)]['language'] == 'english'):          
                    description = json_data['anns'][str(annId)]['utf8_string']
                    aux = ""
                    for letter in description:
                        if(('0' <= letter and letter <= '9') or                             ('a' <= letter and letter <= 'z') or                             ('A' <= letter and letter <= 'Z')):
                            letters += letter
                            aux += letter      
                    
    return Counter(letters)


# In[8]:


letters = get_letters()


# In[9]:


set_letters = set(letters)
letters = sorted(list(set_letters))


# In[13]:


def get_data(path, load_train = True):
 
    arr_samples = []
    
    for ix, imgId in enumerate(json_data['imgs']):
        if(json_data['imgs'][imgId]['set'] == 'train' and load_train):
            anns = json_data['imgToAnns'][imgId]
            if(len(anns) != 0):
                for annId in anns:
                    if(json_data['anns'][str(annId)]['legibility'] == 'legible' and json_data['anns'][str(annId)]['language'] == 'english'):          
                        bbox = json_data['anns'][str(annId)]['bbox']  
                        bbox = list(map(int, bbox))
                        name_img = json_data['imgs'][str(imgId)]['file_name']
                        image_temp = cv2.imread(path + name_img)
                        image_temp = image_temp[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                        description = json_data['anns'][str(annId)]['utf8_string']
                        
                        letters = ""
                        for letter in description:
                            if(('0' <= letter and letter <= '9') or                                 ('a' <= letter and letter <= 'z') or                                 ('A' <= letter and letter <= 'Z')):
                                letters += letter
                        
                        if(len(letters) > 0 and not image_temp is None):
                            arr_samples.append([image_temp, letters])
        else:
            anns = json_data['imgToAnns'][imgId]
            if(len(anns) != 0):
                for annId in anns:
                    if(json_data['anns'][str(annId)]['legibility'] == 'legible' and json_data['anns'][str(annId)]['language'] == 'english'):
                        bbox = json_data['anns'][str(annId)]['bbox']  
                        bbox = list(map(int, bbox))
                        name_img = json_data['imgs'][str(imgId)]['file_name']
                        image_temp = cv2.imread(path + name_img)
                        image_temp = image_temp[int(bbox[1]):int(bbox[1] + bbox[3]), int(bbox[0]):int(bbox[0] + bbox[2])]
                        description = json_data['anns'][str(annId)]['utf8_string']
                        
                        letters = ""
                        for letter in description:
                            if(('0' <= letter and letter <= '9') or                                 ('a' <= letter and letter <= 'z') or                                 ('A' <= letter and letter <= 'Z')):
                                letters += letter
                        
                        if(len(letters) > 0 and not image_temp is None):
                            arr_samples.append([image_temp, letters])
                    
    return arr_samples


# In[14]:


train_data = get_data('train2014/')


# In[55]:


ix_random = np.random.choice(range(len(train_data)))
print(ix_random)
print(train_data[ix_random][1])
plt.imshow(train_data[ix_random][0])


# In[21]:


test_data = get_data('train2014/', False)


# In[22]:


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def text_to_labels(text, max_len):
    labels =  list(map(lambda x: letters.index(x), text))
    for i in range(max_len - len(labels)):
        labels.append(1)
    return labels

def is_valid_str(s):
    for ch in s:
        if not ch in letters:
            return False
    return True

class TextImageGenerator:
    
    def __init__(self,                   
                 img_w, img_h, 
                 batch_size, 
                 downsample_factor,
                 data=None,
                 load_train=True,
                 max_text_len=50):
        
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        
        if(data):
            self.samples = data
        else:
            self.samples = get_data('train2014/', load_train)
        
        self.n = len(self.samples)
        self.indexes = list(range(self.n))
        self.cur_index = 0
        
    def build_data(self):
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []
        cont_none = 0
        for i, (img, text) in enumerate(self.samples):
            if(img is None):
                cont_none += 1
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (self.img_w, self.img_h))
            img = img.astype(np.float32)
            img /= 255
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            self.imgs[i, :, :] = img
            self.texts.append(text)
        print("Number of Nones " + str(cont_none))
        
    def get_output_size(self):
        return len(letters) + 1
    
    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            np.random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]
    
    def next_batch(self):
        while True:
            # width and height are backwards from typical Keras convention
            # because width is the time dimension when it gets fed into the RNN
            if K.image_data_format() == 'channels_first':
                X_data = np.ones([self.batch_size, 1, self.img_w, self.img_h])
            else:
                X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])
            Y_data = np.ones([self.batch_size, self.max_text_len])
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)
            label_length = np.zeros((self.batch_size, 1))
            source_str = []
                                   
            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                if K.image_data_format() == 'channels_first':
                    img = np.expand_dims(img, 0)
                else:
                    img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i] = text_to_labels(text, self.max_text_len)
                source_str.append(text)
                label_length[i] = len(text)
                
            inputs = {
                'the_input': X_data,
                'the_labels': Y_data,
                'input_length': input_length,
                'label_length': label_length,
                #'source_str': source_str
            }

            outputs = {'ctc': np.zeros([self.batch_size])}
            yield (inputs, outputs)


# ## Construct detector network

# In[23]:


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    print(y_pred.shape)
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# In[24]:


conv_filters = 16
kernel_size = (3, 3)
pool_size = 2
time_dense_size = 32
rnn_size = 512
img_w = 128
img_h = 64


# In[62]:


if K.image_data_format() == 'channels_first':
    input_shape = (1, img_w, img_h)
else:
    input_shape = (img_w, img_h, 1)

print("input shape " + str(input_shape))
batch_size = 32
downsample_factor = pool_size ** 2
tiger_train = TextImageGenerator(img_w, img_h, batch_size, downsample_factor, data=train_data)
tiger_train.build_data()
print('Finish constructing tiger train')
    
tiger_val = TextImageGenerator(img_w, img_h, batch_size, downsample_factor, data=test_data)
tiger_val.build_data()
print('Finish constructing tiger test')


# In[26]:


act = 'relu'
input_data = Input(name='the_input', shape=input_shape, dtype='float32')


x = Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_1')(input_data)
x = LeakyReLU(alpha=0.1)(x)

# Layer 2
x = Conv2D(32, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_2')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_3')(x)
x = LeakyReLU(alpha=0.1)(x)

# Layer 5
x = Conv2D(64, (3,3), strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_4')(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * 64)
inner = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x)

# cuts down input size going into RNN:
inner = Dense(time_dense_size, activation=act, name='dense1')(inner)

# Two layers of bidirecitonal GRUs
# GRU seems to work as well, if not better than LSTM:
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(gru1_merged)

# transforms RNN output to character activations:
inner = Dense(tiger_train.get_output_size(), kernel_initializer='he_normal',               name='dense2')(concatenate([gru_2, gru_2b]))
y_pred = Activation('softmax', name='softmax')(inner) 

Model(inputs=input_data, outputs=y_pred).summary()

labels = Input(name='the_labels', shape=[tiger_train.max_text_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
# Keras doesn't currently support loss funcs with extra parameters
# so CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

# clipnorm seems to speeds up convergence
#sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)


# In[27]:


# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
print('Finish compiling model')
    

test_func = K.function([input_data], [y_pred])

model.fit_generator(generator=tiger_train.next_batch(), 
                            steps_per_epoch=tiger_train.n,
                            epochs=1, 
                            validation_data=tiger_val.next_batch(), 
                            validation_steps=tiger_val.n)


# In[28]:


def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret


# In[72]:


from pylab import rcParams
def evaluate_randomly(bs=1):

    net_inp = model.get_layer(name='the_input').input
    net_out = model.get_layer(name='softmax').output
    
    X_data = []
    for step in range(bs):
        ix_random = np.random.choice(range(len(test_data)))
        temp_image = test_data[ix_random][0]
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        temp_image = cv2.resize(temp_image, (img_w, img_h))
        temp_image = temp_image.astype(np.float32)
        temp_image /= 255
        X_data.append(temp_image.T)
    
    X_data = np.array(X_data)
    X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
    texts = []
    texts.append(test_data[ix_random][1])
        
    net_out_value = sess.run(net_out, feed_dict={net_inp:X_data})
    pred_texts = decode_batch(net_out_value)
    for i in range(bs):
        fig = plt.figure(figsize=(10, 10))
        outer = gridspec.GridSpec(2, 1, wspace=10, hspace=0.1)
        ax1 = plt.Subplot(fig, outer[0])
        fig.add_subplot(ax1)
        ax2 = plt.Subplot(fig, outer[1])
        fig.add_subplot(ax2)
        print('Predicted: %s\nTrue: %s' % (pred_texts[i], texts[i]))
        img = X_data[i][:, :, 0].T
        ax1.set_title('Input img')
        ax1.imshow(img, cmap='gray')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_title('Activations')
        ax2.imshow(net_out_value[i].T, cmap='binary', interpolation='nearest')
        ax2.set_yticks(list(range(len(letters) + 1)))
        ax2.set_yticklabels(letters + ['blank'])
        ax2.grid(False)
        for h in np.arange(-0.5, len(letters) + 1 + 0.5, 1):
            ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)
        
        #ax.axvline(x, linestyle='--', color='k')
        plt.show()


# In[134]:


evaluate_randomly()

