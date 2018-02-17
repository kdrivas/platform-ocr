import keras
import cv2
import itertools
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GRU
from keras.layers import Input, Activation, BatchNormalization, add
from keras.layers import Reshape, Lambda, LeakyReLU, concatenate
from keras.optimizers import Adam
from keras.models import Model

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import variables as tf_variables

from Utils import LETTERS

def ctc_batch(y_true, y_pred, input_length, label_length):
    
    label_length = tf.to_int32(tf.squeeze(label_length))
    input_length = tf.to_int32(tf.squeeze(input_length))
    sparse_labels = tf.to_int32(K.ctc_label_dense_to_sparse(y_true, label_length))

    y_pred = tf.log(tf.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

    return tf.expand_dims(ctc.ctc_loss(inputs=y_pred,
                                       labels=sparse_labels,
                                       sequence_length=input_length, ignore_longer_outputs_than_inputs=True), 1)




def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return ctc_batch(labels, y_pred, input_length, label_length)


class Recognizer():
    
    def __init__(self, max_text_len, act='relu', kernel_size=(3, 3), pool_size=2, img_w=128, img_h=64, rnn_size=512, conv_filters=32, time_dense_size=32, load_model=None, tiger_train=None, tiger_val=None):        
        
        if(not load_model and not tiger_train and not tiger_val):
            print('error loading train data')
            return
        
        self.sess = tf.Session()
        K.set_session(self.sess)
        
        self.img_w = img_w
        self.img_h = img_h
        output_size = len(LETTERS) + 1

        if K.image_data_format() == 'channels_first':
            input_shape = (1, img_w, img_h)
        else:
            input_shape = (img_w, img_h, 1)
        
        input_data = Input(name='the_input', shape=input_shape, dtype='float32')
        
        x = Conv2D(conv_filters, kernel_size, strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_1')(input_data)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(conv_filters, kernel_size, strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

        x = Conv2D(conv_filters * 2, kernel_size, strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        x = Conv2D(conv_filters * 2, kernel_size, strides=(1,1), padding='same', kernel_initializer='he_normal', name='conv_4')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(pool_size, pool_size))(x)

        conv_to_rnn_dims = (img_w // (pool_size ** 2), (img_h // (pool_size ** 2)) * conv_filters * 2)
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
        
        inner = Dense(output_size, kernel_initializer='he_normal', \
                      name='dense2')(concatenate([gru_2, gru_2b]))
        y_pred = Activation('softmax', name='softmax')(inner) 

        labels = Input(name='the_labels', shape=[max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        # clipnorm seems to speeds up convergence
        #sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

        self.model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
        print('Finish creating model')
        
        if(load_model):
            self.model.load_weights(load_model)
            print('Finish loading model')
        else:
            self.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
            print('Finish compiling model')

            test_func = K.function([input_data], [y_pred])
            
            self.model.fit_generator(generator=tiger_train.next_batch(), 
                            steps_per_epoch=tiger_train.n,
                            epochs=1, 
                            validation_data=tiger_val.next_batch(), 
                            validation_steps=tiger_val.n)
    
    def decode_batch(self, out):
        ret = []
        for j in range(out.shape[0]):
            out_best = list(np.argmax(out[j, 2:], 1))
            out_best = [k for k, g in itertools.groupby(out_best)]
            outstr = ''
            for c in out_best:
                if c < len(LETTERS):
                    outstr += LETTERS[c]
            ret.append(outstr)
        return ret
    
    def evaluate_randomly(self, test_data, bs=1):

        net_inp = self.model.get_layer(name='the_input').input
        net_out = self.model.get_layer(name='softmax').output

        X_data = []
        for step in range(bs):
            ix_random = np.random.choice(range(len(test_data)))
            temp_image = test_data[ix_random][0]
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
            temp_image = cv2.resize(temp_image, (self.img_w, self.img_h))
            temp_image = temp_image.astype(np.float32)
            temp_image /= 255
            X_data.append(temp_image.T)

        X_data = np.array(X_data)
        X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
        texts = []
        texts.append(test_data[ix_random][1])

        net_out_value = self.sess.run(net_out, feed_dict={net_inp:X_data})
        pred_texts = self.decode_batch(net_out_value)
        
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
            ax2.set_yticks(list(range(len(LETTERS) + 1)))
            ax2.set_yticklabels(LETTERS + ['blank'])
            ax2.grid(False)
            for h in np.arange(-0.5, len(LETTERS) + 1 + 0.5, 1):
                ax2.axhline(h, linestyle='-', color='k', alpha=0.5, linewidth=1)

            #ax.axvline(x, linestyle='--', color='k')
            plt.show()

    def eval_image(self, image):
        net_inp = self.model.get_layer(name='the_input').input
        net_out = self.model.get_layer(name='softmax').output
  
        temp_image = image
        temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        temp_image = cv2.resize(temp_image, (self.img_w, self.img_h))
        temp_image = temp_image.astype(np.float32)
        temp_image /= 255

        X_data = []
        X_data.append(temp_image.T)

        X_data = np.array(X_data)
        X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))

        net_out_value = self.sess.run(net_out, feed_dict={net_inp:X_data})
        pred_texts = self.decode_batch(net_out_value)

        return pred_texts