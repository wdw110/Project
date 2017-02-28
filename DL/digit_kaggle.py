# encoding=utf-8
#author: wdw110
#Classification of kaggle digits recongnize by convolutional neural networks

import os,random
import pandas as pd
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
#import matplotlib.pyplot as plt

import cPickle, random, sys, keras
from keras.models import Model
from IPython import display
from keras.utils import np_utils

from keras.models import load_model

data = pd.read_csv('/Users/wdw/Desktop/test/Kaggle_data/digit_recongnizer/train.csv').values
d_train = data[0:40000,:]
d_test = data[40000:,:]
d_pre = pd.read_csv('/Users/wdw/Desktop/test/Kaggle_data/digit_recongnizer/test.csv').values

m = d_train.shape[0]
n = d_test.shape[0]
X_train = d_train[:,1:].reshape(m, 28, 28)
y_train = d_train[:,0]
X_test = d_test[:,1:].reshape(n, 28, 28)
y_test = d_test[:,0]

img_rows, img_cols = 28, 28

#the data, shuffled and split between train and test sets
path = os.path.join(os.getcwd(),'mnist_little.pkl.gz')
#(X_train, y_train), (X_test, y_test) = mnist.load_data(path)

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
Shape=X_train.shape[1:]
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
d_pre = d_pre.reshape(d_pre.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
d_pre = d_pre.astype('float32')
X_train /= 255
X_test /= 255
d_pre /= 255



print np.min(X_train), np.max(X_train)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
print(d_pre.shape[0], 'predict samples')
print '************************'
print y_train.shape
print y_test.shape
print '************************'

shp = X_train.shape[1:]

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

class_num=y_train.shape[1]
   

# discriminator 

d_input = Input(shape=shp)
H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
H = Dense(256)(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
Features=H
d_V = Dense(class_num,activation='softmax')(H)
discriminator = Model(d_input,d_V)
discriminator.compile(loss='binary_crossentropy', optimizer=dopt,metrics=['accuracy'])


#discriminator.summary()
discriminator.fit(X_train,y_train ,nb_epoch=1, batch_size=128,verbose=1)
Extractor=Model(d_input,Features)


Ftr = Extractor.predict(X_train) # Ftr is 256 dim
Fts = Extractor.predict(X_test)
Ftp = Extractor.predict(d_pre)

import keras 
adm=keras.optimizers.Adam(lr=0.2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)


input_v=Input(shape=[256])
output_v=Dense(10,activation='sigmoid',init='glorot_normal')(input_v)
Transformer= Model(input_v, output_v)
Transformer.compile(optimizer=adm, loss='binary_crossentropy',metrics=['accuracy'])
Transformer.fit(Ftr, y_train, validation_data=(Fts, y_test),nb_epoch=2000,batch_size=1000,shuffle=True,verbose=1)

scores = Transformer.evaluate(Fts, y_test, verbose=0)
print("Baseline Error X_test test: %.2f%%" % (100-scores[1]*100))

classes = Transformer.predict(Ftp)
result = classes.argmax(axis=1)
df = pd.DataFrame(result,index=range(1,classes.shape[0]+1),columns=['Label'])
df.to_csv('submission.csv')

print type(classes),classes.shape

