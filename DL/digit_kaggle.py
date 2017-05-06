# encoding=utf-8
#author: wdw110
#Classification of kaggle digits recongnize by convolutional neural networks

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import *


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

batch_size = 128
num_classes = 10
epochs = 12

dropout_rate = 0.25
opt = Adam(lr=1e-4)
dopt = Adam(lr=1e-3)

img_rows, img_cols = 28, 28

#the data, shuffled and split between train and test sets
#path = os.path.join(os.getcwd(),'mnist_little.pkl.gz')
#(X_train, y_train), (X_test, y_test) = mnist.load_data(path)

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
Shape=X_train.shape[1:]
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
d_pre = d_pre.reshape(d_pre.shape[0], img_rows, img_cols, 1)
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


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

class_num=y_train.shape[1]
   

# Parametrize the image augmentation class
datagen = ImageDataGenerator(
	rotation_range = 20,
	width_shift_range = 0.15,
	height_shift_range = 0.15,
	shear_range = 0.4,
	zoom_range = 0.3,                    
	channel_shift_range = 0.1) 

## CNN MODEL ##
model = Sequential()

model.add(Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(LeakyReLU(0.2))

model.add(Dropout(dropout_rate))

model.add(Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu'))
model.add(LeakyReLU(0.2))

model.add(Dropout(dropout_rate))

model.add(Flatten())
model.add(Dense(256))
model.add(LeakyReLU(0.2))
model.add(Dropout(dropout_rate))
model.add(Dense(num_classes, activation='softmax'))


## LEARNING ##

# First, use AdaDelta for some epochs because AdaMax gets stuck
model.compile(loss='categorical_crossentropy',
			  optimizer='adadelta', 
			  metrics=['accuracy'])

# Fit
model.fit(X_train, y_train, batch_size=batch_size,
		  nb_epoch=1,
		  validation_data=(X_test,y_test),
		  verbose=1)

# Now, use AdaMax
model.compile(loss='categorical_crossentropy',
			  optimizer='adamax', 
			  metrics=['accuracy'])

# We want to keep the best model. This callback will store
# in a file the weights of the model with the highest validation accuracy
saveBestModel = ModelCheckpoint('best.kerasModelWeights', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True)

# Make the model learn using the image generator
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
					samples_per_epoch=len(X_train),
					nb_epoch=epochs,
					validation_data=(X_test,y_test),
					callbacks=[saveBestModel],
					verbose=1)

## PREDICTON ##

# Load the model with  the hightest validation accuracy
model.load_weights('best.kerasModelWeights')

# Predict the class (give the index in the one-hot vector of the most probable class)
Y_pred = model.predict_classes(d_pre)

# Save the predicitions in Kaggle format
np.savetxt("CNN_pred.csv", np.c_[range(1,len(Y_pre)),Y_pred], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%s')

