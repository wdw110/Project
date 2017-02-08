# encoding=utf-8

import cPickle
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

########################################
# define the classifer constructs
########################################

class LogisticRegression(object):
	def __init__(self, input, W=None, b=None):

		if W is None:
			fle = open("../model_param/lr_sgd_best.pkl")
			W, b = cPickle.load(fle)
			fle.close()

		self.W = W
		self.b = b

		self.outputs = T.nnet.softmax(T.dot(input, self.W) + b)

		self.pred = T.argmax(self.outputs, axis=1)

class MLP(object):
	def __init__(self, input, params=None):
		if params is None:
			fle = open("../model_param/mlp_best.pkl")
			params = cPickle.load(fle)
			fle.close()

		self.hidden_W, self.hidden_b, self.lr_W, self.lr_b = params

		self.hiddenlayer = T.tanh(T.dot(input, self.hidden_W) + self.hidden_b)

		self.outputs = T.nnet.softmax(T.dot(self.hiddenlayer, self.lr_W) \
					+ self.lr_b)

		self.pred = T.argmax(self.outputs, axis=1)

class CNN(object):
	def __init__(self, input, params=None):
		if params is None: 
			fle = open("../model_param/cnn_best.pkl")
			params = cPickle.load(fle)
			fle.close()

		################
		self.layer3_W, self.layer3_b, self.layer2_W, self.layer2_b, \
			self.layer1_W, self.layer1_b, self.layer0_W, self.layer0_b = params

		# compute layer0 
		self.conv_out0 = conv.conv2d(input=input, filters=self.layer0_W)
#                    filter_shape=(20, 1, 5, 5), image_shape=(1, 1, \
#                        28, 28))
		self.pooled_out0 = downsample.max_pool_2d(input=self.conv_out0, \
					ds=(2, 2), ignore_border=True)
		self.layer0_output = T.tanh(self.pooled_out0 + \
					self.layer0_b.dimshuffle('x', 0, 'x', 'x'))

		# compute layer1 
		self.conv_out1 = conv.conv2d(input=self.layer0_output, filters=self.layer1_W)
#                    filter_shape=(50, 20, 5, 5), image_shape=(1, 20, \
#                        12, 12))
		self.pooled_out1 = downsample.max_pool_2d(input=self.conv_out1, \
					ds=(2, 2), ignore_border=True)
		self.layer1_output = T.tanh(self.pooled_out1 + \
					self.layer1_b.dimshuffle('x', 0, 'x', 'x'))
		
		# compute layer2
		self.layer2_input = self.layer1_output.flatten(2)

		self.layer2_output = T.tanh(T.dot(self.layer2_input, self.layer2_W) + \
					self.layer2_b)

		# compute layer3
		self.outputs = T.nnet.softmax(T.dot(self.layer2_output, self.layer3_W)\
					+ self.layer3_b)

		self.pred = T.argmax(self.outputs, axis=1)

########################################
# build classifier
########################################

def lr(input):
	input.shape = 1, -1

	x = T.fmatrix('x')
	classifer = LogisticRegression(input=x)

	get_p_y = theano.function(inputs=[x], outputs=classifer.outputs)
	pred_y = theano.function(inputs=[x], outputs=classifer.pred)
	return (get_p_y(input), pred_y(input))

def mlp(input):
	input.shape = 1, -1

	x = T.fmatrix('x')
	classifer = MLP(input=x)

	get_p_y = theano.function(inputs=[x], outputs=classifer.outputs)
	pred_y = theano.function(inputs=[x], outputs=classifer.pred)
	return (get_p_y(input), pred_y(input))

def cnn(input):
	input.shape = (1, 1, 28, 28)
	x = T.dtensor4('x')
	classifer = CNN(input=x)
	get_p_y = theano.function(inputs=[x], outputs=classifer.outputs)
	pred_y = theano.function(inputs=[x], outputs=classifer.pred)
	return (get_p_y(input), pred_y(input))