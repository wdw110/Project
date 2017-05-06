# encoding=utf-8

import random
import numpy as np 
import pandas as pd 
import skimage.io as io
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

path1 = '/Users/wdw/Desktop/test/Project/kaggle_pic/gray/train/*.jpg'
path2 = '/Users/wdw/Desktop/test/Project/kaggle_pic/gray/test/*.jpg'

def label2int(ch):
	asciiVal = ord(ch)
	if(asciiVal<=57): #0-9
		asciiVal-=48
	elif(asciiVal<=90): #A-Z
		asciiVal-=55
	else: #a-z
		asciiVal-=61
	return asciiVal
	
def int2label(i):
	if(i<=9): #0-9
		i+=48
	elif(i<=35): #A-Z
		i+=55
	else: #a-z
		i+=61
	return chr(i)

def input_data(path):
	pp = []
	pic_arr = io.ImageCollection(path)
	for i in range(len(pic_arr)):
		tmp = (pic_arr[i]>200) * 1
		pp.append(tmp)
	return np.array(pp)

def next_batch(num): #num:每次随机选取图片的个数
	rows = random.sample(xrange(5500), num)
	args = train[rows,]
	label = Label[rows,]
	return [args, label]

def weight_varible(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

data = input_data(path1)
LL = pd.read_csv('./trainLabels.csv').values[:,1]
L2in = np.array(map(label2int, LL))
Label = sess.run(tf.one_hot(L2in, 62))
X_pre = input_data(path2)
train = data[:5500]
test = data[5500:]

# paras
W_conv1 = weight_varible([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer-1
x = tf.placeholder(tf.float32, [None, 32, 32])
x_image = tf.reshape(x, [-1, 32, 32, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer-2
W_conv2 = weight_varible([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_varible([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_varible([1024, 62])
b_fc2 = bias_variable([62])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
y_ = tf.placeholder(tf.float32, [None, 62])

# model training
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())


for i in range(15000):
	batch = next_batch(50)
	#print batch[0].shape,batch[1]
	if i % 100 == 0:
		train_accuacy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1],keep_prob:1.0})
		print 'step %d, training accuracy %g' % (i, train_accuacy)
	train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

# accuacy on test
print 'test accuracy %g' %(accuracy.eval(feed_dict={x:test, y_:Label[5500:], keep_prob:1.0}))


#预测分类：
pre = y_conv.eval(feed_dict={x:X_pre, keep_prob:1.0})
result = sess.run(tf.arg_max(pre, 1))
result = map(int2label, result)
df = pd.DataFrame(result, index=range(6284,len(result)+6284),columns=['Class'])
df.to_csv('pic_submission.csv')

