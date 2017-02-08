# encoding=utf-8

import os 
import sys
import timeit
import numpy
import theano
import theano.tensor as T 
from logistic_sgd import LogisticRegression,load_data

class HiddenLayer(object):
	"""docstring for HiddenLayer"""
	def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.tanh):
		"""
		2. rng: numpy.random.RandomState是随机数生成器用于初始化W
		3. input: 类型为theano.tensor.dmatrix,是一个二维的矩阵(n_examples, n_in)
		第一维表示训练样本的个数，第二位表示特征维数，比如:input(i,j)表示第i个样本的第j个特征值
		4. n_in: 输入特征数，也就是输入神经元的个数
		5. n_out: 输出神经元的个数
		6. W如果有输入，那么为(n_in,n_out)大小的矩阵
		7. b如果有输入，那么为(n_out,)的向量
		8. activation: 激活函数选项
		"""
		self.input = input

		'''W的初始化选择[-a,a]进行均匀分布采样，其中如果激活函数选择tanh，则a=sqrt(6./(本层输入神经元数
		+本层输出神经元数))。如果激活函数是选择sigmod，那么a=4*sqrt(6./(本层输入神经元数+本层输出神经元数))
		dtype类型需要设置成theano.config.floatX，这样GPU才能调用'''
		if W is None: # 如果外部没有输入W，那么创建W
			W_values = numpy.asarray(
				rng.uniform(
					low = -numpy.sqrt(6. / (n_in + n_out)),
					high = numpy.sqrt(6. / (n_in + n_out)),
					size = (n_in, n_out)
				),
				dtype = theano.config.floatX 
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None: # 如果外部没有输入b，那么创建b
			b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b
		#激活函数映射
		lin_output = T.dot(input, self.W) + self.b 
		self.output = (
			lin_output if activation is None else activation(lin_output)
		)
		#模型参数
		self.params = [self.W, self.b]


# MLP类是三层神经网络：输入，隐层，输出，第一层为简单的人工神经网络，第二层为逻辑回归层
class MLP(object):
	"""docstring for MLP"""
	def __init__(self, rng, input, n_in, n_hidden, n_out):
		"""
		n_in: 输入层神经元个数
		n_hidden: 隐层神经元个数
		n_out: 输出层神经元个数
		"""

		#创建隐藏层
		self.hiddenLayer = HiddenLayer(
			rng = rng,
			input = input,
			n_in = n_in,
			n_out = n_hidden,
			activation = T.tanh
		)

		#创建逻辑回归层
		self.logRegressionLayer = LogisticRegression(
			input = self.hiddenLayer.output,
			n_in = n_hidden,
			n_out = n_out
		)

		#整个网络的L1正则项，也就是使得所有的连接权值W的绝对值总和最小化
		self.L1 = (
			abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum()
		)

		#整个网络的L2正则项，也就是使得所有的链接权重的平方和最小化
		self.L2_sqr = (
			(self.hiddenLayer.W**2).sum() + (self.logRegressionLayer.W**2).sum()
		)

		self.negative_log_likelihood = (
			self.logRegressionLayer.negative_log_likelihood
		)

		# same holds for the function computing the number of errors
		self.errors = self.logRegressionLayer.errors

		#把所有的参数保存在用同一个列表中，这样后面就可以直接求导
		self.params = self.hiddenLayer.params + self.logRegressionLayer.params

		self.input = input

#手写数字识别测试，BP算法
def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
	"""
	leanring_rate: 梯度下降法的学习率
	L1_reg: L1正则项的权值
	L2_reg: L2正则项的权值
	n_epochs: 最大迭代次数
	dataset: 里面的数据是28*28的手写图片数据
	"""
	datasets = load_data(dataset)

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	#批量训练，计算总共有多少批
	n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
	n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
	n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

	#分配符号变量
	index = T.lscalar() # index to a [mini]batch
	x = T.matrix('x')  # 训练数据
	y = T.ivector('y') # 训练数据的标签
	rng = numpy.random.RandomState(1234)

	#构建三层神经网络
	classifier = MLP(
		rng = rng,
		input = x,
		n_in = 28*28,
		n_hidden = n_hidden,
		n_out = 10
	)

	#计算损失函数
	cost = (
		classifier.negative_log_likelihood(y) 
		+ L1_reg * classifier.L1
		+ L2_reg * classifier.L2_sqr
	)
	#损失函数求解偏导数
	gparams = [T.grad(cost, param) for param in classifier.params]
	#梯度下降法参数更新
	updates = [
		(param, param-learning_rate*gparam)
		for param, gparam in zip(classifier.params, gparams)
	]

	#定义训练函数
	train_model = theano.function(
		inputs = [index],
		outputs = cost,
		updates = updates,
		givens = {
			x: train_set_x[index*batch_size: (index+1)*batch_size],
			y: train_set_y[index*batch_size: (index+1)*batch_size]
		}
	)

	#运行
	epoch = 0
	while epoch < 100:
		cost = 0
		for minibatch_index in xrange(n_train_batches):
			cost += train_model(minibatch_index)
		print 'epoch:',epoch,'	error:',cost/n_train_batches
		epoch += 1
		
if __name__ == '__main__':
	test_mlp()

