import time
import math
import boto3
import base64
import json
import copy
import subprocess
import os
import pickle
import datetime
from urllib2 import urlopen
import tensorflow as tf
import random
import numpy as np
import s3func
import sys

def get_data_set(name="train", cifar=10):
	x = None
	y = None
	l = None

	folder_name = "cifar_10" if cifar == 10 else "cifar_100"

	f = open('./cifar_10/batches.meta', 'rb')
	datadict = pickle.load(f)
	f.close()
	l = datadict['label_names']

	if name is "train":
		for i in range(5):
			f = open('./cifar_10/data_batch_' + str(i + 1), 'rb')
			datadict = pickle.load(f)
			f.close()

			_X = datadict["data"]
			_Y = datadict['labels']

			_X = np.array(_X, dtype=float) / 255.0
			_X = _X.reshape([-1, 3, 32, 32])
			_X = _X.transpose([0, 2, 3, 1])
			_X = _X.reshape(-1, 32*32*3)

			if x is None:
				x = _X
				y = _Y
			else:
				x = np.concatenate((x, _X), axis=0)
				y = np.concatenate((y, _Y), axis=0)

	elif name is "test":
		f = open('./cifar_10/test_batch', 'rb')
		datadict = pickle.load(f)
		f.close()

		x = datadict["data"]
		y = np.array(datadict['labels'])

		x = np.array(x, dtype=float) / 255.0
		x = x.reshape([-1, 3, 32, 32])
		x = x.transpose([0, 2, 3, 1])
		x = x.reshape(-1, 32*32*3)

	def dense_to_one_hot(labels_dense, num_classes=10):
		num_labels = labels_dense.shape[0]
		index_offset = np.arange(num_labels) * num_classes
		labels_one_hot = np.zeros((num_labels, num_classes))
		labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

		return labels_one_hot

	return x, dense_to_one_hot(y), l

def initialize_samples():
	train_x, train_y, _ = get_data_set()
	s3 = boto3.resource('s3')
	split=500
	fn=len(train_x)/split
	print fn
	for n in range(fn):
		print n
		with open('/tmp/samples', 'w') as f:
			pickle.dump({'data':train_x[n*split:(n+1)*split],'label':train_y[n*split:(n+1)*split]}, f)
		s3.Bucket('lf-sourcet').upload_file('/tmp/samples', 'data/samples_cifar_'+str(n))

def model_old():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 10
	_RESHAPE_SIZE = 4*4*128

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

	def variable_with_weight_decay(name, shape, stddev, wd):
		dtype = tf.float32
		var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def variable_on_cpu(name, shape, initializer):
		with tf.device('/cpu:0'):
			dtype = tf.float32
			var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		return var

	with tf.variable_scope('conv1') as scope:
		kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(x_image, kernel, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name=scope.name)
	tf.summary.histogram('Convolution_layers/conv1', conv1)
	tf.summary.scalar('Convolution_layers/conv1', tf.nn.zero_fraction(conv1))

	norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
	pool1 = tf.nn.max_pool(norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

	with tf.variable_scope('conv2') as scope:
		kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv2 = tf.nn.relu(pre_activation, name=scope.name)
	tf.summary.histogram('Convolution_layers/conv2', conv2)
	tf.summary.scalar('Convolution_layers/conv2', tf.nn.zero_fraction(conv2))

	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

	with tf.variable_scope('conv3') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv3 = tf.nn.relu(pre_activation, name=scope.name)
	tf.summary.histogram('Convolution_layers/conv3', conv3)
	tf.summary.scalar('Convolution_layers/conv3', tf.nn.zero_fraction(conv3))

	with tf.variable_scope('conv4') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv4 = tf.nn.relu(pre_activation, name=scope.name)
	tf.summary.histogram('Convolution_layers/conv4', conv4)
	tf.summary.scalar('Convolution_layers/conv4', tf.nn.zero_fraction(conv4))

	with tf.variable_scope('conv5') as scope:
		kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
		conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv5 = tf.nn.relu(pre_activation, name=scope.name)
	tf.summary.histogram('Convolution_layers/conv5', conv5)
	tf.summary.scalar('Convolution_layers/conv5', tf.nn.zero_fraction(conv5))

	norm3 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
	pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
	print pool3
	with tf.variable_scope('fully_connected1') as scope:
		reshape = tf.reshape(pool3, [-1, _RESHAPE_SIZE])
		dim = reshape.get_shape()[1].value
		weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
		biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
		local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
	tf.summary.histogram('Fully connected layers/fc1', local3)
	tf.summary.scalar('Fully connected layers/fc1', tf.nn.zero_fraction(local3))

	with tf.variable_scope('fully_connected2') as scope:
		weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
		biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
		local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
	tf.summary.histogram('Fully connected layers/fc2', local4)
	tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(local4))

	with tf.variable_scope('output') as scope:
		weights = variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
		biases = variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
		softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
	tf.summary.histogram('Fully connected layers/output', softmax_linear)

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	y_pred_cls = tf.argmax(softmax_linear, axis=1)
	
	return x, y, softmax_linear, global_step, y_pred_cls

def predict_test(show_confusion_matrix=False):
	'''
		Make prediction for all images in test_x
	'''
	i = 0
	predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
	while i < len(test_x):
		j = min(i + _BATCH_SIZE, len(test_x))
		batch_xs = test_x[i:j, :]
		batch_ys = test_y[i:j, :]
		predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
		i = j

	correct = (np.argmax(test_y, axis=1) == predicted_class)
	acc = correct.mean()*100
	correct_numbers = correct.sum()
	print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

	if show_confusion_matrix is True:
		cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
		for i in range(_CLASS_SIZE):
			class_name = "({}) {}".format(i, test_l[i])
			print(cm[i, :], class_name)
		class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
		print("".join(class_numbers))

	return acc

def model_p():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 10
	_RESHAPE_SIZE = 8*8*64

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

	def variable_with_weight_decay(name, shape, stddev, wd):
		dtype = tf.float32
		var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def variable_on_cpu(name, shape, initializer):
		with tf.device('/cpu:0'):
			dtype = tf.float32
			var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		return var

	#with tf.variable_scope('conv1') as scope:
	conv1_kernel = variable_with_weight_decay('weights1', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
	conv1_conv = tf.nn.conv2d(x_image, conv1_kernel, [1, 1, 1, 1], padding='SAME')
	conv1_biases = variable_on_cpu('biases1', [64], tf.constant_initializer(0.0))
	conv1_pre_activation = tf.nn.bias_add(conv1_conv, conv1_biases)
	conv1 = tf.nn.relu(conv1_pre_activation, name='conv1')
	
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
	
	#with tf.variable_scope('conv2') as scope:
	conv2_kernel = variable_with_weight_decay('weights2', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
	conv2_conv = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
	conv2_biases = variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
	conv2_pre_activation = tf.nn.bias_add(conv2_conv, conv2_biases)
	conv2 = tf.nn.relu(conv2_pre_activation, name='conv2')

	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	print pool2.shape
	#with tf.variable_scope('fully_connected1') as scope:
	local3_reshape = tf.reshape(pool2, [-1, _RESHAPE_SIZE])
	local3_dim = local3_reshape.get_shape()[1].value
	local3_weights = variable_with_weight_decay('weights3', shape=[local3_dim, 384], stddev=0.04, wd=0.004)
	local3_biases = variable_on_cpu('biases3', [384], tf.constant_initializer(0.1))
	local3 = tf.nn.relu(tf.matmul(local3_reshape, local3_weights) + local3_biases, name='local3')

	#with tf.variable_scope('fully_connected2') as scope:
	local4_weights = variable_with_weight_decay('weights4', shape=[384, 192], stddev=0.04, wd=0.004)
	local4_biases = variable_on_cpu('biases4', [192], tf.constant_initializer(0.1))
	local4 = tf.nn.relu(tf.matmul(local3, local4_weights) + local4_biases, name='local4')

	#with tf.variable_scope('output') as scope:
	output_weights = variable_with_weight_decay('weights5', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
	output_biases = variable_on_cpu('biases5', [_NUM_CLASSES], tf.constant_initializer(0.0))
	softmax_linear = tf.add(tf.matmul(local4, output_weights), output_biases, name='output')

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	y_pred_cls = tf.argmax(softmax_linear, axis=1)
	
	params=[conv1_kernel,conv1_biases,conv2_kernel,conv2_biases,local3_weights,local3_biases,local4_weights,local4_biases,output_weights,output_biases]
	
	return x, y, softmax_linear, global_step, y_pred_cls , params

def model():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 10
	_RESHAPE_SIZE = 8*8*64

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

	def variable_with_weight_decay(name, shape, stddev, wd):
		dtype = tf.float32
		var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
		if wd is not None:
			weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
			tf.add_to_collection('losses', weight_decay)
		return var

	def variable_on_cpu(name, shape, initializer):
		with tf.device('/cpu:0'):
			dtype = tf.float32
			var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		return var

	#with tf.variable_scope('conv1') as scope:
	#conv1_kernel = variable_with_weight_decay('weights1', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
	conv1_kernel = tf.Variable(tf.random_normal([5, 5, 3, 64]))
	conv1_conv = tf.nn.conv2d(x_image, conv1_kernel, [1, 1, 1, 1], padding='SAME')
	#conv1_biases = variable_on_cpu('biases1', [64], tf.constant_initializer(0.0))
	conv1_biases = tf.Variable(tf.random_normal([64]))
	conv1_pre_activation = tf.nn.bias_add(conv1_conv, conv1_biases)
	conv1 = tf.nn.relu(conv1_pre_activation, name='conv1')
	
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
	
	#with tf.variable_scope('conv2') as scope:
	#conv2_kernel = variable_with_weight_decay('weights2', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
	conv2_kernel = tf.Variable(tf.random_normal([5, 5, 64, 64]))
	conv2_conv = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
	#conv2_biases = variable_on_cpu('biases2', [64], tf.constant_initializer(0.1))
	conv2_biases = tf.Variable(tf.random_normal([64]))
	conv2_pre_activation = tf.nn.bias_add(conv2_conv, conv2_biases)
	conv2 = tf.nn.relu(conv2_pre_activation, name='conv2')

	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	print pool2.shape
	#with tf.variable_scope('fully_connected1') as scope:
	local3_reshape = tf.reshape(pool2, [-1, _RESHAPE_SIZE])
	local3_dim = local3_reshape.get_shape()[1].value
	#local3_weights = variable_with_weight_decay('weights3', shape=[local3_dim, 384], stddev=0.04, wd=0.004)
	local3_weights = tf.Variable(tf.random_normal([local3_dim, 384]))
	#local3_biases = variable_on_cpu('biases3', [384], tf.constant_initializer(0.1))
	local3_biases = tf.Variable(tf.random_normal([384]))
	local3 = tf.nn.relu(tf.matmul(local3_reshape, local3_weights) + local3_biases, name='local3')

	#with tf.variable_scope('fully_connected2') as scope:
	#local4_weights = variable_with_weight_decay('weights4', shape=[384, 192], stddev=0.04, wd=0.004)
	local4_weights = tf.Variable(tf.random_normal([384, 192]))
	#local4_biases = variable_on_cpu('biases4', [192], tf.constant_initializer(0.1))
	local4_biases = tf.Variable(tf.random_normal([192]))
	local4 = tf.nn.relu(tf.matmul(local3, local4_weights) + local4_biases, name='local4')

	#with tf.variable_scope('output') as scope:
	#output_weights = variable_with_weight_decay('weights5', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
	output_weights = tf.Variable(tf.random_normal([192, _NUM_CLASSES]))
	#output_biases = variable_on_cpu('biases5', [_NUM_CLASSES], tf.constant_initializer(0.0))
	output_biases = tf.Variable(tf.random_normal([_NUM_CLASSES]))
	softmax_linear = tf.add(tf.matmul(local4, output_weights), output_biases, name='output')

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	y_pred_cls = tf.argmax(softmax_linear, axis=1)
	
	params=[conv1_kernel,conv1_biases,conv2_kernel,conv2_biases,local3_weights,local3_biases,local4_weights,local4_biases,output_weights,output_biases]
	
	return x, y, softmax_linear, global_step, y_pred_cls , params

def top(pid):
	
	def predict_test(show_confusion_matrix=False):
		'''
			Make prediction for all images in test_x
		'''
		i = 0
		predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
		while i < len(test_x):
			j = min(i + _BATCH_SIZE, len(test_x))
			batch_xs = test_x[i:j, :]
			batch_ys = test_y[i:j, :]
			predicted_class[i:j] = sess.run(y_pred_cls, feed_dict={x: batch_xs, y: batch_ys})
			i = j

		correct = (np.argmax(test_y, axis=1) == predicted_class)
		acc = correct.mean()*100
		correct_numbers = correct.sum()
		print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

		if show_confusion_matrix is True:
			cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
			for i in range(_CLASS_SIZE):
				class_name = "({}) {}".format(i, test_l[i])
				print(cm[i, :], class_name)
			class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
			print("".join(class_numbers))

		return acc
	
	_IMG_SIZE = 32
	_NUM_CHANNELS = 3
	_BATCH_SIZE = 100
	_CLASS_SIZE = 10
	_ITERATION = 20
	_SAVE_PATH='./save/'

	train_x, train_y, _ = get_data_set()
	test_x, test_y,_ = get_data_set("test")
	x, y, output, global_step, y_pred_cls , params= model()
	
	startt=0.0
	st=time.time()
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
	a=tf.train.exponential_decay(0.1,1,1,0.1,staircase=True)
	optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)


	sess = tf.Session()
	#saver = tf.train.Saver()

	sess.run(tf.global_variables_initializer())
	
	startt+=time.time()-st

	train_x=train_x
	train_y=train_y

	#fid.close()
	
	st=time.time()
	
	train_x=np.append(train_x,train_y,axis=1)
	np.random.shuffle(train_x)
	train_y=train_x[:,-10:]
	train_x=train_x[:,0:-10]
	print train_x.shape,train_y.shape

	nn=100
	bn=500
	bs=len(train_x)/nn/bn
	
	startt+=time.time()-st
	
	alltime=0.0
	#fid=open('../record/cifarrecord','a')
	#lr=0.01
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
	for it in range(_ITERATION):
		st=time.time()
		bs=len(train_x)/bn
		for b in range(bn):
			print [it,b]
			"""
			randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
			batch_xs = train_x[randidx]
			batch_ys = train_y[randidx]
			"""
			batch_xs = train_x[b*bs:(b+1)*bs]
			batch_ys = train_y[b*bs:(b+1)*bs]
			i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
			"""
			if b==bn/2:
				lr=lr
				optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)
			"""
		thist=time.time()-st+startt
		startt=0.0
		alltime+=time.time()-st
		#fid.write(str(time.time()-st)+' ')
		print 'now cost-----',alltime
		print 'test'
		acc = predict_test()
		#fid.write(str(acc)+'\n')
		timerecord+=str(thist)+' '+str(acc)+'\n'
		"""
		with open('recordtf','a') as fid:
			fid.write(str(acc)+' iteration '+str(it)+'\n')
		"""
			
	def trainmul(num_iterations):
		initmodel=[]
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss, global_step=global_step)
		for i in range(len(params)):
			initmodel.append(sess.run(params[i]))
		for it in range(num_iterations):
			allparams=[]
			for n in range(nn):
				print [it,n]
				for i in range(len(initmodel)):
					sess.run(tf.assign(params[i],initmodel[i]))
				randidx = np.random.randint(len(train_x), size=bs)
				batch_xs = train_x[randidx]
				batch_ys = train_y[randidx]
				i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
				"""
				for b in range(bn):
					rs=n*bn*bs+b*bs
					re=n*bn*bs+(b+1)*bs
					print [it,n,b,rs,re]
					batch_xs = train_x[rs:re]
					batch_ys = train_y[rs:re]
					i_global, _ = sess.run([global_step, optimizer], feed_dict={x: batch_xs, y: batch_ys})
				"""
				pw=[]
				for i in range(len(params)):
					pw.append(sess.run(params[i]))
				if allparams==[]:
					allparams=pw
				else:
					for i in range(len(allparams)):
						allparams[i]=allparams[i]+pw[i]
			for i in range(len(allparams)):
				allparams[i]=allparams[i]/nn
			initmodel=allparams
			for i in range(len(initmodel)):
				sess.run(tf.assign(params[i],initmodel[i]))
			print 'test'
			acc = predict_test()
			with open('recordtf','a') as fid:
				fid.write(str(acc)+' iteration '+str(it)+'\n')		
			
	#train(_ITERATION)
	with open('cnnrecord','a') as f:
		f.write(timerecord+'\n')
	#trainmul(_ITERATION)
	#initialize_samples()
	#saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)

top(int(sys.argv[1]))
