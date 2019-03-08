import numpy as np
import tensorflow as tf
import gzip
import os
import sys
import time
import math
import s3func
import pickle
import boto3
import random

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def model(layers,lr):
	IMAGE_SIZE = 28
	NUM_CLASSES = 10
	IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
	layers[0]=IMAGE_PIXELS
	layers[-1]=NUM_CLASSES
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None])
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.truncated_normal([layers[l-1], layers[l]],stddev=1.0 / math.sqrt(float(layers[l-1]))))
			biases[l]=tf.Variable(tf.zeros([layers[l]]))
	
	for l in range(len(layers)):
		if l==0:
			outputs[l]=x
		elif l>0 and l<len(layers)-1:
			outputs[l]=tf.nn.relu(tf.matmul(outputs[l-1], weights[l]) + biases[l])
		elif l==len(layers)-1:
			outputs[l]=tf.matmul(outputs[l-1], weights[l]) + biases[l]
	labels = tf.to_int64(y)
	loss=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=outputs[-1])
	optimizer = tf.train.GradientDescentOptimizer(lr)
	train_op = optimizer.minimize(loss)
	
	return outputs,x,y,labels,train_op,weights,biases

def extract_data(filename, num_images):
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
		data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
		return data


def extract_labels(filename, num_images):
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return labels

def top():
	st=time.time()

	train_x = extract_data('train-images-idx3-ubyte.gz', 60000)
	train_y = extract_labels('train-labels-idx1-ubyte.gz', 60000)
	test_x = extract_data('t10k-images-idx3-ubyte.gz', 10000)
	test_y = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

	train_x=train_x.reshape([60000,28*28])
	test_x=test_x.reshape([10000,28*28])

	num_iterations=50
	bn=600
	data=''
	for searchpoints in [int(sys.argv[1])]:
		for times in range(1):
			gst=time.time()
			best=0.0
			bests=[]
			#data='0 0\n'
			for loop in range(searchpoints):
				hnumber=random.randint(1,3)
				layers=[0 for l in range(hnumber+2)]
				for l in range(hnumber):
					layers[l+1]=random.randint(10,100)
				lr=random.uniform(0.001,0.1)
				bn=random.randint(10,1000)
				
				outputs,x,y,labels,train_op,weights,biases=model(layers,lr)
				sess = tf.Session()
				sess.run(tf.global_variables_initializer())
				
				alltime=0.0
				for it in range(num_iterations):
					bs=len(train_x)/bn
					print [searchpoints,times,loop,it]
					for b in range(bn):
						batch_xs = train_x[b*bs:(b+1)*bs]
						batch_ys = train_y[b*bs:(b+1)*bs]
						sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
				print 'test'
				result=sess.run(outputs,feed_dict={x: test_x})
				acc=(np.argmax(result[-1], axis=1)==test_y).mean()
				if acc>best:
					best=acc
				bests=[layers,lr,bn]
				#data+=str(loop)+' '+str(time.time()-st)+'\n'
			print time.time()-gst
			data+=str(searchpoints)+' '+str(time.time()-gst)+'\n'
		with open('hyperrecord','a') as f:
			f.write(data+'\n')
			#print best,bests
			#print time.time()-st
			#print data

top()

