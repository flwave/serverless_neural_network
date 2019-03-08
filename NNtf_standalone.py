import tensorflow as tf
import random
import time
import datetime
import pickle
import numpy
import threading
import multiprocessing
import numpy as np
import s3func
import os

def add_weights(w1,w2):
	for i in range(len(w1)):
		for j in range(len(w1[i])):
			for k in range(len(w1[i][j])):
				w1[i][j][k]+=w2[i][j][k]
				
def sub_weights(w1,w2):
	for i in range(len(w1)):
		for j in range(len(w1[i])):
			for k in range(len(w1[i][j])):
				w1[i][j][k]-=w2[i][j][k]

def div_weights(w,d):
	for i in range(len(w)):
		for j in range(len(w[i])):
			for k in range(len(w[i][j])):
				w[i][j][k]/=d

def printbin(num,l):
	num=int(num)
	data=''
	for j in range(l):
		data+=str(num>>(l-1-j)&0b1)
	print data

def num2float8(num):
	anum=abs(num)
	anum=int(anum*2048)
	if anum==0:
		return chr(anum)
	if anum>61440:
		anum=61440
	check=32768
	for i in range(16):
		if (check>>i)&anum:
			temp=(28672>>i)&anum
			move=15-i-3
			if move>=0:
				if num>0:
					return chr((temp>>move)|(15-i)<<3)
				else:
					return chr((temp>>move)|(15-i)<<3|128)
			else:
				if num>0:
					return chr((temp<<-move)|(15-i)<<3)
				else:
					return chr((temp<<-move)|(15-i)<<3|128)
	
def float82num(float8):
	float8=ord(float8)
	if float8==0:
		return float(0.0)
	exp=(float8&0b01111000)>>3
	num=(float8&0b00000111)|0b1000
	pn=float8&0b10000000
	if pn:
		return -float(num)*(2**(exp-3-11))
	else:
		return float(num)*(2**(exp-3-11))


def gen_random_sample(layers,num):
	sample=[]
	ds=[]
	allnum=range(pow(2,layers[0]))
	random.shuffle(allnum)
	sampletemp=allnum[0:num]
	sampletemp.sort()
	for i in range(num):
		sample.append([2*(sampletemp[i]>>m&0b1)-1 for m in range(layers[0])])
		ds.append([2*round(random.random())-1 for i in range(layers[-1])])
	return sample,ds

def tftrain(layers,num):#normal
	[samples,ds]=gen_random_sample(layers,num)
	samples=np.array(samples)
	ds=np.array(ds)
	
	ds[0:len(ds)/2,:]=-1.0
	ds[len(ds)/2:,:]=1.0
	samples=np.append(samples,ds,axis=1)
	np.random.shuffle(samples)
	ds=samples[:,-layers[-1]:]
	samples=samples[:,0:-layers[-1]]
	print samples.shape,ds.shape
	st=time.time()
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	train=tf.train.GradientDescentOptimizer(1e-2).minimize(cost)
	#train=tf.train.AdamOptimizer(0.1).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	results=session.run(outputs,feed_dict={x:training_inputs})
	test_output=results[-1]
	test_output=list(test_output)
	for i in range(len(test_output)):
		for j in range(len(test_output[i])):
			if test_output[i][j]>=0:
				test_output[i][j]=1
			else:
				test_output[i][j]=-1
	error=0
	for i in range(len(training_outputs)):
		if training_outputs[i]!=test_output[i]:
			#print training_outputs[i],test_output[i]
			error+=1
	print float(error)/len(training_outputs)
	timerecord='0 '+str(100.0-100*float(error)/len(training_outputs))+'\n'
	
	#st=time.time()
	"""
	s3func.s3_download_file(0,'lf-source','data/model','/tmp/model',0,1,0)
	with open('/tmp/model', 'r') as f:
		temp=pickle.load(f)
	for l in range(len(layers)):
		if l>0:
			session.run(tf.assign(weights[l],temp[0][l]))
			session.run(tf.assign(biases[l],temp[1][l]))
	"""
	bn=10
	bs=len(samples)/bn
	for i in range(10):
		for b in range(bn):
			print [i,b]
			session.run(train,feed_dict={x:training_inputs[b*bs:(b+1)*bs,:],y:training_outputs[b*bs:(b+1)*bs,:]})
		et=time.time()
		#print 'time cost:----------------',time.time()-st
		results=session.run(outputs,feed_dict={x:training_inputs})
		test_output=results[-1]
		test_output=list(test_output)
		for i in range(len(test_output)):
			for j in range(len(test_output[i])):
				if test_output[i][j]>=0:
					test_output[i][j]=1
				else:
					test_output[i][j]=-1
		error=0
		for i in range(len(training_outputs)):
			if training_outputs[i]!=test_output[i]:
				#print training_outputs[i],test_output[i]
				error+=1
		print float(error)/len(training_outputs)
		timerecord+=str(et-st)+' '+str(100.0-100*float(error)/len(training_outputs))+'\n'
		st=time.time()
	with open('fcrecord','a') as fid:
		fid.write(timerecord+'\n')
	#print training_inputs
	#print training_outputs

def tftrainmul(layers,nums):#multiprocessing
	jobs=[]
	for i in range(20):
		#thread = threading.Thread(target=tftrain,args=(layers,nums))
		thread = multiprocessing.Process(target=tftrain,args=(layers,nums))
		jobs.append(thread)
		jobs[-1].start()
	for j in jobs:
		j.join()

def tfwrtrain(layers,num):#train,write,read,reinitialize and train
	[samples,ds]=gen_random_sample(layers,num)
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	
	with open('/tmp/model', 'r') as f:
		temp=pickle.load(f)
	weights=temp[0]
	biases=temp[1]
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(weights[l])
			biases[l]=tf.Variable(biases[l])
	for i in range(100):
		print i
		x=tf.placeholder(tf.float32,[None,layers[0]])
		y=tf.placeholder(tf.float32,[None,layers[-1]])
		
		outputs[0]=x
		for l in range(len(layers)):
			if l>0:
				outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
		
		cost=0.5*(y-outputs[-1])**2
		
		train=tf.train.GradientDescentOptimizer(0.01).minimize(cost)
		
		init=tf.global_variables_initializer()
		session=tf.Session()
		session.run(init)
		
		session.run(train,feed_dict={x:training_inputs,y:training_outputs})
		
		rew=session.run(weights)
		reb=session.run(biases)
		model=[rew,reb]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
		
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		weights=temp[0]
		biases=temp[1]
		for l in range(len(layers)):
			if l>0:
				weights[l]=tf.Variable(weights[l])
				biases[l]=tf.Variable(biases[l])
		outputs=[[] for i in range(len(layers))]
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	print session.run(outputs,feed_dict={x:training_inputs})
	print training_inputs
	print training_outputs

def tfwrtrain2(layers,num):#train,write,read,and train
	[samples,ds]=gen_random_sample(layers,num)
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	train=tf.train.AdamOptimizer(0.1).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	for i in range(100):
		print i
		session.run(train,feed_dict={x:training_inputs,y:training_outputs})
		
		rew=session.run(weights)
		reb=session.run(biases)
		model=[rew,reb]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
		
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		for l in range(len(layers)):
			if l>0:
				tf.assign(weights[l],temp[0][l])
				tf.assign(biases[l],temp[1][l])
	
	print session.run(outputs,feed_dict={x:training_inputs})
	print training_inputs
	print training_outputs
	
def tfwrbtrain(layers,num):#batch train,write,read,and batch train
	if num%2!=0:
		print 'please input even samples number'
		return 
	[samples,ds]=gen_random_sample(layers,num)
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	train=tf.train.AdamOptimizer(0.01).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	rew=session.run(weights)
	reb=session.run(biases)
	model=[rew,reb]
	with open('/tmp/model', 'w') as f:
		pickle.dump(model, f)
	for i in range(100):
		print i
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		for l in range(len(layers)):
			if l>0:
				tf.assign(weights[l],temp[0][l])
				tf.assign(biases[l],temp[1][l])
		
		batch_inputs=training_inputs[:num/2]
		batch_outputs=training_outputs[:num/2]
		session.run(train,feed_dict={x:batch_inputs,y:batch_outputs})
		
		rew0=session.run(weights)
		reb0=session.run(biases)
		
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		for l in range(len(layers)):
			if l>0:
				tf.assign(weights[l],temp[0][l])
				tf.assign(biases[l],temp[1][l])
		
		batch_inputs=training_inputs[num/2:]
		batch_outputs=training_outputs[num/2:]
		session.run(train,feed_dict={x:batch_inputs,y:batch_outputs})
		
		rew1=session.run(weights)
		reb1=session.run(biases)
		
		add_weights(rew0,rew1)
		div_weights(rew0,2)
		add_weights(reb0,reb1)
		div_weights(reb0,2)
		model=[rew0,reb0]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
	
	print session.run(outputs,feed_dict={x:training_inputs})
	print training_inputs
	print training_outputs
	
def tfregtrain(layers):#for regression
	training_inputs=[]
	training_outputs=[]
	"""
	for x in range(10):
		for y in range(10):
			x0=float(x)
			y0=float(y)
			training_inputs.append([(x0-5),(y0-2)])
			training_outputs.append([((x0-5))**2+((y0-2))**2])
	"""
	training_inputs=[[1.0],[5.0],[6.0]]
	training_outputs=[[2.0],[1.0],[3.0]]
	print training_inputs
	print training_outputs
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			if l<len(layers)-1:
				outputs[l]= tf.nn.relu(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
			else:
				outputs[l]= tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l])
			
	cost=0.5*(y-outputs[-1])**2
	
	train=tf.train.AdamOptimizer(0.01).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	for i in range(1000):
		print i
		session.run(train,feed_dict={x:training_inputs,y:training_outputs})
	results=session.run(outputs,feed_dict={x:training_inputs})
	test_output=results[-1]
	test_output=list(test_output)
	for i in range(len(test_output)):
		test_output[i]=list(test_output[i])
		test_output[i].append(training_outputs[i][0])
		test_output[i].append(abs(test_output[i][0]-test_output[i][1]))
		print test_output[i]
	training_inputs=[[i*0.1] for i in range(100)]
	st=time.time()
	results=session.run(outputs,feed_dict={x:training_inputs})
	print time.time()-st
	test_output=results[-1]
	test_output=list(test_output)
	plt.plot(training_inputs,test_output)
	plt.show()

def tfmultitrain(layers,num):#simulate lambda training
	[samples,ds]=gen_random_sample(layers,num)
	samples=np.array(samples)
	ds=np.array(ds)
	ds[0:len(ds)/2,:]=-1.0
	ds[len(ds)/2:,:]=1.0
	print ds
	samples=np.append(samples,ds,axis=1)
	np.random.shuffle(samples)
	ds=samples[:,-layers[-1]:]
	samples=samples[:,0:-layers[-1]]
	print samples.shape,ds.shape
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
			
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	#train=tf.train.AdamOptimizer(0.01).minimize(cost)
	train=tf.train.GradientDescentOptimizer(1e-2).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	st=time.time()
	
	s3func.s3_download_file(0,'lf-source','data/model_0_10','/tmp/model',0,1,0)
	with open('/tmp/model', 'r') as f:
		temp=pickle.load(f)
	for l in range(len(layers)):
		if l>0:
			session.run(tf.assign(weights[l],temp[0][l]))
			session.run(tf.assign(biases[l],temp[1][l]))
	
	rew=session.run(weights)
	reb=session.run(biases)
	startw=rew
	startb=reb
	
	nn=10
	bn=10
	bs=len(samples)/nn/bn
	for i in range(0):
		allw=[]
		allb=[]
		for n in range(nn):#simulate every parallel worker
			for l in range(len(layers)):
				if l>0:
					session.run(tf.assign(weights[l],startw[l]))
					session.run(tf.assign(biases[l],startb[l]))
			for b in range(bn):#every batch
				rs=n*bn*bs+b*bs
				re=n*bn*bs+(b+1)*bs
				print [i,n,b,rs,re]
				session.run(train,feed_dict={x:training_inputs[rs:re,:],y:training_outputs[rs:re,:]})
			if allw==[]:
				allw=session.run(weights)
				allb=session.run(biases)
			else:
				add_weights(allw,session.run(weights))
				add_weights(allb,session.run(biases))
		startw=allw
		startb=allb
	print 'time cost:----------------',time.time()-st
	results=session.run(outputs,feed_dict={x:training_inputs})
	test_output=results[-1]
	test_output=list(test_output)
	for i in range(len(test_output)):
		for j in range(len(test_output[i])):
			if test_output[i][j]>=0:
				test_output[i][j]=1
			else:
				test_output[i][j]=-1
	error=0
	for i in range(len(training_outputs)):
		if training_outputs[i]!=test_output[i]:
			#print training_outputs[i],test_output[i]
			error+=1
	print float(error)/len(training_outputs)
	"""
	s3func.s3_download_file(0,'lf-source','data/model_0_new','/tmp/model',0,1,0)
	with open('/tmp/model', 'r') as f:
		temp=pickle.load(f)
	for l in range(len(layers)):
		if l>0:
			session.run(tf.assign(weights[l],temp[0][l]))
			session.run(tf.assign(biases[l],temp[1][l]))
	results=session.run(outputs,feed_dict={x:training_inputs})
	test_output=results[-1]
	test_output=list(test_output)
	for i in range(len(test_output)):
		for j in range(len(test_output[i])):
			if test_output[i][j]>=0:
				test_output[i][j]=1
			else:
				test_output[i][j]=-1
	error=0
	for i in range(len(training_outputs)):
		if training_outputs[i]!=test_output[i]:
			error+=1
	print float(error)/len(training_outputs)
	"""
	#print training_inputs
	#print training_outputs
	
def tfmptrain():
	#2,2,1
	train_x=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
	train_y=np.array([[-1],[1],[1],[-1]])
	
	x=tf.placeholder(tf.float32,[None,2])
	y=tf.placeholder(tf.float32,[None,1])
	
	weight1=tf.Variable(tf.random_normal([2,2]))
	bias1=tf.Variable(tf.random_normal([1,2]))
	
	output1=tf.nn.tanh(tf.add(tf.matmul(x, weight1), bias1))
	
	weight2=tf.Variable(tf.random_normal([2,1]))
	bias2=tf.Variable(tf.random_normal([1,1]))
	
	output2=tf.nn.tanh(tf.add(tf.matmul(output1, weight2), bias2))
	
	params=[weight1,bias1,weight2,bias2]
	
	cost=0.5*(y-output2)**2
	
	eta=1e-2
	train=tf.train.GradientDescentOptimizer(eta).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	if os.path.exists('./tfmpmodel'):
		print 'load'
		with open('./tfmpmodel','r') as f:
			temp=pickle.load(f)
		for i in range(len(params)):
			session.run(tf.assign(params[i],temp[i]))
	else:
		print 'write'
		with open('./tfmpmodel','w') as f:
			pickle.dump(session.run(params),f)
	
	print session.run(params)
	print '========'
	for i in range(1):
		session.run(train,feed_dict={x:train_x,y:train_y})
	print session.run(params)
	
	results=session.run(output2,feed_dict={x:train_x})
	print results

def tfmptrain2():
	#2,2,1
	train_x=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
	train_y=np.array([[-1],[1],[1],[-1]])
	
	x=tf.placeholder(tf.float32,[None,2])
	y=tf.placeholder(tf.float32,[None,1])
	
	weight1=tf.Variable(tf.random_normal([2,2]))
	bias1=tf.Variable(tf.random_normal([1,2]))
	
	output1=tf.nn.tanh(tf.add(tf.matmul(x, weight1), bias1))
	
	error=tf.placeholder(tf.float32,[None,2])
	micost=error+output1-output1
	
	x2=tf.placeholder(tf.float32,[None,2])
	
	weight2=tf.Variable(tf.random_normal([2,1]))
	bias2=tf.Variable(tf.random_normal([1,1]))
	
	output2=tf.nn.tanh(tf.add(tf.matmul(x2, weight2), bias2))
	cost=0.5*(y-output2)**2
	
	params=[weight1,bias1,weight2,bias2]
	
	eta=1e-1
	train=tf.train.GradientDescentOptimizer(eta).minimize(cost)
	mitrain=tf.train.GradientDescentOptimizer(eta).minimize(micost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	if os.path.exists('./tfmpmodel'):
		print 'load'
		with open('./tfmpmodel','r') as f:
			temp=pickle.load(f)
		for i in range(len(params)):
			session.run(tf.assign(params[i],temp[i]))
	else:
		print 'write'
		with open('./tfmpmodel','w') as f:
			pickle.dump(session.run(params),f)
	
	#print session.run(params)
	#print '========'
	for i in range(2000):
		weight2b=session.run(params[2])
		o2=session.run(output1,feed_dict={x:train_x})
		session.run(train,feed_dict={x2:o2,y:train_y})
		weight2a=session.run(params[2])
		dweight2=weight2a-weight2b
		e=np.sum(dweight2,axis=1)
		e=np.array([e])
		session.run(mitrain,feed_dict={x:train_x,error:e})
	#print session.run(params)
	
	o2=session.run(output1,feed_dict={x:train_x})
	results=session.run(output2,feed_dict={x2:o2})
	print results

def tfpredict(layers,num,modelnum):
	[samples,ds]=gen_random_sample(layers,num)
	samples=np.array(samples)
	ds=np.array(ds)
	ds[0:len(ds)/2,:]=-1.0
	ds[len(ds)/2:,:]=1.0
	print ds
	samples=np.append(samples,ds,axis=1)
	np.random.shuffle(samples)
	ds=samples[:,-layers[-1]:]
	samples=samples[:,0:-layers[-1]]
	print samples.shape,ds.shape
	training_inputs=samples
	training_outputs=ds
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	outputs=[[] for i in range(len(layers))]
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
			
	x=tf.placeholder(tf.float32,[None,layers[0]])
	y=tf.placeholder(tf.float32,[None,layers[-1]])
	
	outputs[0]=x
	for l in range(len(layers)):
		if l>0:
			outputs[l]= tf.nn.tanh(tf.add(tf.matmul(outputs[l-1], weights[l]), biases[l]))
	
	cost=0.5*(y-outputs[-1])**2
	
	#train=tf.train.AdamOptimizer(0.01).minimize(cost)
	train=tf.train.GradientDescentOptimizer(1e-2).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	st=time.time()
	for mo in range(modelnum+1):
		s3func.s3_download_file(0,'lf-source','data/model_0_'+str(mo),'/tmp/model',0,1,0)
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		for l in range(len(layers)):
			if l>0:
				session.run(tf.assign(weights[l],temp[0][l]))
				session.run(tf.assign(biases[l],temp[1][l]))
		
		results=session.run(outputs,feed_dict={x:training_inputs})
		test_output=results[-1]
		test_output=list(test_output)
		for i in range(len(test_output)):
			for j in range(len(test_output[i])):
				if test_output[i][j]>=0:
					test_output[i][j]=1
				else:
					test_output[i][j]=-1
		error=0
		for i in range(len(training_outputs)):
			if training_outputs[i]!=test_output[i]:
				#print training_outputs[i],test_output[i]
				error+=1
		print 1.0-float(error)/len(training_outputs)


tftrain([20,100,100,100,100,100,1],1000000)
#tfmultitrain([20,100,100,100,100,100,1],1000000)
#tfpredict([20,100,100,100,100,100,1],1000000,15)
#tfregtrain([1,10,1])
#tftrainmul([100,100,100,100,100],100000)
#tfwrtrain2([2,10,1],4)
#tfwrbtrain([2,10,1],4)
#tfsgdtrain([100,100,100,100,100,1],10000)
#tfmptrain2()
