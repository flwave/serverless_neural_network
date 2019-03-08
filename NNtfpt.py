
"""
this model is for hyperparameter tunning
"""
import time
import math
import boto3
import base64
import json
import copy
import s3func
import subprocess
import os
import pickle
import datetime
from urllib2 import urlopen
import tensorflow as tf
import random
import numpy as np
import gzip

AWS_region='us-west-2'
AWS_S3_bucket='lf-src'
AWS_lambda_role='arn:aws:iam::xxx:role/lambdarole'

IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10

def add_weights(w1,w2):
	for i in range(len(w1)):
		for j in range(len(w1[i])):
			for k in range(len(w1[i][j])):
				w1[i][j][k]+=w2[i][j][k]

def div_weights(w,d):
	for i in range(len(w)):
		for j in range(len(w[i])):
			for k in range(len(w[i][j])):
				w[i][j][k]/=d

def ceil_step(n,s):
	n=float(n)
	if n>=0:
		if n%s>0:
			return s*int(n/s)+s
		else:
			return s*int(n/s)
	else:
		if n%s>0:
			return s*int(n/s)-s
		else:
			return s*int(n/s)

def insert_sort(n,seq):
	found=0
	for i in range(len(seq)):
		if n<=seq[i]:
			seq.insert(i,n)
			found=1
			break
	if found==0:
		seq.insert(len(seq),n)
		
def avg_no_abnormal(seq,thre):
	if len(seq)==0:
		return 0
	if len(seq)<=2:
		return sum(seq)/len(seq)
	search=[i+2 for i in range(len(seq)-2)]
	cut=-1
	for i in search:
		if seq[i]-seq[i-1]>=thre*(seq[i-1]-seq[0]):
			cut=i
	if cut==-1:
		return sum(seq)/len(seq)
	else:
		return sum(seq[:cut])/cut

def initialize_samples():
	s32 = boto3.resource('s3')
	s32.Bucket(AWS_S3_bucket).upload_file('./train-images-idx3-ubyte.gz', 'data/samples_mnist_x')
	s32.Bucket(AWS_S3_bucket).upload_file('./train-labels-idx1-ubyte.gz', 'data/samples_mnist_y')
	s32.Bucket(AWS_S3_bucket).upload_file('./t10k-images-idx3-ubyte.gz', 'data/samples_mnist_test_x')
	s32.Bucket(AWS_S3_bucket).upload_file('./t10k-labels-idx1-ubyte.gz', 'data/samples_mnist_test_y')

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

def invoke_lambda(client,name,payload):
	response = client.invoke(
		FunctionName=name,
		InvocationType='Event',
		Payload=json.dumps(payload)
	)

def invoke_lambda_test(client,payload):
	print '-'*20
	print payload
	
def startup(event):
	s3 = boto3.client('s3')
	data=''
	print 'start up'
	s3func.s3_clear_bucket(AWS_S3_bucket,'data/model_')
	s3func.s3_clear_bucket(AWS_S3_bucket,'flag/')
	s3func.s3_clear_bucket(AWS_S3_bucket,'error/')
	s3func.s3_clear_bucket(AWS_S3_bucket,'timestamp/')
	st=datetime.datetime.now()
	client=boto3.client('lambda',region_name = AWS_region)
	funclist = client.list_functions(
		MaxItems=5
	)
	mems=[512,1536,1536]
	for i in range(3):
		funcname='nntffunc_'+str(i)
		found=0
		for k in funclist['Functions']:
			if k['FunctionName']==funcname:
				found=1
				break
		if found:
			data+='found '+funcname+', delete\n'
			response = client.delete_function(
				FunctionName=funcname
			)
		response = client.create_function(
			Code={
			'S3Bucket': AWS_S3_bucket,
				'S3Key': 'funcz.lamf',
			},
			Description='',
			FunctionName=funcname,
			Handler='NNtfpt.lambda_handler',
			MemorySize=mems[i],
			Publish=True,
			Role=AWS_lambda_role,
			#Role='arn:aws:iam::657297083105:role/lambdarole',
			Runtime='python2.7',
			Timeout=300,
			VpcConfig={
			},
		)
		data+='create '+funcname+' with memory '+str(mems[i])+'\n'
	time.sleep(10)
	s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='flag/work_pt')
	stt=time.time()
	for i in range(event['mlayers'][-1]):
		hnumber=random.randint(1,3)
		event['layers']=[0 for l in range(hnumber+2)]
		for l in range(hnumber):
			event['layers'][l+1]=random.randint(10,100)
		event['lr']=random.uniform(0.001,0.1)
		event['batchnumber']=random.randint(10,1000)
		event['maxiter']=50
		event['pos']=i
		
		event['nworker']=event['mlayers'][-1]
		
		event['funcname']='nntffunc_1'
		event['state']=1
		event['pid']=0
		event['memory']=mems[1]
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	
	s3.put_object(Bucket=AWS_S3_bucket,Body=str(stt), Key='timestamp/timestamp_startup.tsp')

def monitor(event):
	st=datetime.datetime.now()
	stt=time.time()
	nworker=event['nworker']
	pid=event['pid']
	if 'roundtime' not in event.keys():
		event['roundtime']=250
	if 'waittime' not in event.keys():
		event['waittime']=event['roundtime']*2/3
	timer=s3func.timer([event['waittime'],event['roundtime']])
	s3 = boto3.client('s3')
	s32 = boto3.resource('s3')
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_pt','/tmp/work',0,1,0)
	if flag==0:
		print 'monitor terminated!!!!!!'
		return
	s3.put_object(Bucket=AWS_S3_bucket,Body=str(st), Key='timestamp/timestamp_monitor')
	finished=[0 for i in range(nworker)]
	timer.local_start(0)
	bresult=0.0
	bpos=0
	while 1:
		if sum(finished)==nworker:
			break
		for now in range(nworker):
			if finished[now]==0:
				tresult=timer.query()
				if tresult[0]==1:
					return 0
				flag=s3func.s3_download_file(s3,AWS_S3_bucket,'timestamp/timestamp_trainresult_'+str(pid)+'_'+str(now),'/tmp/result',0,1,0)
				if flag==1:
					finished[now]=1
					with open('/tmp/result', 'r') as f:
						temp=f.read()
					r=float(temp)
					if r>bresult:
						bresult=r
						bpos=now
	s3.put_object(Bucket=AWS_S3_bucket,Body=str([bresult,bpos]), Key='timestamp/timestamp_final_result')
	et=time.time()
	st=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_startup.tsp',0,1,0)
	filerecord=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'results/timecost',0,1,0)
	if filerecord==0:
		filerecord=''
	filerecord+=str(et-float(st))+'\n'
	s3.put_object(Bucket=AWS_S3_bucket,Body=filerecord, Key='results/timecost')
	"""
	client=boto3.client('lambda',region_name = AWS_region)
	if 'testtime' not in event.keys():
		event['testtime']=9
	elif event['testtime']<=0:
		s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='resuls/break')
		return
	inputs={'testtime':event['testtime']-1,'state':0,'mod':0,'batchnumber':20,'slayers':[],'mlayers':[1,1],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':60000,'maxiter':50,'nowiter':0,'roundtime':250,'rounditer':60}
	s3.put_object(Bucket=AWS_S3_bucket,Body=str(inputs), Key='results/now_start')
	invoke_lambda(client,'testfunc',inputs)
	"""
			
def train(event):
	st=datetime.datetime.now()
	stt=time.time()
	tcount=time.time()
	if 'roundtime' not in event.keys():
		event['roundtime']=250
	tend=event['roundtime']
	ns=event['ns']
	pos=event['pos']
	layers=event['layers']
	lr=event['lr']
	if 'batchnumber' not in event.keys():
		event['batchnumber']=1
	pid=event['pid']
	if 'testtime' not in event.keys():
		event['testtime']=10
	if 'waittime' not in event.keys():
		event['waittime']=tend*2/3
	waittime=event['waittime']
	timer=s3func.timer([waittime,tend])
	#waittime=tend/4
	if 'round' not in event.keys():
		event['round']=0
	else:
		event['round']+=1
	rounditer=event['rounditer']
	s3 = boto3.client('s3')
	s32 = boto3.resource('s3')
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_pt','/tmp/work',0,1,0)
	if flag==0:
		print 'terminated!!!!!!'
		return
	client=boto3.client('lambda',region_name = AWS_region)
	filerecord=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'results/result',0,1,0)
	if filerecord==0:
		filerecord=''
	filerecord+='====='+' starttime: '+str(st)+'\n'
	filerecord+='====='+str(stt)+'\n'
	data='train round '+str(event['round'])+', round time '+str(event['roundtime'])+', start at '+str(st)+' ##'+str(time.time())+'\n'
	data+='info: pos '+str(pos)+'\n'
	print '='*5,'train node',pos,'='*5,'train phase start'
	data+='start up time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round']))
	#=========================================read========================================
	stt=time.time()
	print '='*5,'train node',pos,'='*5,'downloading samples'
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_mnist_x','/tmp/mnist_x',0,1,0)
	if flag==0:
		print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read sample file x'
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_mnist_y','/tmp/mnist_y',0,1,0)
	if flag==0:
		print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read sample file y'
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_mnist_test_x','/tmp/mnist_test_x',0,1,0)
	if flag==0:
		print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read test file x'
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_mnist_test_y','/tmp/mnist_test_y',0,1,0)
	if flag==0:
		print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read test file y'
	train_x = extract_data('/tmp/mnist_x', 60000)
	train_y = extract_labels('/tmp/mnist_y', 60000)
	test_x = extract_data('/tmp/mnist_test_x', 10000)
	test_y = extract_labels('/tmp/mnist_test_y', 10000)
	train_x=train_x.reshape([60000,28*28])
	test_x=test_x.reshape([10000,28*28])
	data+='samples length: '+str(len(train_x))+'\n'
	#=========================================read========================================
	#=========================================initialize==================================
	stt=time.time()
	outputs,x,y,labels,train_op,weights,biases=model(layers,lr)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	data+='training initialize time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	#=========================================initialize==================================
	#=========================================LOOP start========================================
	num_iterations=event['maxiter']
	bn=event['batchnumber']
	alltime=0.0
	for it in range(num_iterations):
		st=time.time()
		bs=len(train_x)/bn
		for b in range(bn):
			batch_xs = train_x[b*bs:(b+1)*bs]
			batch_ys = train_y[b*bs:(b+1)*bs]
			sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})
		alltime+=time.time()-st
	result=sess.run(outputs,feed_dict={x: test_x})
	acc=(np.argmax(result[-1], axis=1)==test_y).mean()
	s3.put_object(Bucket=AWS_S3_bucket,Body=str(acc), Key='timestamp/timestamp_trainresult_'+str(pid)+'_'+str(pos))
	s3.put_object(Bucket=AWS_S3_bucket,Body=str([event['layers'],event['lr'],event['batchnumber'],event['maxiter']]), Key='timestamp/timestamp_traininfo_'+str(pid)+'_'+str(pos))
	if pos==event['nworker']-1:
		event['state']=2
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round']))

def lambda_handler(event,context):
	state=event['state']
	if state==0:#invoke
		startup(event)
	elif state==1:#work
		train(event)
	elif state==2:
		monitor(event)

#initialize_samples()
