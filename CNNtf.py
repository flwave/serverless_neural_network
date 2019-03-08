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

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 100
_CLASS_SIZE = 10
_ITERATION = 100
_SAVE_PATH='./save/'

AWS_region='us-west-2'
AWS_S3_bucket='lf-src'
AWS_lambda_role='arn:aws:iam::xxx:role/lambdarole'

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

def find_median(lis):
	lis.sort()
	return lis[len(lis)/2]

def estimate_best_layers(n,k):
	mlayers=[0 for i in range(k+2)]
	mlayers[0]=1
	mlayers[-1]=n
	mlayers[1]=pow(n,1.0/(k+1))
	for l in range(k-1):
		mlayers[-2-l]=pow(mlayers[1],k-l)
	for l in range(k+2):
		mlayers[l]=int(round(mlayers[l]))
	return mlayers

def estimate_merging(n):
	k=int(math.log(n)-1)
	if k>0:
		mlayers=estimate_best_layers(n,k)
		return mlayers
	else:
		return [1,n]

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

def model():
	_IMAGE_SIZE = 32
	_IMAGE_CHANNELS = 3
	_NUM_CLASSES = 10
	_RESHAPE_SIZE = 8*8*64

	with tf.name_scope('data'):
		x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
		y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
		x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')

	conv1_kernel = tf.Variable(tf.random_normal([5, 5, 3, 64]))
	conv1_conv = tf.nn.conv2d(x_image, conv1_kernel, [1, 1, 1, 1], padding='SAME')
	conv1_biases = tf.Variable(tf.random_normal([64]))
	conv1_pre_activation = tf.nn.bias_add(conv1_conv, conv1_biases)
	conv1 = tf.nn.relu(conv1_pre_activation, name='conv1')
	
	pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

	conv2_kernel = tf.Variable(tf.random_normal([5, 5, 64, 64]))
	conv2_conv = tf.nn.conv2d(norm1, conv2_kernel, [1, 1, 1, 1], padding='SAME')
	conv2_biases = tf.Variable(tf.random_normal([64]))
	conv2_pre_activation = tf.nn.bias_add(conv2_conv, conv2_biases)
	conv2 = tf.nn.relu(conv2_pre_activation, name='conv2')

	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
	pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
	
	local3_reshape = tf.reshape(pool2, [-1, _RESHAPE_SIZE])
	local3_dim = local3_reshape.get_shape()[1].value
	local3_weights = tf.Variable(tf.random_normal([local3_dim, 384]))
	local3_biases = tf.Variable(tf.random_normal([384]))
	local3 = tf.nn.relu(tf.matmul(local3_reshape, local3_weights) + local3_biases, name='local3')

	local4_weights = tf.Variable(tf.random_normal([384, 192]))
	local4_biases = tf.Variable(tf.random_normal([192]))
	local4 = tf.nn.relu(tf.matmul(local3, local4_weights) + local4_biases, name='local4')

	output_weights = tf.Variable(tf.random_normal([192, _NUM_CLASSES]))
	output_biases = tf.Variable(tf.random_normal([_NUM_CLASSES]))
	softmax_linear = tf.add(tf.matmul(local4, output_weights), output_biases, name='output')

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	y_pred_cls = tf.argmax(softmax_linear, axis=1)
	
	params=[conv1_kernel,conv1_biases,conv2_kernel,conv2_biases,local3_weights,local3_biases,local4_weights,local4_biases,output_weights,output_biases]
	
	return x, y, softmax_linear, global_step, y_pred_cls , params

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
		s3.Bucket(AWS_S3_bucket).upload_file('/tmp/samples', 'data/samples_cifar_'+str(n))
	
def initialize_weights():
	x, y, output, global_step, y_pred_cls , params= model()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	
	s32 = boto3.resource('s3')
	pw=[]
	for i in range(len(params)):
		pw.append(sess.run(params[i]))
	with open('/tmp/model', 'w') as f:
		pickle.dump(pw, f)
	s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar')

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
	s3func.s3_clear_bucket(AWS_S3_bucket,'data/modelcifar_')
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
			Handler='CNNtf.lambda_handler',
			MemorySize=mems[i],
			Publish=True,
			Role=AWS_lambda_role,
			Runtime='python2.7',
			Timeout=300,
			VpcConfig={
			},
		)
		data+='create '+funcname+' with memory '+str(mems[i])+'\n'
	time.sleep(10)
	if 'cratio' not in event.keys():
		event['cratio']=1000
	if 'pcratio' not in event.keys():
		event['pcratio']=[700,10,2*100*0.0001,2*100*0.0001/20]
	event['rangen']=[50,200]
	event['rangemem']=[256,1536]
	
	if 'mod' not in event.keys() or event['mod']==0:
		event['funcname']='nntffunc_1'
		event['state']=2
		event['pid']=0
		event['memory']=mems[1]
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	elif event['mod']==1:
		event['funcname']='nntffunc_1'
		event['state']=2
		event['pid']=0
		event['memory']=mems[1]
		event['fixedmlayers']=1
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	elif event['mod']==2:
		first=int((event['rangen'][1]-event['rangen'][0])*random.random()+event['rangen'][0])
		if first-event['rangen'][0]<30:
			second=first+30
		elif event['rangen'][1]-first<30:
			second=fisrt-30
		else:
			if random.random()>=0.5:
				second=first+30
			else:
				second=first-30
		event['state']=3
		event['process']=[[1,second,mems[2],'nntffunc_2'],[0,first,mems[1],'nntffunc_1']]
		event['regtimes']=[4,0,1]
		event['regmod']=0
		event['maxchange']=[50,256]
		event['changestep']=[10,64]
		event['diffrange']=2
		event['memory']=mems[0]
		event['funcname']='nntffunc_0'
		s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='flag/work_monitor')
		invoke_lambda(client,'nntffunc_0',event)
	
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_startup.tsp')

def start_structure(event):
	st=datetime.datetime.now()
	pid=event['pid']
	mlayers=event['mlayers']
	nowiter=event['nowiter']
	s3 = boto3.client('s3')
	s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='flag/work_'+str(pid))
	data=''
	print 'structure start:',pid
	if 'modelname' in event.keys():
		data+='modelname found: '+event['modelname']+'\n'
		flag=s3func.s3_download_file(0,AWS_S3_bucket,event['modelname'],'/tmp/model',0,1,0)
		if flag==0:
			print 'ERROR!!!',event['modelname'],'not found'
			s3.put_object(Bucket=AWS_S3_bucket,Body=event['modelname']+' not found', Key='error/error_start_'+str(pid))
	else:
		data+='modelname not found\n'
		s3func.s3_download_file(0,AWS_S3_bucket,'data/modelcifar','/tmp/model',0,1,0)
	s32 = boto3.resource('s3')
	s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar_'+str(pid)+'_'+str(nowiter))
	
	client=boto3.client('lambda',region_name = AWS_region)
	
	if ('fixedmlayers' not in event.keys()) or event['fixedmlayers']==0:
		mlayers=estimate_merging(mlayers[-1])
		if len(mlayers)>4:
			mlayers=estimate_best_layers(mlayers[-1],2)
		event['mlayers']=mlayers
		data+='mlayers is: '+str(mlayers)+'\n'
	
	search=range(len(mlayers)-1)
	search.reverse()
	mstart=[]
	for l in search:
		base=int(mlayers[l+1]/mlayers[l])
		remin=mlayers[l+1]%mlayers[l]
		temp=[]
		now=0
		for i in range(mlayers[l]):
			now+=base
			if remin>0:
				now+=1
				remin-=1
			temp.append(now-1)
		if l!=len(mlayers)-2:
			for i in range(len(temp)):
				temp[i]=mstart[-1][temp[i]]
		mstart.append(temp)
	mstart.reverse()
	for i in range(mlayers[-1]):
		event['pos']=i
		event['state']=1
		event['mergepos']=[]
		for l in range(len(mstart)):
			if i in mstart[l]:
				event['mergepos'].append([l,mstart[l].index(i)])
		event['mergepos'].reverse()
		data+='work node '+str(event['pos'])+', mergepos: '+str(event['mergepos'])+'\n'
		invoke_lambda(client,event['funcname'],event)
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_startstructure_'+str(pid)+'.tsp')

def train(event):
	st=datetime.datetime.now()
	stt=time.time()
	tcount=time.time()
	if 'roundtime' not in event.keys():
		event['roundtime']=250
	tend=event['roundtime']
	ns=event['ns']
	pos=event['pos']
	mlayers=event['mlayers']
	maxiter=event['maxiter']
	nowiter=event['nowiter']
	funcname=event['funcname']
	if 'batchnumber' not in event.keys():
		event['batchnumber']=1
	bn=event['batchnumber']
	pid=event['pid']
	if 'testtime' not in event.keys():
		event['testtime']=10
	if 'waittime' not in event.keys():
		event['waittime']=tend*2/3
	if 'learningrate' not in event.keys():
		event['learningrate']=0.1
	waittime=event['waittime']
	timer=s3func.timer([waittime,tend])
	if 'round' not in event.keys():
		event['round']=0
	else:
		event['round']+=1
	rounditer=event['rounditer']
	s3 = boto3.client('s3')
	s32 = boto3.resource('s3')
	client=boto3.client('lambda',region_name = AWS_region)
	response = client.get_function(
		FunctionName=funcname,
	)
	if nowiter==0 and pos==0:
		s3.put_object(Bucket=AWS_S3_bucket,Body=str(stt), Key='timestamp/timestamp_train_start_'+str(pid))
	response = client.get_function(
		FunctionName=funcname,
	)
	filerecord=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'results/result',0,1,0)
	if filerecord==0:
		filerecord=''
	filerecord+='====='+' merge: '+str(mlayers)+' samples: '+str(ns)+' memory: '+str(event['memory'])+' testtime left :'+str(event['testtime'])+' starttime: '+str(st)+'\n'
	filerecord+='====='+str(stt)+'\n'
	data='train round '+str(event['round'])+', round time '+str(event['roundtime'])+', start at '+str(st)+' ##'+str(time.time())+'\n'
	data+='info: pos '+str(pos)+', memory '+str(response['Configuration']['MemorySize'])+', mlayers '+str(mlayers)+', ns '+str(ns)+'\n'
	print '='*5,'train node',pos,'='*5,'train phase start'
	split=500
	base=int(ns/mlayers[-1])
	remin=ns%mlayers[-1]
	sn=0
	for n in range(pos):
		sn+=base
		if remin:
			sn+=1
			remin-=1
	en=sn+base
	if remin:
		en+=1
	print '='*5,'train node',pos,'='*5,'read samples from',sn,'to',en
	train_x=[]
	train_y=[]
	sfile=int(sn/split)
	efile=int((en-1)/split)
	print '='*5,'train node',pos,'='*5,'read files from',sfile,'to',efile
	data+='start up time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
	#=========================================read========================================
	stt=time.time()
	if os.path.exists('/tmp/samples_save'):
		print '='*5,'train node',pos,'='*5,'found samples!!!'
		with open('/tmp/samples_save', 'r') as f:
			temp=pickle.load(f)
		#os.remove('/tmp/samples_save_'+str(pos))
		train_x=temp['data']
		train_y=temp['label']
		data+='found samples!!! time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	else:
		print '='*5,'train node',pos,'='*5,'samples not found, downloading'
		for now in range(sfile,efile+1):
			print 'downloading',now,'from range',sfile,efile+1
			flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_cifar_'+str(now),'/tmp/samples',0,1,0)
			if flag==0:
				print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read sample file:',now
			with open('/tmp/samples', 'r') as f:
				temp=pickle.load(f)
			sread=max([split*now,sn])-split*now
			eread=min([split*(now+1),en])-split*now
			
			if train_x==[]:
				train_x=temp['data'][sread:eread]
				train_y=temp['label'][sread:eread]
			else:
				train_x=np.append(temp['data'],temp['data'][sread:eread],axis=0)
				train_y=np.append(temp['label'],temp['label'][sread:eread],axis=0)
			
		if os.path.exists('/tmp/samples'):
			os.remove('/tmp/samples')
		with open('/tmp/samples_save', 'w') as f:
			pickle.dump({'data':train_x,'label':train_y}, f)
		data+='read from '+str(sfile)+' to '+str(efile)+' time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	data+='samples length: '+str(len(train_x))+'\n'
	if nowiter==0 and pos==0:
		s3.put_object(Bucket=AWS_S3_bucket,Body=str(time.time()), Key='timestamp/timestamp_train_start_'+str(pid))
	#=========================================read========================================
	#=========================================initialize==================================
	stt=time.time()
	x, y, output, global_step, y_pred_cls , params= model()
	
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
	#optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss, global_step=global_step)
	
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	data+='training initialize time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	#=========================================initialize==================================
	#=========================================LOOP start========================================
	avgitertime=0.0
	avgitertimereal=0.0
	minitertimereal=100000.0
	timerecord=[]
	smt=0.0
	while nowiter<maxiter:
		itertime=[0.0]
		stiter=time.time()
		flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_'+str(pid),'/tmp/work',0,1,0)
		if flag==0:
			print '='*5,'train node',pos,'='*5,'Abandon!!!! pid:',pid
			return
		stt=time.time()
		print '+'*5,'train node',pos,'pid',pid,'+'*5,'now start iteration',nowiter
		print '='*5,'train node',pos,'='*5,'now start iteration',nowiter
		stt2=time.time()
		flag=s3func.s3_download_file_timer(s3,AWS_S3_bucket,'data/modelcifar_'+str(pid)+'_'+str(nowiter),'/tmp/model',timer,0,0)
		itertime[0]+=time.time()-stt2
		data+='training '+str(nowiter)+' model waiting time: '+str(time.time()-stt2)+' ##'+str(stt2)+'--'+str(time.time())+'\n'
		if flag==0:
			if timer.query()[1]>waittime/4:
				print '++++++++lambda train',pos,'at iteration',nowiter,'end at',datetime.datetime.now()
				s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
				event['nowiter']=nowiter
				return
			else:
				print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read model',nowiter
				s3.put_object(Bucket=AWS_S3_bucket,Body='fail to read model '+str(nowiter), Key='error/error_train_'+str(pid)+'_'+str(pos))
				return
		if nowiter>=(event['round']+1)*rounditer:
			if [0,0] in event['mergepos']:
				s3.put_object(Bucket=AWS_S3_bucket,Body=filerecord, Key='results/result')
			print '++++++++lambda train',pos,'at iteration',nowiter,'end at',datetime.datetime.now()
			s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
			event['nowiter']=nowiter
			invoke_lambda(client,funcname,event)
			return
		stt2=time.time()
		with open('/tmp/model', 'r') as f:
			temp=pickle.load(f)
		if temp[0]==[]:
			print '='*5,'train node',pos,'='*5,'ERROR!!!: model format wrong',nowiter
			s3.put_object(Bucket=AWS_S3_bucket,Body='model format wrong '+str(nowiter), Key='error/error_train_'+str(pid)+'_'+str(pos))
			return
		for i in range(len(temp)):
			sess.run(tf.assign(params[i],temp[i]))
		itertime[0]+=time.time()-stt2
		data+='training '+str(nowiter)+' download model time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
		stt=time.time()
		bs=len(train_x)/bn
		print '='*5,'train node',pos,'='*5,'train start'
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=event['learningrate']).minimize(loss, global_step=global_step)
		for b in range(bn):
			i_global, _ = sess.run([global_step, optimizer], feed_dict={x: train_x[b*bs:(b+1)*bs,:], y: train_y[b*bs:(b+1)*bs,:]})#=========================train
		event['learningrate']=event['learningrate']
		itertime[0]+=time.time()-stt
		data+='training '+str(nowiter)+' train time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
		stt=time.time()
		pw=[]
		for i in range(len(params)):
			pw.append(sess.run(params[i]))
		with open('/tmp/model', 'w') as f:
			pickle.dump(pw, f)
		print '='*5,'train node',pos,'='*5,'write result as layer',len(mlayers)-1,'node',pos
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar_'+str(pid)+'_'+str(len(mlayers)-1)+'_'+str(pos))
		itertime[0]+=time.time()-stt
		data+='training '+str(nowiter)+' model write time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
		if len(event['mergepos'])>0:
			mergepos=copy.deepcopy(event['mergepos'])
			thismergepos=mergepos[0]
			del mergepos[0]
			#tempd=merge(mlayers,thismergepos,mergepos,nowiter,timer,max(waittime,tend-time.time()+tcount),itertime,pid)
			smt=time.time()
			tempd=merge(mlayers,thismergepos,mergepos,nowiter,timer,waittime,itertime,pid)
			smt=time.time()-smt
			if tempd==0:
				return
			elif tempd==1:
				print '++++++++lambda train',pos,'at iteration',nowiter,'end at',datetime.datetime.now()
				s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
				event['nowiter']=nowiter
				return
			else:
				data+=tempd
		data+='training '+str(nowiter)+' valid iteration time: '+str(itertime[0])+'\n'
		print '-'*5,'train node',pos,'-'*5,'now end iteration',nowiter
		avgitertime+=itertime[0]
		"""
		if nowiter>=min(10,maxiter-1) and [0,0] in event['mergepos']:
			s3.put_object(Bucket=AWS_S3_bucket,Body=str(avgitertime/(nowiter+1)), Key='timestamp/timestamp_iteration_'+str(pid)+'.tsp')
		"""
		thisitertime=time.time()-stiter
		filerecord+=str(time.time())+'\n'
		#filerecord+=str(time.time()-stiter)+'\n'
		if thisitertime<minitertimereal:
			minitertimereal=thisitertime
		if nowiter>=2:
			avgitertimereal+=time.time()-stiter
			insert_sort(time.time()-stiter,timerecord)
			#filerecord+=str(time.time()-stiter)+'\n'
			#filerecord+=str(smt)+'\n'
		if nowiter>=min(10,maxiter-1) and [0,0] in event['mergepos']:
			aaaa=0
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(timerecord), Key='timestamp/timestamp_iteration_each_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(avg_no_abnormal(timerecord,2)), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(find_median(timerecord)), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(avgitertimereal/(nowiter-2+1)), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(minitertimereal), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
		nowiter+=1
	if [0,0] in event['mergepos']:
		s3.put_object(Bucket=AWS_S3_bucket,Body=filerecord, Key='results/result')
		"""
		event['testtime']-=1
		if event['testtime']>0:
			inputs={'state':0,'mod':0,'slayers':[],'mlayers':event['mlayers'],'layers':[100,100,100,100,100],'pos':[0,0],'ns':1000000,'maxiter':10,'nowiter':0,'roundtime':250}
			inputs['testtime']=event['testtime']
			invoke_lambda(client,'testfunc',inputs)
			time.sleep(10)
		"""
	if [0,0] in event['mergepos']:
		s3.put_object(Bucket=AWS_S3_bucket,Body=str(time.time()), Key='timestamp/timestamp_train_end_'+str(pid))
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
		
def merge(mlayers,pos,mergepos,nowiter,timer,waittime,itertime,pid):
	stt=time.time()
	tcount=time.time()
	layer=pos[0]
	node=pos[1]
	data=''
	
	base=int(mlayers[layer+1]/mlayers[layer])
	remin=mlayers[layer+1]%mlayers[layer]
	print '='*5,'merge node at layer',layer,'node',node,'='*5,'merge phase start'
	sn=0
	for n in range(node):
		sn+=base
		if remin:
			sn+=1
			remin-=1
	en=sn+base
	if remin:
		en+=1
	print '='*5,'merge node at layer',layer,'node',node,'='*5,'merge model file at layer',layer+1,'from',sn,'to',en
	s3 = boto3.client('s3')
	s32 = boto3.resource('s3')
	itertime[0]+=time.time()-stt
	data+='merge '+str(nowiter)+' layer '+str(layer)+' node '+str(node)+' merge '+str([sn,en-1])+' start up time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	
	#============================================start==============================================
	stt=time.time()
	print '='*5,'merge node at layer',layer,'node',node,'='*5,'iteration',nowiter
	params=[]
	flagt=0.0
	modelt=0.0
	itertime[0]+=1
	#files=['data/modelcifar_'+str(pid)+'_'+str(layer+1)+'_'+str(now) for now in range(sn,en)]
	
	finished=[0 for i in range(en-sn)]
	timer.local_start(0)
	while 1:
		if sum(finished)==(en-sn):
			break
		for now in range(sn,en):
			if finished[now-sn]==0:
				tresult=timer.query()
				if tresult[0]==1:
					if tresult[1]>waittime/4:
						return 0
					else:
						print '='*5,'merge node at layer',layer,'node',node,'='*5,'ERROR!!!: fail to read model: layer',layer+1,'finished state',str(finished)
						s3.put_object(Bucket=AWS_S3_bucket,Body='fail to read model at iteration '+str(nowiter)+' at layer '+str(layer+1)+', finished state'+str(finished), Key='error/error_merge_'+str(pid)+'_'+str(layer)+'_'+str(node))
						return 0
				flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/modelcifar_'+str(pid)+'_'+str(layer+1)+'_'+str(now),'/tmp/model',0,1,1)
				if flag==1:
					finished[now-sn]=1
					with open('/tmp/model', 'r') as f:
						temp=pickle.load(f)
					if not temp[0]==[]:
						if params==[]:
							params=temp
						else:
							for i in range(len(temp)):
								params[i]=params[i]+temp[i]
	data+='merge '+str(nowiter)+' layer '+str(layer)+' node '+str(node)+' model read time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	stt=time.time()
	if layer==0:
		print '='*5,'merge node at layer',layer,'node',node,'='*5,'now is the final node'
		if not params==[]:
			for i in range(len(params)):
				params[i]=params[i]/mlayers[-1]
		with open('/tmp/model', 'w') as f:
			pickle.dump(params, f)
		s32 = boto3.resource('s3')
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar_'+str(pid)+'_new')
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar_'+str(pid)+'_'+str(nowiter+1))
	else:
		with open('/tmp/model', 'w') as f:
			pickle.dump(params, f)
		print '='*5,'merge node at layer',layer,'node',node,'='*5,'write model as layer',layer,'node',node
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/modelcifar_'+str(pid)+'_'+str(layer)+'_'+str(node))
	itertime[0]+=time.time()-stt
	data+='merge '+str(nowiter)+' layer '+str(layer)+' node '+str(node)+' model write time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	if len(mergepos)>0:
		thismergepos=mergepos[0]
		del mergepos[0]
		tempd=merge(mlayers,thismergepos,mergepos,nowiter,timer,waittime,itertime,pid)
		if type(tempd)==int:
			return tempd
		else:
			data+=tempd
	
	return data

def lambda_handler(event,context):
	state=event['state']
	if state==0:#invoke
		startup(event)
	elif state==1:#work
		train(event)
	elif state==2:#start a structure
		start_structure(event)

#initialize_samples()
#initialize_weights()
