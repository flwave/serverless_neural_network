"""
This file is for Case A
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

AWS_region='us-west-2'
#AWS_region='us-east-1'
AWS_S3_bucket='lf-src'
AWS_lambda_role='arn:aws:iam::xxx:role/lambdarole'

OPTIMIZATION=0#0: optimize cost, 1: optimize performance/cost

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

def extract_valid(seq):
	cut=-1
	search=[i+1 for i in range(len(seq)-1)]
	search.reverse()
	for i in search:
		if abs(np.mean(seq[:i])-np.median(seq[:i]))>abs(np.mean(seq[:i+1])-np.median(seq[:i+1])):
			cut=i
			break
	if cut>-1:
		seq=seq[:cut]
	return seq

def find_median(lis):
	lis.sort()
	return lis[len(lis)/2]

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

def initialize_samples(layers,num):
	[samples,ds]=gen_random_sample(layers,num)
	samples=np.array(samples)
	ds=np.array(ds)
	ds[0:len(ds)/2,:]=-1.0
	ds[len(ds)/2:,:]=1.0
	samples=np.append(samples,ds,axis=1)
	np.random.shuffle(samples)
	ds=samples[:,-layers[-1]:]
	samples=samples[:,0:-layers[-1]]
	split=10000
	base=int(num/split)
	remin=num%split
	s3 = boto3.resource('s3')
	now=0
	print str(base)+' parts and reminding '+str(remin)
	for n in range(base):
		print n,'from '+str(n*split)+' to '+str((n+1)*split)
		with open('/tmp/samples', 'w') as f:
			pickle.dump([samples[n*split:(n+1)*split],ds[n*split:(n+1)*split]], f)
		s3.Bucket(AWS_S3_bucket).upload_file('/tmp/samples', 'data/samples_'+str(now))
		now+=1
	if remin:
		print now,'ex from '+str(now*split)+' to end'
		with open('/tmp/samples', 'w') as f:
			pickle.dump(np.array([samples[now*split:],ds[now*split:]]), f)
		s3.Bucket(AWS_S3_bucket).upload_file('/tmp/samples', 'data/samples_'+str(now))
	
	"""
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	rew=session.run(weights)
	reb=session.run(biases)
	model=[rew,reb]
	with open('/tmp/model', 'w') as f:
		pickle.dump(model, f)
	s32 = boto3.resource('s3')
	s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model')
	"""
	
def initialize_weights(layers):
	
	weights=[[] for i in range(len(layers))]
	biases=[[] for i in range(len(layers))]
	for l in range(len(layers)):
		if l>0:
			weights[l]=tf.Variable(tf.random_normal([layers[l-1], layers[l]]))
			biases[l]=tf.Variable(tf.random_normal([1, layers[l]]))
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	rew=session.run(weights)
	reb=session.run(biases)
	model=[rew,reb]
	with open('/tmp/model', 'w') as f:
		pickle.dump(model, f)
	s32 = boto3.resource('s3')
	s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model')

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
	"""
	response = client.get_function(
		FunctionName='testfunc',
	)
	event['mem']=float(response['Configuration']['MemorySize'])
	"""
	funclist = client.list_functions(
		MaxItems=5
	)
	event['rangen']=[50,200]
	event['rangemem']=[256,1536]
	rstep=64
	first=rstep*random.randint(event['rangemem'][0]/rstep,event['rangemem'][1]/rstep-1)
	second=first+rstep
	"""
	if first-event['rangemem'][0]<rstep:
		second=first+rstep
	elif event['rangemem'][1]-first<rstep:
		second=fisrt-rstep
	else:
		if random.random()>=0.5:
			second=first+rstep
		else:
			second=first-rstep
	"""
	if event['mod']==2:
		mems=[512,first,second]
	else:
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
			Handler='NNtf5.lambda_handler',
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
	if 'cratio' not in event.keys():
		event['cratio']=1000#cost ratio
	if 'pcratio' not in event.keys():
		#event['pcratio']=[700,10,1e9,1e9/20]
		event['pcratio']=[700,10,250,250/20]#700,10 is for node(NA), 250, 250/10 is for memory
	
	if 'mod' not in event.keys() or event['mod']==0:#run training, the merge layer structure is calculated
		event['funcname']='nntffunc_1'
		event['state']=2
		event['pid']=0
		event['memory']=mems[1]
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	elif event['mod']==1:#run training, but the merge layer structure is specified by user
		event['funcname']='nntffunc_1'
		event['state']=2
		event['pid']=0
		event['memory']=mems[1]
		event['fixedmlayers']=1
		cevent=copy.deepcopy(event)
		invoke_lambda(client,'nntffunc_1',cevent)
	elif event['mod']==2:#run cost (preformance/cost) optimization
		first=int((event['rangen'][1]-event['rangen'][0])*random.random()+event['rangen'][0])
		if first-event['rangen'][0]<30:
			second=first+30
		elif event['rangen'][1]-first<30:
			second=first-30
		else:
			if random.random()>=0.5:
				second=first+30
			else:
				second=first-30
		event['state']=3#call monitor
		event['process']=[[1,100,mems[2],'nntffunc_2'],[0,100,mems[1],'nntffunc_1']]#FIFO
		event['regtimes']=[0,5,1]
		event['regmod']=1
		event['maxchange']=[50,512]#50 is the number of the nodes(NA), 512 is the max memory change for one time
		event['changestep']=[10,64]#10 is for node(NA), 64 is the memory, which means we have to change 64n memory for one time
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
		s3func.s3_download_file(0,AWS_S3_bucket,'data/model','/tmp/model',0,1,0)
	s32 = boto3.resource('s3')
	s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model_'+str(pid)+'_0')
	
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

def monitor(event):
	s3 = boto3.client('s3')
	client=boto3.client('lambda',region_name = AWS_region)
	data=''
	print 'monitor begin'
	#====================================================load info===================================================================
	if 'process' not in event.keys():
		print 'ERROR!!!!! process key not found'
		s3.put_object(Bucket=AWS_S3_bucket,Body='process key not found', Key='error/error_monitor.tsp')
		return
	if 'round' not in event.keys():
		event['round']=0
	process=event['process']
	tend=250
	cratio=event['cratio']# cost ratio
	pcratio=event['pcratio']# performance/cost ratio
	rangen=event['rangen']
	rangemem=event['rangemem']
	maxchange=event['maxchange']
	changestep=event['changestep']
	lastpid=[]
	data+='basic info: '+'pcratio: '+str(pcratio)+'rangen: '+str(rangen)+'maxchange: '+str(maxchange)+'changestep: '+str(changestep)+'\n'
	if 'avgrecord' not in event.keys():
		event['avgrecord']=[[0.0,0] for i in range(20)]
	if 'lastslop' not in event.keys():
		event['lastslop']=[]
	if 'regmod' not in event.keys():
		event['regmod']=0
	if 'regcount' not in event.keys():
		event['regcount']=[0,0,0]
	diffrange=event['diffrange']
	tcount=time.time()
	data+='UTC:'+str(datetime.datetime.now())+'. '+str(tcount)+'\n'
	s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
	if event['round']>0:
		temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_monitor_'+str(event['round']-1)+'.tsp',0,1,0)
		if temp!=0:
			data+=temp
	data+='=============================='
	best=process[0]
	#====================================================load info===================================================================
	while time.time()-tcount<tend:
		#load the control file "work_monitor". If it is deleted manually, monitor stops.
		flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_monitor','/tmp/work',0,1,0)
		if flag==0:
			print 'monitor terminated!!!!!!'
			return
		if event['regmod']==0:
			if event['regcount'][0]>=event['regtimes'][0]:
				event['regcount'][0]=0
				event['regmod']=1
		if event['regmod']==1:
			if event['regcount'][1]>=event['regtimes'][1]:
				event['regcount'][1]=0
				event['regmod']=0
				event['regcount'][2]+=1
		if event['regcount'][2]>=event['regtimes'][2]:
			print 'final result:',best
			data+='final result: '+str(best)+'\n'
			print 'achieve the maximum regression constraint, terminated!!!!!'
			data+='achieve the maximum regression constraint, terminated!!!!!\n'
			s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
			return
		times=[-1,-1]
		print 'regression times:',event['regcount']
		data+='regression times: '+str(event['regcount'])+'\n'
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
		temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_startstructure_'+str(process[0][0])+'.tsp',0,1,0)
		if temp==0:
			cevent=copy.deepcopy(event)
			cevent['pid']=process[0][0]
			cevent['mlayers']=[1,process[0][1]]
			cevent['state']=2
			cevent['funcname']=process[0][3]
			cevent['memory']=process[0][2]
			cevent['round']=-1
			if lastpid!=[]:
				cevent['modelname']='data/model_'+str(lastpid)+'_new'
			else:
				cevent['modelname']='data/model'
			print 'invoke',process[0][3]
			invoke_lambda(client,process[0][3],cevent)
			temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_startstructure_'+str(process[0][0])+'.tsp',tend-time.time()+tcount,0,0)
		data+='waiting the first iteration data:'+'timestamp/timestamp_iteration_real_'+str(process[0][0])+'.tsp'+'\n'
		s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
		temp=0
		while tend-time.time()+tcount>0 and temp==0:
			flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_monitor','/tmp/work',0,1,0)
			if flag==0:
				print 'monitor terminated!!!!!!'
				return
			temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_iteration_each_'+str(process[0][0])+'.tsp',0,1,0)
			time.sleep(5)
		if temp==0:
			print 'ERROR!!!!! connot read timestamp',process[0]
			s3.put_object(Bucket=AWS_S3_bucket,Body='connot read timestamp: '+str(process[0]), Key='error/error_monitor.tsp')
			break
			#return
		times[0]=json.loads(temp)
		s3func.s3_delete_file(s3,AWS_S3_bucket,'flag/work_'+str(process[0][0]))
		s3func.s3_clear_bucket(AWS_S3_bucket,'timestamp/timestamp_train_'+str(process[0][0]))
		lastpid=process[0][0]
		data+='the time cost is: '+str(times[0])+'\n'
		#--------------------------------------------------------
		temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_startstructure_'+str(process[1][0])+'.tsp',0,1,0)
		if temp==0:
			cevent=copy.deepcopy(event)
			cevent['pid']=process[1][0]
			cevent['mlayers']=[1,process[1][1]]
			cevent['state']=2
			cevent['funcname']=process[1][3]
			cevent['memory']=process[1][2]
			cevent['round']=-1
			if lastpid!=[]:
				cevent['modelname']='data/model_'+str(lastpid)+'_new'
			else:
				cevent['modelname']='data/model'
			print 'invoke',process[1][3]
			invoke_lambda(client,process[1][3],cevent)
			temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_startstructure_'+str(process[1][0])+'.tsp',tend-time.time()+tcount,0,0)
		data+='waiting the second iteration data:'+'timestamp/timestamp_iteration_real_'+str(process[1][0])+'.tsp'+'\n'
		s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
		temp=0
		while tend-time.time()+tcount>0 and temp==0:
			flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_monitor','/tmp/work',0,1,0)
			if flag==0:
				print 'monitor terminated!!!!!!'
				return
			temp=s3func.s3_read_file_v2(s3,AWS_S3_bucket,'timestamp/timestamp_iteration_each_'+str(process[1][0])+'.tsp',0,1,0)
			time.sleep(5)
		if temp==0:
			print 'ERROR!!!!! connot read timesteamp',process[1]
			s3.put_object(Bucket=AWS_S3_bucket,Body='connot read timestamp: '+str(process[1]), Key='error/error_monitor.tsp')
			break
			#return
		times[1]=json.loads(temp)
		s3func.s3_delete_file(s3,AWS_S3_bucket,'flag/work_'+str(process[1][0]))
		s3func.s3_clear_bucket(AWS_S3_bucket,'timestamp/timestamp_train_'+str(process[1][0]))
		lastpid=process[1][0]
		data+='the time cost is: '+str(times[1])+'\n'
		#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
		if event['regmod']==0:
			times[0]=float(np.median(times[0]))
			times[1]=float(np.median(times[1]))
			if times[1]<times[0]:
				process.reverse()
				times.reverse()
			best=process[0]
			nextfunc=process[1][3]
			F=times[0]-times[1]
			data+=str(process[0])+': '+str(times[0])+'. '+str(process[1])+': '+str(times[1])+'. \n'
			data+='delta target value is '+str(F)+'\n'
			if process[0][1]==process[1][1]:
				nextn=process[0][1]
			else:
				if event['lastslop']==[]:
					change=-pcratio[0]*F/(process[0][1]-process[1][1])
					event['lastslop']=F/(process[0][1]-process[1][1])
				else:
					change=-pcratio[0]*F/(process[0][1]-process[1][1])+pcratio[1]*(F/(process[0][1]-process[1][1])-event['lastslop'])
					event['lastslop']=F/(process[0][1]-process[1][1])
				change=ceil_step(change,changestep[0])
				if change>maxchange[0]:
					change=maxchange[0]
				elif change<-maxchange[0]:
					change=-maxchange[0]
				nextn=int(process[0][1]+change)
			nextmem=process[0][2]
		else:
			metric=[0.0,0.0]
			if OPTIMIZATION==0:
				metric[0]=float(np.median([(process[0][2]*ts) for ts in times[0]]))
				metric[1]=float(np.median([(process[1][2]*ts) for ts in times[1]]))
			else:
				metric[0]=float(np.median([1.0/(process[0][2]*ts**2) for ts in times[0]]))
				metric[1]=float(np.median([1.0/(process[1][2]*ts**2) for ts in times[1]]))
			if metric[1]<metric[0]:
			#if process[1][2]*times[1]<process[0][2]*times[0]:
				process.reverse()
				times.reverse()
				metric.reverse()
			best=process[0]
			nextfunc=process[1][3]
			F=metric[0]-metric[1]
			data+=str(process[0])+': '+str(metric[0])+'. '+str(process[1])+': '+str(metric[1])+'. \n'
			data+='delta target value is '+str(F)+'\n'
			nextn=process[0][1]
			if process[0][2]==process[1][2]:
				nextmem=process[0][2]
			else:
				#change=-pcratio[2]*F/(process[0][2]-process[1][2])
				if event['lastslop']==[]:
					change=-pcratio[2]*F/(process[0][2]-process[1][2])
					event['lastslop']=F/(process[0][2]-process[1][2])
				else:
					change=-pcratio[2]*F/(process[0][2]-process[1][2])+pcratio[3]*(F/(process[0][2]-process[1][2])-event['lastslop'])
					event['lastslop']=F/(process[0][2]-process[1][2])
				#change=-change#for 1/mt2
				change=ceil_step(change,changestep[1])
				if change>maxchange[1]:
					change=maxchange[1]
				elif change<-maxchange[1]:
					change=-maxchange[1]
				nextmem=int(process[0][2]+change)
		nextpid=max(process[0][0],process[1][0])+1
		if nextn>rangen[1]:
			nextn=rangen[1]
		elif nextn<rangen[0]:
			nextn=rangen[0]
		if nextmem>rangemem[1]:
			nextmem=rangemem[1]
		elif nextmem<rangemem[0]:
			nextmem=rangemem[0]
		#s3func.s3_clear_bucket(AWS_S3_bucket,'data/model_'+str(process[1][0]))
		process[1]=process[0]
		process[0]=[nextpid,nextn,nextmem,nextfunc]
		#process[1][0]=max(process[0][0],process[1][0])+1
		event['process']=process
		response = client.update_function_configuration(
			FunctionName=nextfunc,
			MemorySize=nextmem,
		)
		time.sleep(5)
		print '---------',process
		if abs(process[0][1]-process[1][1])<1 and process[0][2]==process[1][2]:
			print 'terminated!!!'
			print 'final result:',best
			data+='final result: '+str(best)+'\n'
			#data+='the final result is '+str(process[0])
			s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
			return
		data+= '---------'+str(process)+'\n'
		if event['regmod']==0:
			event['regcount'][0]+=1
		else:
			event['regcount'][1]+=1
		s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_monitor_'+str(event['round'])+'.tsp')
	event['round']+=1
	invoke_lambda(client,event['funcname'],event)
	
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
	layers=event['layers']
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
	client=boto3.client('lambda',region_name = AWS_region)
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
	data+='info: pos '+str(pos)+', memory '+str(response['Configuration']['MemorySize'])+', mlayers '+str(mlayers)+', ns '+str(ns)+', layers '+str(layers)+'\n'
	print '='*5,'train node',pos,'='*5,'train phase start'
	split=10000
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
	training_inputs=[]
	training_outputs=[]
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
		training_inputs=temp[0]
		training_outputs=temp[1]
		data+='found samples!!! time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	else:
		print '='*5,'train node',pos,'='*5,'samples not found, downloading'
		for now in range(sfile,efile+1):
			print 'downloading',now,'from range',sfile,efile+1
			#flag=s3func.s3_download_file_v2(s3,s32,AWS_S3_bucket,'data/samples_'+str(now),'/tmp/samples',0,1,0)
			flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_'+str(now),'/tmp/samples',0,1,0)
			if flag==0:
				print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read sample file:',now
			with open('/tmp/samples', 'r') as f:
				temp=pickle.load(f)
			sread=max([split*now,sn])-split*now
			eread=min([split*(now+1),en])-split*now
			
			if training_inputs==[]:
				training_inputs=temp[0][sread:eread]
				training_outputs=temp[1][sread:eread]
			else:
				training_inputs=np.append(training_inputs,temp[0][sread:eread],axis=0)
				training_outputs=np.append(training_outputs,temp[1][sread:eread],axis=0)
			
			#training_inputs.extend(temp[0][sread:eread])
			#training_outputs.extend(temp[1][sread:eread])
		if os.path.exists('/tmp/samples'):
			os.remove('/tmp/samples')
		with open('/tmp/samples_save', 'w') as f:
			pickle.dump([training_inputs,training_outputs], f)
		data+='read from '+str(sfile)+' to '+str(efile)+' time: '+str(time.time()-stt)+' ##'+str(time.time())+' ##'+str(stt)+'--'+str(time.time())+'\n'
	data+='samples length: '+str(len(training_inputs))+'\n'
	if nowiter==0 and pos==0:
		s3.put_object(Bucket=AWS_S3_bucket,Body=str(time.time()), Key='timestamp/timestamp_train_start_2_'+str(pid))
	#=========================================read========================================
	#=========================================initialize==================================
	stt=time.time()
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

	#train=tf.train.AdamOptimizer(0.1).minimize(cost)
	train=tf.train.GradientDescentOptimizer(1e-1).minimize(cost)
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
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
		#flag=s3func.s3_download_file_v2(s3,s32,AWS_S3_bucket,'flag/work_'+str(pid),'/tmp/work',0,1,0)
		flag=s3func.s3_download_file(s3,AWS_S3_bucket,'flag/work_'+str(pid),'/tmp/work',0,1,0)
		if flag==0:
			print '='*5,'train node',pos,'='*5,'Abandon!!!! pid:',pid
			return
		stt=time.time()
		print '+'*5,'train node',pos,'pid',pid,'+'*5,'now start iteration',nowiter
		print '='*5,'train node',pos,'='*5,'now start iteration',nowiter
		stt2=time.time()
		#flag=s3func.s3_download_file_v2(s3,s32,AWS_S3_bucket,'data/model_'+str(pid)+'_'+str(nowiter),'/tmp/model',max(waittime,tend-time.time()+tcount),0,0)
		#flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/model_'+str(pid)+'_'+str(nowiter),'/tmp/model',max(waittime,tend-time.time()+tcount),0,0)
		flag=s3func.s3_download_file_timer(s3,AWS_S3_bucket,'data/model_'+str(pid)+'_'+str(nowiter),'/tmp/model',timer,0,0)
		itertime[0]+=time.time()-stt2
		data+='training '+str(nowiter)+' model waiting time: '+str(time.time()-stt2)+' ##'+str(stt2)+'--'+str(time.time())+'\n'
		#print 'flag',flag
		if flag==0:
			if timer.query()[1]>waittime/4:
				print '++++++++lambda train',pos,'at iteration',nowiter,'end at',datetime.datetime.now()
				"""
				with open('/tmp/samples_save_'+str(pos), 'w') as f:
					pickle.dump([training_inputs,training_outputs], f)
				"""
				s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
				event['nowiter']=nowiter
				#invoke_lambda(client,funcname,event)
				return
			else:
				print '='*5,'train node',pos,'='*5,'ERROR!!!: fail to read model',nowiter
				s3.put_object(Bucket=AWS_S3_bucket,Body='fail to read model '+str(nowiter), Key='error/error_train_'+str(pid)+'_'+str(pos))
				return
		if nowiter>=(event['round']+1)*rounditer:
			print '++++++++lambda train',pos,'at iteration',nowiter,'end at',datetime.datetime.now()
			if [0,0] in event['mergepos']:
				s3.put_object(Bucket=AWS_S3_bucket,Body=filerecord, Key='results/result')
			"""
			with open('/tmp/samples_save_'+str(pos), 'w') as f:
				pickle.dump([training_inputs,training_outputs], f)
			"""
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
		for l in range(len(layers)):
			if l>0:
				session.run(tf.assign(weights[l],temp[0][l]))
				session.run(tf.assign(biases[l],temp[1][l]))
		itertime[0]+=time.time()-stt2
		data+='training '+str(nowiter)+' download model time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
		stt=time.time()
		print '='*5,'train node',pos,'='*5,'train start'
		bs=len(training_inputs)/bn
		for b in range(bn):
			session.run(train,feed_dict={x:training_inputs[b*bs:(b+1)*bs,:],y:training_outputs[b*bs:(b+1)*bs,:]})#=========================train
		itertime[0]+=time.time()-stt
		data+='training '+str(nowiter)+' train time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
		stt=time.time()
		rew=session.run(weights)
		reb=session.run(biases)
		model=[rew,reb]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
		print '='*5,'train node',pos,'='*5,'write result as layer',len(mlayers)-1,'node',pos
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model_'+str(pid)+'_'+str(len(mlayers)-1)+'_'+str(pos))
		#s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='flag/flag_'+str(pid)+'_'+str(nowiter)+'_'+str(len(mlayers)-1)+'_'+str(pos))
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
				"""
				with open('/tmp/samples_save_'+str(pos), 'w') as f:
					pickle.dump([training_inputs,training_outputs], f)
				"""
				s3.put_object(Bucket=AWS_S3_bucket,Body=data, Key='timestamp/timestamp_train_'+str(pid)+'_'+str(pos)+'_'+str(event['round'])+'.tsp')
				event['nowiter']=nowiter
				#invoke_lambda(client,funcname,event)
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
		#insert_sort(time.time()-stiter,timerecord)
		if nowiter>=2:
			avgitertimereal+=time.time()-stiter
			insert_sort(time.time()-stiter,timerecord)
			#filerecord+=str(smt)+'\n'
		if nowiter>=min(10,maxiter-1) and [0,0] in event['mergepos']:
			s3.put_object(Bucket=AWS_S3_bucket,Body=str(timerecord), Key='timestamp/timestamp_iteration_each_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(avg_no_abnormal(timerecord,2)), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(float(np.mean(timerecord))), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			s3.put_object(Bucket=AWS_S3_bucket,Body=str(float(np.mean(extract_valid(timerecord)))), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(avgitertimereal/(nowiter-2+1)), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
			#s3.put_object(Bucket=AWS_S3_bucket,Body=str(minitertimereal), Key='timestamp/timestamp_iteration_real_'+str(pid)+'.tsp')
		nowiter+=1
	if [0,0] in event['mergepos']:
		s3.put_object(Bucket=AWS_S3_bucket,Body=filerecord, Key='results/result')
		"""
		event['testtime']-=1
		if event['testtime']>0:
			inputs={'state':0,'mod':0,'batchnumber':20,'slayers':[],'mlayers':[1,100],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':1000000,'maxiter':10,'nowiter':0,'roundtime':250,'rounditer':15}
			inputs['testtime']=event['testtime']
			invoke_lambda(client,'testfunc',inputs)
			time.sleep(10)
		"""
	if [0,0] in event['mergepos'] and nowiter>=maxiter:
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
	data+='merge '+str(nowiter)+' layer '+str(layer)+' node '+str(node)+'merge '+str([sn,en-1])+' start up time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	
	#============================================start==============================================
	stt=time.time()
	print '='*5,'merge node at layer',layer,'node',node,'='*5,'iteration',nowiter
	weights=[]
	biases=[]
	flagt=0.0
	modelt=0.0
	itertime[0]+=1
	files=['data/model_'+str(pid)+'_'+str(layer+1)+'_'+str(now) for now in range(sn,en)]
	#flag=s3func.s3_check_multi_exist(s3,AWS_S3_bucket,'data/model_',files,waittime,0)
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
						print '='*5,'merge node at layer',layer,'node',node,'='*5,'ERROR!!!: fail to read model: layer',layer+1
						s3.put_object(Bucket=AWS_S3_bucket,Body='fail to read model at iteration '+str(nowiter)+' at layer '+str(layer+1), Key='error/error_merge_'+str(pid)+'_'+str(layer)+'_'+str(node))
						return 0
				flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/model_'+str(pid)+'_'+str(layer+1)+'_'+str(now),'/tmp/model',0,1,1)
				if flag==1:
					finished[now-sn]=1
					with open('/tmp/model', 'r') as f:
						temp=pickle.load(f)
					if not temp[0]==[]:
						if weights==[]:
							weights=temp[0]
							biases=temp[1]
						else:
							add_weights(weights,temp[0])
							add_weights(biases,temp[1])
	data+='merge '+str(nowiter)+' layer '+str(layer)+' node '+str(node)+' model read time: '+str(time.time()-stt)+' ##'+str(stt)+'--'+str(time.time())+'\n'
	stt=time.time()
	if layer==0:
		print '='*5,'merge node at layer',layer,'node',node,'='*5,'now is the final node'
		if not weights==[]:
			div_weights(weights,mlayers[-1])
			div_weights(biases,mlayers[-1])
		print len(weights),len(biases)
		model=[weights,biases]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
		s32 = boto3.resource('s3')
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model_'+str(pid)+'_new')
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model_'+str(pid)+'_'+str(nowiter+1))
	else:
		model=[weights,biases]
		with open('/tmp/model', 'w') as f:
			pickle.dump(model, f)
		print '='*5,'merge node at layer',layer,'node',node,'='*5,'write model as layer',layer,'node',node
		s32.Bucket(AWS_S3_bucket).upload_file('/tmp/model', 'data/model_'+str(pid)+'_'+str(layer)+'_'+str(node))
		#s3.put_object(Bucket=AWS_S3_bucket,Body='true', Key='flag/flag_'+str(pid)+'_'+str(nowiter)+'_'+str(layer)+'_'+str(node))
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
	elif state==3:#monitor the state
		monitor(event)
	
def test_result(layers,num):
	st=time.time()
	s32 = boto3.resource('s3')
	s3 = boto3.client('s3')
	training_inputs=[]
	training_outputs=[]
	
	for now in range(99):
		print now
		flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/samples_'+str(now),'/tmp/samples',0,1,0)
		with open('/tmp/samples', 'r') as f:
			temp=pickle.load(f)
			print len(temp[0])
		if training_inputs==[]:
			training_inputs=temp[0]
			training_outputs=temp[1]
		else:
			training_inputs=np.append(training_inputs,temp[0],axis=0)
			training_outputs=np.append(training_outputs,temp[1],axis=0)
		print len(training_inputs)
	
	print len(training_inputs)
	
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
	
	
	init=tf.global_variables_initializer()
	session=tf.Session()
	session.run(init)
	
	flag=s3func.s3_download_file(s3,AWS_S3_bucket,'data/model_0_new','/tmp/model',0,1,0)
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
	print time.time()-st

#test_result([20,100,100,100,100,100,1],1000000)
#initialize_samples([20,100,100,100,100,100,1],1000000)
#initialize_weights([20,100,100,100,100,100,1])
