import boto3
import time

SLEEPTIME=1.0

class timer(object):
	start=[0.0,0.0]
	maxtime=[0.0,0.0]
	backuplocalmax=0.0
	endstate=[0,0.0]
	local=0
	stop=0
	def __init__(self,maxt):
		self.start=[0.0,time.time()]
		self.maxtime=maxt
		self.local=0
		self.stop=0
	def outtime(self):
		return [self.start,time.time(),self.maxtime]
	def local_start(self,sptime):
		self.start[0]=time.time()
		self.backuplocalmax=self.maxtime[0]
		if sptime>0.0:
			self.maxtime[0]=sptime
		self.local=1
	def query(self):
		t=time.time()
		tg=t-self.start[1]
		tl=t-self.start[0]
		if self.stop==1:
			return self.endstate
		if tg>self.maxtime[1]:
			self.stop=1
			if self.local==1 and tl<self.maxtime[0]:
				self.local=0
				self.endstate=[1,self.maxtime[0]-tl]
				self.maxtime[0]=self.backuplocalmax
				return self.endstate
			else:
				self.local=0
				self.endstate=[1,0.0]
				self.maxtime[0]=self.backuplocalmax
				return self.endstate
		if self.local==1 and tl>self.maxtime[0]:
			self.local=0
			self.maxtime[0]=self.backuplocalmax
			return [1,0.0]
		if self.local==1:
			return [0,self.maxtime[0]-tl]
		return [0,0.0]

def s3_create_bucket(client,bucket):
	if type(client)==int:
		client= boto3.client('s3')
	response = client.create_bucket(
    ACL='private',
    Bucket=bucket,
	)

def s3_delete_bucket(client,bucket):
	if type(client)==int:
		client= boto3.client('s3')
	response = client.delete_bucket(
    Bucket=bucket
	)

def s3_clear_bucket(bucket,prefix):
	client= boto3.client('s3')
	response = client.list_objects(
		Bucket=bucket,
		Prefix=prefix
	)
	if 'Contents' in response.keys():
		objs=response['Contents']
		for obj in objs:
			response = client.delete_object(
			Bucket=bucket,
			Key=obj['Key']
			)

def s3_write_string(bucket,key,string):
	s3 = boto3.client('s3')
	s3.put_object(Bucket=bucket,Body=string, Key=key)
	"""
	s3 = boto3.resource('s3')
	s3bucket = s3.Bucket(bucket)
	s3bucket.put_object(Body=string, Key=key)
	"""

def s3_read_file_v2(s3,bucket,key,timeout,once,delete):
	if type(timeout)!=int and type(timeout)!=float:
		return 0
	if type(s3)==int:
		s3 = boto3.client('s3')
	st=time.time()
	found=0
	if once:
		try:
			with open('/tmp/s3rtemp', 'wb') as data:
				s3.download_fileobj(bucket,key,data)
			data = open('/tmp/s3rtemp','r')
			found=1
		except:
			found=0
	else:
		while time.time()-st<timeout:
			try:
				with open('/tmp/s3rtemp', 'wb') as data:
					s3.download_fileobj(bucket,key,data)
				data = open('/tmp/s3rtemp','r')
				found=1
				break
			except:
				found=0
				time.sleep(SLEEPTIME)
	if found:
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return data.read()
	else:
		return 0

def s3_read_file(s3,bucket,key,timeout,once,delete):
	if type(timeout)!=int and type(timeout)!=float:
		return 0
	if type(s3)==int:
		s3 = boto3.client('s3')
	st=time.time()
	found=0
	if once:
		response=s3.list_objects(
			Bucket=bucket,
			Prefix=key
		)
		if 'Contents' in response.keys() and key in [i['Key'] for i in response['Contents']]:
			found=1
		"""
		if key in [i['Key'] for i in response['Contents']]:
			found=1
		"""
	else:
		while time.time()-st<timeout:
			response=s3.list_objects(
				Bucket=bucket,
				Prefix=key
			)
			if 'Contents' in response.keys() and key in [i['Key'] for i in response['Contents']]:
				found=1
				break
			time.sleep(SLEEPTIME)
			"""
			if key in [i['Key'] for i in response['Contents']]:
				found=1
				break
			"""
	if found:
		st=time.time()
		"""
		s3r = boto3.resource('s3')
		s3r.meta.client.download_file(bucket, key, key)
		"""
		with open('/tmp/s3rtemp', 'wb') as data:
			s3.download_fileobj(bucket,key,data)

		data = open('/tmp/s3rtemp','r')
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return data.read()
	else:
		return 0

def s3_download_file(s3,bucket,key,tkey,timeout,once,delete):
	if type(timeout)!=int and type(timeout)!=float:
		return 0
	if type(s3)==int:
		s3 = boto3.client('s3')
	st=time.time()
	found=0
	if once:
		try:
			with open(tkey, 'wb') as data:
				s3.download_fileobj(bucket,key,data)
			found=1
		except:
			found=0
	else:
		while time.time()-st<timeout:
			try:
				with open(tkey, 'wb') as data:
					s3.download_fileobj(bucket,key,data)
				found=1
				break
			except:
				found=0
				time.sleep(SLEEPTIME)
	if found:
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return 1
	else:
		return 0
		
def s3_download_file_timer(s3,bucket,key,tkey,tr,lst,delete):
	if type(s3)==int:
		s3 = boto3.client('s3')
	st=time.time()
	found=0
	tr.local_start(lst)
	while tr.query()[0]==0:
		try:
			with open(tkey, 'wb') as data:
				s3.download_fileobj(bucket,key,data)
			found=1
			break
		except:
			found=0
			time.sleep(SLEEPTIME)
	if found:
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return 1
	else:
		return 0
		
def s3_download_file_v2(s3,s3r,bucket,key,tkey,timeout,once,delete):
	if type(timeout)!=int and type(timeout)!=float:
		return 0
	if type(s3)==int:
		s3 = boto3.client('s3')
	if type(s3r)==int:
		s3r = boto3.resource('s3')
	st=time.time()
	found=0
	if once:
		try:
			s3r.meta.client.download_file(bucket,key,tkey)
			found=1
		except:
			found=0
	else:
		while time.time()-st<timeout:
			try:
				s3r.meta.client.download_file(bucket,key,tkey)
				found=1
				break
			except:
				found=0
				time.sleep(SLEEPTIME)
	if found:
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return 1
	else:
		return 0

def s3_delete_file(s3,bucket,key):
	if type(s3)==int:
		s3 = boto3.client('s3')
	response = s3.delete_object(
		Bucket=bucket,
		Key=key
	)

def s3_check_exist(s3,bucket,key,timeout,once,delete):
	if type(s3)==int:
		s3 = boto3.client('s3')
	found=0
	st=time.time()
	if once:
		response = s3.list_objects(
			Bucket=bucket,
			Prefix=key,
		)
		if 'Contents' in response.keys() and key in [i['Key'] for i in response['Contents']]:
			found=1
		else:
			found=0
	else:
		while time.time()-st<timeout:
			response = s3.list_objects(
				Bucket=bucket,
				Prefix=key,
			)
			if 'Contents' in response.keys() and key in [i['Key'] for i in response['Contents']]:
				found=1
				break
			time.sleep(SLEEPTIME)
	if found:
		if delete:
			response = s3.delete_object(
				Bucket=bucket,
				Key=key
			)
		return 1
	else:
		return 0
		
def s3_check_multi_exist(s3,bucket,prefix,keys,timeout,once):
	if type(s3)==int:
		s3 = boto3.client('s3')
	found=0
	st=time.time()
	if once:
		response = s3.list_objects(
			Bucket=bucket,
			Prefix=prefix,
		)
		if 'Contents' in response.keys():
			found=1
			for key in keys:
				if key not in [i['Key'] for i in response['Contents']]:
					found=0
					break
		else:
			found=0
	else:
		while time.time()-st<timeout:
			response = s3.list_objects(
				Bucket=bucket,
				Prefix=prefix,
			)
			if 'Contents' in response.keys():
				found=1
				for key in keys:
					if key not in [i['Key'] for i in response['Contents']]:
						found=0
						break
			else:
				found=0
			if found==1:
				break
			time.sleep(SLEEPTIME)
	if found:
		return 1
	else:
		return 0

def s3_check_multi_exist_timer(s3,bucket,prefix,keys,tr,lst):
	if type(s3)==int:
		s3 = boto3.client('s3')
	found=0
	tr.local_start(lst)
	while tr.query()[0]==0:
		response = s3.list_objects(
			Bucket=bucket,
			Prefix=prefix,
		)
		if 'Contents' in response.keys():
			found=1
			for key in keys:
				if key not in [i['Key'] for i in response['Contents']]:
					found=0
					break
		else:
			found=0
		if found==1:
			break
		time.sleep(SLEEPTIME)
	if found:
		return 1
	else:
		return 0
