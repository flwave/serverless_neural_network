import time
import os
import math

def BMDP_readM():
	fid=open('M','r')
	a=fid.readlines()
	M=[]
	for r in range(len(a)):
		temp=''
		Mr=[]
		for c in range(len(a[r])):
			if a[r][c]!=',':
				temp=temp+a[r][c]
			else:
				Mr.append(float(temp))
				temp=''
		Mr.append(float(temp))
		M.append(Mr)
	return M

def write_file(key,string):
	with open(key, 'wb') as data:
		data.write(string)
	return 0
	
def add_file(key,string):
	with open(key, 'a') as data:
		data.write(string)
	return 0
	
def read_file(key,timeout,once):
	if once:
		if os.path.exists(key):
			fid=open(key,'r')
			data=fid.read()
			fid.close()
			return data
		else:
			return 0
	else:
		st=time.time()
		found=0
		while time.time()-st<timeout:
			if os.path.exists(key):
				found=1
				break
			time.sleep(3)
		if found:
			fid=open(key,'r')
			data=fid.read()
			fid.close()
			return data
		else:
			return 0


def write_2Dmatrix(key,m):
	with open(key, 'wb') as data:
		data.write(str(m))
	return 0
	
def read_2Dmatrix(key):
	fid=open(key,'r')
	M=[]
	line='N'
	while len(line)!=0:
		line=fid.readline()
		temp=''
		Mr=[]
		start=0
		for i in range(len(line)):
			if line[i]>='0' and line[i]<='9' or line[i]=='.' or line[i]=='-':
				start=1
				temp+=line[i]
			elif start==1:
				start=0
				Mr.append(float(temp))
				temp=''
		if len(temp)>0:
			Mr.append(float(temp))
		if len(Mr)>0:
			M.append(Mr)
	fid.close()
	return M

def read_alllines(filename):
	output=[]
	fid=open(filename,'r')
	data=fid.readline()
	while len(data)>0:
		if data!='\n':
			if data[-1]=='\n':
				data=data[:-1]
			output.append(data)
		data=fid.readline()
	fid.close()
	return output

def vdist(v1,v2):
	dist=0
	for i in range(len(v1)):
		dist+=(v1[i]-v2[i])**2
	return math.sqrt(dist)
	
