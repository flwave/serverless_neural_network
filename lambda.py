import boto3
import base64
import json
import botocore
import util
import time
import s3func
import datetime

AWS_region='us-west-2'
#AWS_region='us-east-1'
AWS_S3_bucket='lf-src'
AWS_lambda_role='arn:aws:iam::402880700135:role/lambdarole'
FUNCTION='testfunc'
inputs={}

client = boto3.client('lambda',region_name = AWS_region)
print 'client created'

response = client.list_functions(
    MaxItems=5
)

found=0
for k in response['Functions']:
	if k['FunctionName']==FUNCTION:
		found=1
		break

if found:
	response = client.delete_function(
    FunctionName=FUNCTION
)
	print 'found replicated function, deleted'

print 'ready to create function'

response = client.create_function(
    Code={
	'S3Bucket': AWS_S3_bucket,
        'S3Key': 'funcz.lamf',
    },
    Description='',
    FunctionName=FUNCTION,
    #Handler='NNtf5.lambda_handler',
    #Handler='CNNtf.lambda_handler',
    Handler='NNtfpt.lambda_handler',
    MemorySize=512,
    Publish=True,
    Role=AWS_lambda_role,
    Runtime='python2.7',
    Timeout=300,
    VpcConfig={
    },
)
print 'function created'

#mlayers: the layers for merging weights, mlayers[0] must be 1, which means the bottom layer. mlayers[-1] is the workers.
#1. For training Case A
#inputs={'state':0,'mod':1,'batchnumber':20,'slayers':[],'mlayers':[1,5,20,50,100],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':100000,'maxiter':20,'nowiter':0,'roundtime':250,'rounditer':20}
#2. For cost(performance/cost) optimiation of Case A
#inputs={'state':0,'mod':2,'batchnumber':20,'slayers':[],'mlayers':[1,5,20,50,100],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':100000,'maxiter':10,'nowiter':0,'roundtime':250,'rounditer':20}
#3. For training Case B
#inputs={'state':0,'mod':1,'batchnumber':20,'slayers':[],'mlayers':[1,5,20,50,100],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':50000,'maxiter':10,'nowiter':0,'roundtime':250,'rounditer':5}
#4. For hyperparameter tuning Case C
#inputs={'state':0,'mod':1,'batchnumber':20,'slayers':[],'mlayers':[1,5],'layers':[20,100,100,100,100,100,1],'pos':[0,0],'ns':50000,'maxiter':10,'nowiter':0,'roundtime':250,'rounditer':5}

response = client.invoke(
    FunctionName=FUNCTION,
	InvocationType='Event',
    Payload=json.dumps(inputs)
)

print datetime.datetime.now()
