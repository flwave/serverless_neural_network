import s3func
import boto3
import sys
AWS_S3_bucket='lf-src'

if len(sys.argv)>1:
	s3func.s3_clear_bucket(AWS_S3_bucket,sys.argv[1])
	
