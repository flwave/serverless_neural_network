import boto3
import sys
AWS_S3_bucket='lf-src'
s3 = boto3.resource('s3')
s3.Bucket(AWS_S3_bucket).upload_file(sys.argv[1], 'funcz.lamf')
print 'success'
