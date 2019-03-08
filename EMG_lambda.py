import boto3
import lambdafunc
import sys

if len(sys.argv)>1:
	if sys.argv[1]=='0':
		lambdafunc.lambda_shut_down('lambdarole','testfunc')
		print 'lambda shut down'
	elif sys.argv[1]=='1':
		lambdafunc.lambda_create_IAM_role('lambdarole')
		print 'lambda created'
	elif sys.argv[1]=='2':
		lambdafunc.lambda_delete_IAM_role('lambdarole')
		print 'lambda deleted'
	else:
		lambdafunc.lambda_shut_down('lambdarole','testfunc')
		print 'lambda shut down'
else:
	lambdafunc.lambda_shut_down('lambdarole','testfunc')
	print 'lambda shut down'
