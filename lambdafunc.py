import boto3
import json

def lambda_create_IAM_role(name):
	client = boto3.client('iam')
	policy={
		'Version': '2012-10-17',
		'Statement': [
		{
			'Effect': 'Allow',
			'Principal': {
			'Service': 'lambda.amazonaws.com'
			},
		'Action': 'sts:AssumeRole'
		}
		]
	}

	response = client.create_role(
		RoleName=name,
		AssumeRolePolicyDocument=json.dumps(policy),
	)
	response = client.attach_role_policy(
		RoleName=name,
		PolicyArn='arn:aws:iam::aws:policy/AWSLambdaFullAccess'
	)
	response = client.attach_role_policy(
		RoleName=name,
		PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
	)
	response = client.attach_role_policy(
		RoleName=name,
		PolicyArn='arn:aws:iam::aws:policy/CloudWatchFullAccess'
	)

def lambda_delete_IAM_role(name):
	client = boto3.client('iam')
	response = client.list_roles()
	if 'Roles' in response.keys() and name in [response['Roles'][i]['RoleName'] for i in range(len(response['Roles']))]:
		response = client.detach_role_policy(
			RoleName=name,
			PolicyArn='arn:aws:iam::aws:policy/AWSLambdaFullAccess'
		)
		response = client.detach_role_policy(
			RoleName=name,
			PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
		)
		response = client.detach_role_policy(
			RoleName=name,
			PolicyArn='arn:aws:iam::aws:policy/CloudWatchFullAccess'
		)
		response = client.delete_role(
			RoleName=name
		)

def lambda_shut_down(rolename,funcname):
	lambda_delete_IAM_role(rolename)
	client = boto3.client('lambda',region_name = 'us-east-1')
	response = client.delete_function(
		FunctionName=funcname
	)
