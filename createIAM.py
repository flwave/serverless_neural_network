import boto3
import json

client = boto3.client('iam')
"""
response = client.create_role(
    AssumeRolePolicyDocument=json.dumps('arn:aws:iam::aws:policy/AmazonS3FullAccess'),
    Path='/',
    RoleName='Test-Role'
)
"""
"""
response = client.list_roles(
    PathPrefix='/',
    MaxItems=123
)

print response['Roles'][0]['AssumeRolePolicyDocument']['Statement']
print response['Roles'][0]['RoleName']
print response['Roles'][1]['AssumeRolePolicyDocument']['Statement']
print response['Roles'][1]['RoleName']
print '\n\n\n\n\n'

print response['Roles'][0]
print '--------------------------------'
print response['Roles'][1]
"""
response = client.list_attached_role_policies(
    RoleName='testrole',
    PathPrefix='/',
    MaxItems=123
)
print response

print 'finished'
