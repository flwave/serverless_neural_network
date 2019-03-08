# Serverless Computing for Neural Network Training

<p align="center">
  <b>Maintained by: Lang Feng, Prabhakar Kudva, Dilma Da Silva and Jiang Hu </b><br>
  <b>Contact: flwave@tamu.edu, kudva@us.ibm.com, dilma@cse.tamu.edu, jianghu@tamu.edu</b><br>
</p>

This repository contains source code to run neural network training on Amazon Web Servers (AWS) Lambda serverless environment. 
The source code is used for the experiments in the following paper:

L. Feng, P. Kudva, D. Da Silva and J. Hu, "Exploring Serverless Computing for
Neural Network Training," IEEE International Conference on Cloud Computing
(IEEE Cloud), 2018

## Prerequisite

Amazon Web Servers (AWS) account is needed. 

All the codes are tested under Ubuntu 16.04 64-bit. 

Packages needed:

Python 2.7

Boto3

Tensorflow

An offline Tensorflow library, which should be put in "tensorflow1_0_0" folder.

Cifar-10 datasets, which should be put in "cifar-10" folder.

MNIST datasets (t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz, train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz),
 which should be put in same folder as the README.md.

## Data Preparation

Open each Python code file, replace the value of "AWS_region", "AWS_S3_bucket" and "AWS_lambda_role" (if exist) according to your AWS account.

Use the following command to upload the codes to AWS Lambda

\>\>./uploadtf

To upload the Case A dataset and model, uncomment last two lines of "NNtf5.py" and use the following command:

\>\>python NNtf5.py

To upload the Case B dataset and model, uncomment last two lines of "CNNtf.py" and use the following command:

\>\>python CNNtf.py

To upload the Case C dataset, uncomment the last line of "NNtfpt.py" and use the following command:

\>\>python NNtfpt.py

Create a IAM role using the following command:

\>\>python EMG_lambda.py 1

Create an S3 bucket of which the name is the same as the value of "AWS_S3_bucket".

## Run Tests

To train Case A or Case B, uncomment the line after "1. For training Case A" or "3. For training Case B", and run the following command:

\>\>python lambda.py

The trained model will be "data/model_0_new" or "data/modelcifar_0_new" in your S3 bucket.

To do cost or performance/cost ratio optimization, uncomment the line after "2. For cost(performance/cost) optimization of Case A",
and run the following command:

\>\>python lambda.py

The traces will be in a file named "timestamp/timestamp_monitor_x.tsp" in your S3 bucket.

To do hyperparameter tuning, uncomment the line after "4. For hyperparameter tuning Case C", and run the following command:

\>\>python lambda.py

The traces will be in the folder named "timestamp" in your S3 bucket.

## Notes:

• Running the codes will generate many AWS Lambda events. It is possible that AWS Lambda will miss some events. If this happened, 
A folder named "error" will be created. If this frequently happens, please change to another AWS region.

• Each time when you try to run a test, please ensure there is no other Lambda instances running. (This can be checked in CloudWatch)

• It is possible that the program gets into a dead lock (It will incur a lot of cost). If so, please run the following command immediately:

\>\>python EMG_lambda.py 2

After using this command, wait for several minutes, and run the following command again:

\>\>python EMG_lambda.py 1

• The authors (Lang Feng, Prabhakar Kudva, Dilma Da Silva and Jiang Hu) have no responsibility to any monetary cost incurred by using the codes in this repository.

Please send your comments to flwave@tamu.edu. We would highly appreciate your comments.
