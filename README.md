# Distributed Graph Embedding
#### This is an implementation of distributed Deepwalk and GCN, which is easy to get started :).
## Requirment
### Database 
> [Hadoop==2.7.7](https://mirrors.tuna.tsinghua.edu.cn/apache/hadoop/common/hadoop-2.7.7/) (Java jdk 1.8 is necessary)

#### Manage distributed data with Hadoop Distributed File System(HDFS) 

### Programming environment
> python==3.6\
> tensorflow==1.12\
> CUDA==9.0 (If GPU is available)\
> numpy==1.16.4\
> hdfs==2.5.8\
> scikit-learn=0.22.1

#### You may use [Anaconda](https://www.anaconda.com/) to create a new environment

## Introduction
### deepwalk.py
> The model of Deepwalk

### GCN.py
> The model of GCN based on sampling

### hdfs_upload.py
> Easily to generate graph with JSON format and upload to HDFS in parameter server

### walks.py
> To generate random walks for Deepwalk and computing graphs to GCN

### module.py
> The other functions: similarity computing, link prediction, node classification, negative sampling, generate training data and testing data

## Graph Format
> {\
>  &emsp;"source_0":\
>  &emsp;&emsp;{\
>  &emsp;&emsp;&emsp;"label":[node_label]\
>  &emsp;&emsp;&emsp;"edge":[target_0, target_1,...]\
> &emsp;&emsp;}\
> &emsp;&emsp;...\
&emsp;"source_N":\
>  &emsp;&emsp;{\
>  &emsp;&emsp;&emsp;"label":[node_label]\
>  &emsp;&emsp;&emsp;"edge":[target_0, target_1,...]\
> &emsp;&emsp;}\
> }

## Easy To Run
### In the parameter server (ps)
> python deepwalk.py --job ps --task [task_id]

### In the worker
> python deepwalk.py --job worker --task [task_id]

### Fill addresses of all servers in parallel computing before running with format [address:port] to deepwalk.py or GCN.py


