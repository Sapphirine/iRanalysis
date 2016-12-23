# iRanalysis
P2P loan Data Analysis

## Authors
Shubham Bansal (shub1905)  
Siddharth Aman Varshney  

## Instructions
### Prerequisites
Please install spark in a standalone or a cluster mode.
### Usage
Submit the script to spark-submit command with the configuration as follows:  

````bash
usage: spark-submit main.py [-h] --train t --plot p [--file f] [--numT rft] [--iter gbi]

Big Data

optional arguments:
  -h, --help  show this help message and exit
  --train t   Train on provided file, must be a csv with header; 1: RF, 2:GBT; 3:Both
  --plot p    Generate prediction Data
  --file f    Dataset File; can be S3, HDFS or Local File;Default: Sample file in S3
  --numT rft  Number of trees in RF; default=500
  --iter gbi  Number of iterations in GBT; default=100
````

To generate the augmented dataset run ````Python src/data_utils.py````. This will generate an augmented dataset as referenced in the report.
