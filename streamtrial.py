#! /usr/bin/python3
from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
import sys
from pyspark.ml.linalg import DenseMatrix, Vectors
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.functions import col
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import abs
from pyspark.ml import Pipeline 
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from telnetlib import Telnet
import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from collections import OrderedDict

def convert_to_row(d: dict) -> Row:
    return Row(**OrderedDict(sorted(d.items())))

TCP_IP = "localhost"
TCP_PORT = 6100
telnet = Telnet(TCP_IP, TCP_PORT)

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

while 1:
    data=telnet.read_until(b'}}',timeout=2)
    my_json = data.decode('utf8')
    if my_json!='\n' and my_json!='':
        d = json.loads(my_json)
        dictList= lambda x: d[x]
        df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
        df.show()