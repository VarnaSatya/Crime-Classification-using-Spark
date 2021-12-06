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


import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession



def preprocess(df):
    #categorical to numerical

    indexers = [
    StringIndexer(inputCol="Category", outputCol="label"),  
    StringIndexer(inputCol="PdDistrict", outputCol="pddis"), 
    StringIndexer(inputCol="DayOfWeek", outputCol="dof")]

    pipeline = Pipeline(stages=indexers) 
    indexed = pipeline.fit(df).transform(df) 

    #normalize X and Y

    vector_assembler = VectorAssembler(inputCols=['X','Y'], outputCol="SS_features")
    indexed = vector_assembler.transform(indexed)
    minmax_scaler = MinMaxScaler(inputCol="SS_features", outputCol="scaled")
    scaled = minmax_scaler.fit(indexed).transform(indexed)
    scaled=scaled.withColumn("s", vector_to_array("scaled")).select(['Dates','Category','pddis','dof','label']+[col("s")[i] for i in range(2)])

    #splitting date

    transformed = (scaled
        .withColumn("day", dayofmonth(col("Dates")))
        .withColumn("month", date_format(col("Dates"), "MM"))
        .withColumn("year", year(col("Dates")))
        .withColumn('second',second(df.Dates))
        .withColumn('minute',minute(df.Dates))
        .withColumn('hour',hour(df.Dates))
        )

    from pyspark.sql.types import IntegerType
    data_df = transformed.withColumn("month", transformed["month"].cast(IntegerType()))

    #making featurized vector

    #['Dates','pddis', 'dof','s[0]','s[1]','day','hour','minute','year']

    #columns with the most correlation with label
    data_df=data_df.select('pddis','s[0]','s[1]','hour','minute','year','label')

    #encode label with dictionary values

    hasher = FeatureHasher(inputCols=['pddis','s[0]','s[1]','hour','minute','year'],
                        outputCol="features")

    featurized = hasher.transform(data_df)

    return featurized



sc=SparkContext('local[2]',appName="crime")
ss=SparkSession(sc)
  
ssc=StreamingContext(sc,2)
dataStream=ssc.socketTextStream('localhost',6100)
words=dataStream.flatMap(lambda line : line.split('}\}'))

deptSchema = 'Dates TIMESTAMP,Category STRING,Descript STRING,DayOfWeek STRING,PdDistrict STRING,Resolution STRING,Address STRING,X DOUBLE,Y DOUBLE'

def j(rdd):
    df1=rdd.collect()
    #deptDF1 = ss.createDataFrame(rdd, schema = deptSchema)
    #deptDF1.show()

    if df1!=[]:
        #print('\n\n',df1,type(df1[0][1:-1]),'\n\n')
        d = json.loads(df1[0])
        #print(d)
        dictList= lambda x: d[x]
        df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])  
        #df.show()
        df1=preprocess(df)
        df1.show()

    
    
words.foreachRDD(lambda x:j(x))

ssc.start()
ssc.awaitTermination()
ssc.stop()
