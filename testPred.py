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
import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, create_map, lit
from itertools import chain
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cluster import MiniBatchKMeans
import logging
from sklearn.neural_network import MLPClassifier
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
import pickle
import os
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()

df = sqlContext.read.load("/opt/spark/bin/Project/test.csv", 
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')


data_dict = {'FRAUD':1, 'SUICIDE':2, 'SEX OFFENSES FORCIBLE':3, 'LIQUOR LAWS':4, 
'SECONDARY CODES':5, 'FAMILY OFFENSES':6, 'MISSING PERSON':7, 'OTHER OFFENSES':8, 
'DRIVING UNDER THE INFLUENCE':9, 'WARRANTS':10, 'ARSON':11, 'SEX OFFENSES NON FORCIBLE':12,
'FORGERY/COUNTERFEITING':13, 'GAMBLING':14, 'BRIBERY':15, 'ASSAULT':16, 'DRUNKENNESS':17,
'EXTORTION':18, 'TREA':19, 'WEAPON LAWS':20, 'LOITERING':21, 'SUSPICIOUS OCC':22, 
'ROBBERY':23, 'PROSTITUTION':24, 'EMBEZZLEMENT':25, 'BAD CHECKS':26, 'DISORDERLY CONDUCT':27,
'RUNAWAY':28, 'RECOVERED VEHICLE':29, 'VANDALISM':30,'DRUG/NARCOTIC':31, 
'PORNOGRAPHY/OBSCENE MAT':32, 'TRESPASS':33,'VEHICLE THEFT':34, 'NON-CRIMINAL':35, 
'STOLEN PROPERTY':36, 'LARCENY/THEFT':37, 'KIDNAPPING':38,'BURGLARY':39}

item_dict = dict([(value, key) for key, value in data_dict.items()])

pddis={'MISSION':1,'BAYVIEW':2,'CENTRAL':3,'TARAVAL':4, 'TENDERLOIN':5,'INGLESIDE':6, 'PARK':7,'SOUTHERN':8, 'RICHMOND':9,'NORTHERN':10}

def indexNum(df):
    mapping_expr2 = create_map([lit(x) for x in chain(*pddis.items())])
    df1=df.withColumn("pddis", mapping_expr2.getItem(col("PdDistrict")))
    return df1

def preprocess(df):
    indexed= indexNum(df)
    #categorical to numerical
    vector_assembler = VectorAssembler(inputCols=['X','Y'], outputCol="SS_features")
    indexed = vector_assembler.transform(indexed)
    minmax_scaler = MinMaxScaler(inputCol="SS_features", outputCol="scaled")
    scaled = minmax_scaler.fit(indexed).transform(indexed)
    scaled=scaled.withColumn("s", vector_to_array("scaled")).select(['Dates','pddis']+[col("s")[i] for i in range(2)])
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
    #normalize year, hour, minute
    #making featurized vector
    #['Dates','pddis', 'dof','s[0]','s[1]','day','hour','minute','year']
    #columns with the most correlation with label
    data_df=data_df.select('pddis','s[0]','s[1]','hour')
    return data_df


filename = 'mlpMod.sav'
filename1 = 'sgd.sav'
filename2 = 'mnb.sav'
filename3 = 'kmeans.sav'

x=preprocess(df)
pandasDF=x.toPandas()

mlp_gs= pickle.load(open(filename, 'rb'))
mlp_y_pred = mlp_gs.predict(pandasDF.iloc[:,:-1])

sgdmodel= pickle.load(open(filename1, 'rb'))
sgd_y_pred=sgdmodel.predict(pandasDF)

mnb= pickle.load(open(filename2, 'rb'))
mnb_y_pred =mnb.predict(pandasDF)

kmns= pickle.load(open(filename3, 'rb'))
kmns_y_pred =kmns.predict(pandasDF)

pandasDF['mlpLABEL']=np.array(list(map(lambda x: item_dict[x], mlp_y_pred)))
pandasDF['sgdLABEL']=np.array(list(map(lambda x: item_dict[x], sgd_y_pred)))
pandasDF['mnbLABEL']=np.array(list(map(lambda x: item_dict[x], mnb_y_pred)))


print(pandasDF)

pandasDF.to_csv('testOut.csv', sep='\t', encoding='utf-8')

df=pd.DataFrame(kmns_y_pred)
df.to_csv('Clustering.csv', sep='\t', encoding='utf-8')


import matplotlib.pyplot as plt
plt.style.use('ggplot')
q8_res = pandasDF

fig, ax = plt.subplots()
ax.scatter(q8_res['s[0]'], q8_res['s[1]'], c=(kmns_y_pred),cmap=plt.cm.jet, alpha=0.9)
ax.set_title("5 Clusters")
plt.show()



