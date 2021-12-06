from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
from pyspark.conf import SparkConf
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
from multiprocessing.pool import ThreadPool
import multiprocessing as multiprocessing

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)
'''
models={}
models[1]=NaiveBayes(modelType='multinomial',smoothing=0.0)
models[2]=NaiveBayes(modelType='multinomial',smoothing=1.0)
classifiers=[1,2]

classif=sc.parallelize(classifiers)

def exec():
    def f(i):
        nbmodel = models[i].fit(train)
        predictions = nbmodel.transform(test)
        evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
        x=evaluator.evaluate(predictions)
        return x
    return f

x=classif.mapPartitions(exec())

if __name__="__main__":
    pool=ThreadPool(processes=3)
    opt=pool.map()
'''

import threading
from pyspark import SparkContext, SparkConf

models={}

def task(sc, i):
    models[i]=NaiveBayes(modelType='multinomial',smoothing=i)
    nbmodel = models[i].fit(train)
    predictions = nbmodel.transform(test)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
    print(evaluator.evaluate(predictions))

def run_multiple_jobs():
  #conf = SparkConf().setMaster('local[*]').setAppName('appname')
  # Set scheduler to FAIR: http://spark.apache.org/docs/latest/job-scheduling.html#scheduling-within-an-application
  #conf.set('spark.scheduler.mode', 'FAIR')
  #sc = SparkContext(conf=conf)
  for i in (0.0,1.0):
    t = threading.Thread(target=task, args=(sc, i))
    t.start()
    print('spark task', i, 'has started')

def kill_current_spark_context():
    SparkContext.getOrCreate().stop()

#kill_current_spark_context()
run_multiple_jobs()
