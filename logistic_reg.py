from csv import reader
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from pyspark.sql.functions import hour, date_format, to_date,month,year,isnull, when, count, col,round,lit
from pyspark.sql import functions as F
import warnings
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler,StringIndexer
from pyspark.ml.classification import LogisticRegression
import os
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

os.environ["PYSPARK_PYTHON"] = "python3"
'''OUR TARGET AND THEIR TARGET IS NOT THE SAME 
WHICH IN TURN IS GIVING DIFFERENT NUMBERS IN THE DICTIONARY'''
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    


train_data = spark.read.format("csv").option("header", "true").load("/home/pes2ug19cs418/Downloads/train.csv")
test_data = spark.read.format("csv").option("header", "true").load("/home/pes2ug19cs418/Downloads/test.csv")
train_data.createOrReplaceTempView("sf_crime")

target=train_data.select("Category").distinct().collect()


data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count+=1

    
indexer = StringIndexer(inputCol="Category", outputCol="categoryIndex")
indexed = indexer.fit(train_data).transform(train_data)

indexed = indexed.withColumn("categoryIndex", indexed["categoryIndex"].cast('float'))
indexed=indexed.withColumn("categoryIndex",col("categoryIndex") + lit(1))


indexer1 = StringIndexer(inputCol="DayOfWeek", outputCol="DayofWeekIndex")
indexed1 = indexer1.fit(indexed).transform(indexed)


indexed1 = indexed1.withColumn("DayofWeekIndex", indexed1["DayofWeekIndex"].cast('float'))
indexed1=indexed1.withColumn("DayofWeekIndex",col("DayofWeekIndex") + lit(1))

indexer2 = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrictIndex")
indexed2 = indexer2.fit(indexed1).transform(indexed1)

indexed2 = indexed2.withColumn("PdDistrictIndex", indexed2["PdDistrictIndex"].cast('float'))
indexed2=indexed2.withColumn("PdDistrictIndex",col("PdDistrictIndex") + lit(1))



indexertest = StringIndexer(inputCol="DayOfWeek", outputCol="DayofWeekIndex")
indexedtest = indexertest.fit(test_data).transform(test_data)
indexedtest = indexedtest.withColumn("DayofWeekIndex", indexedtest["DayofWeekIndex"].cast('float'))
indexedtest=indexedtest.withColumn("DayofWeekIndex",col("DayofWeekIndex") + lit(1))

indexertest1 = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrictIndex")
indexedtest1 = indexertest1.fit(indexedtest).transform(indexedtest)
indexedtest1 = indexedtest1.withColumn("PdDistrictIndex", indexedtest1["PdDistrictIndex"].cast('float'))
indexedtest1=indexedtest1.withColumn("PdDistrictIndex",col("PdDistrictIndex") + lit(1))


indexed2=indexed2.drop('Resolution')
indexed2=indexed2.drop('DayOfWeek')
indexed2=indexed2.drop('Category')



indexedtest1=indexedtest1.drop('DayOfWeek')

"""
X_train = X_train.withColumn("DayofWeekIndex", 
                                  X_train["DayofWeekIndex"]
                                  .cast('float'))
                                  """
                                 
features = ["DayofWeekIndex", "PdDistrictIndex",  "X", "Y"]


train,test,val=indexed2.randomSplit([0.7, 0.15, 0.15],seed=50)


X_train=train.select(features)

y_train=train.select("categoryIndex")

#X_train=X_train.astype('float64')


X_test=indexedtest1.select(features)
#X_test=X_test.astype('float64')



lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
lrModel = lr.fit(X_train)
predictions = lrModel.transform(val)

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
evaluator.evaluate(predictions)

"""lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(X_train1)
print("Coefficients: \n" + str(lrModel.coefficientMatrix))
print("Intercept: " + str(lrModel.interceptVector))

trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))
"""
















