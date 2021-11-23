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
import os
from sklearn.model_selection import train_test_split
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.functions import vector_to_array

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

indexed = indexed.withColumn("categoryIndex", indexed["categoryIndex"].cast(IntegerType()))
indexed=indexed.withColumn("categoryIndex",col("categoryIndex") + lit(1))


indexer1 = StringIndexer(inputCol="DayOfWeek", outputCol="DayofWeekIndex")
indexed1 = indexer1.fit(indexed).transform(indexed)


indexed1 = indexed1.withColumn("DayofWeekIndex", indexed1["DayofWeekIndex"].cast(IntegerType()))
indexed1=indexed1.withColumn("DayofWeekIndex",col("DayofWeekIndex") + lit(1))

indexer2 = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrictIndex")
indexed2 = indexer2.fit(indexed1).transform(indexed1)

indexed2 = indexed2.withColumn("PdDistrictIndex", indexed2["PdDistrictIndex"].cast(IntegerType()))
indexed2=indexed2.withColumn("PdDistrictIndex",col("PdDistrictIndex") + lit(1))



indexertest = StringIndexer(inputCol="DayOfWeek", outputCol="DayofWeekIndex")
indexedtest = indexertest.fit(test_data).transform(test_data)
indexedtest = indexedtest.withColumn("DayofWeekIndex", indexedtest["DayofWeekIndex"].cast(IntegerType()))
indexedtest=indexedtest.withColumn("DayofWeekIndex",col("DayofWeekIndex") + lit(1))

indexertest1 = StringIndexer(inputCol="PdDistrict", outputCol="PdDistrictIndex")
indexedtest1 = indexertest1.fit(indexedtest).transform(indexedtest)
indexedtest1 = indexedtest1.withColumn("PdDistrictIndex", indexedtest1["PdDistrictIndex"].cast(IntegerType()))
indexedtest1=indexedtest1.withColumn("PdDistrictIndex",col("PdDistrictIndex") + lit(1))
indexedtest1 = indexedtest1.withColumn("Y", indexedtest1["Y"].cast(FloatType()))
indexedtest1 = indexedtest1.withColumn("X", indexedtest1["X"].cast(FloatType()))

indexed2=indexed2.drop('Resolution')
indexed2=indexed2.drop('DayOfWeek')
indexed2=indexed2.drop('Category')
indexed2 = indexed2.withColumn("Y", indexed2["Y"].cast(FloatType()))
indexed2 = indexed2.withColumn("X", indexed2["X"].cast(FloatType()))


indexedtest1=indexedtest1.drop('DayOfWeek')


vector_assembler = VectorAssembler(inputCols=['X','Y'], outputCol="SS_features")
indexed3 = vector_assembler.transform(indexed2)
minmax_scaler = MinMaxScaler(inputCol="SS_features", outputCol="scaled")
scaled = minmax_scaler.fit(indexed3).transform(indexed3)
scaled=scaled.withColumn("s", vector_to_array("scaled")).select(['Dates','categoryIndex','PdDistrictIndex','DayofWeekIndex']+[col("s")[i] for i in range(2)])

vector_assembler = VectorAssembler(inputCols=['X','Y'], outputCol="SS_features")
indexedtest2 = vector_assembler.transform(indexedtest1)
minmax_scaler = MinMaxScaler(inputCol="SS_features", outputCol="scaled")
scaledtest = minmax_scaler.fit(indexedtest2).transform(indexedtest2)
scaledtest=scaledtest.withColumn("s", vector_to_array("scaled")).select(['Id','Dates','PdDistrictIndex','DayofWeekIndex']+[col("s")[i] for i in range(2)])

features = ["DayofWeekIndex", "PdDistrictIndex",  "s[0]", "s[1]"]



X_train=np.array(scaled.select(features).collect())

y_train=np.array(scaled.select("categoryIndex").collect()).reshape(-1)

X_train=X_train.astype('float64')


X_test=np.array(scaledtest.select(features).collect())
X_test=X_test.astype('float64')

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train,y_train,test_size=0.25,random_state=42)


"""
import xgboost as xgb
from sklearn.metrics import mean_squared_error

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train1,y_train1)

preds = xg_reg.predict(X_test1)

rmse = np.sqrt(mean_squared_error(y_test1, preds))
print("RMSE: %f" % (rmse))
print("ACC-->",accuracy_score(y_test1,preds))
"""

lj_m = LogisticRegression(solver="liblinear").fit(X_train1,y_train1)
predict = lj_m.predict(X_test1)
rmse = np.sqrt(mean_squared_error(y_test1, predict))
print("RMSE: %f" % (rmse))
print("ACC-->",accuracy_score(y_test1,predict))





