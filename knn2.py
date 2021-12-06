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


indexed2=indexed2.drop('Resolution')
indexed2=indexed2.drop('DayOfWeek')
indexed2=indexed2.drop('Category')



indexedtest1=indexedtest1.drop('DayOfWeek')






features = ["DayofWeekIndex", "PdDistrictIndex",  "X", "Y"]



X_train=np.array(indexed2.select(features).collect())

y_train=np.array(indexed2.select("categoryIndex").collect()).reshape(-1)

X_train=X_train.astype('float64')


X_test=np.array(indexedtest1.select(features).collect())
X_test=X_test.astype('float64')

from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train,y_train,test_size=0.25,random_state=42)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=100)
knn.fit(X_train1,y_train1)
pred = knn.predict(X_test1)
from sklearn.metrics import accuracy_score
accuracy_score(pred,y_test1)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)


indexedtest2 = indexedtest1.toPandas()


result_dataframe = pd.DataFrame({
    "Id": indexedtest2["Id"]
})
data_dict1 = sorted(data_dict.items(), key = lambda kv: kv[0])


result_dataframe["Prediction"]=predictions.tolist()
key_list = list(data_dict.keys())
value_list = list(data_dict.values())

arr=[]

for i in result_dataframe.iloc[:,1]:
	position=value_list.index(i)
	arr.append(key_list[position])
	


result_dataframe["CrimeCategory"]=arr

#result_dataframe.to_csv("submission_knn.csv", index=False)
