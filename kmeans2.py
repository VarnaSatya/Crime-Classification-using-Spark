from csv import reader
from pyspark.sql import SparkSession,Row
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from pyspark.sql.functions import hour, date_format, to_date,month,year,isnull, when, count, col,round
from pyspark.sql import functions as F
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
import os
os.environ["PYSPARK_PYTHON"] = "python3"

# read data from the data storage
# i've uploaded the data into databricks community at first. 
crime_data_lines = sc.textFile('/home/pes2ug19cs418/Downloads/train.csv')
#prepare data 
df_crimes = crime_data_lines.map(lambda line: [x.strip('"') for x in next(reader([line]))])
#get header
header = df_crimes.first()

#remove the first line of data
crimes = df_crimes.filter(lambda x: x != header)




spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
    


df_opt1 = spark.read.format("csv").option("header", "true").load("/home/pes2ug19cs418/Downloads/train.csv")
df_opt1.createOrReplaceTempView("sf_crime")
df_new = df_opt1.withColumn('Longitude', df_opt1['X'].cast('double')) \
                 .withColumn('Latitude', df_opt1['Y'].cast('double')) \
                 .withColumn("Incident Date",to_date(df_opt1.Dates, "MM/dd/yyyy")) 
# extract month and year from incident date
df_new = df_new.withColumn('month',month(df_new['Incident Date']))
df_new = df_new.withColumn('year', year(df_new['Incident Date']))
test = df_new
test = test.withColumn('Latitude', when(test.Latitude==90,np.nan).otherwise(test.Latitude))
test = test.withColumn('Longitude', when(test.Longitude==-120.5,np.nan).otherwise(test.Longitude))
from pyspark.sql.functions import col,avg
avg_lat = df_new.select(avg(df_new.Latitude)).collect()[0][0] 
avg_long = df_new.select(avg(df_new.Longitude)).collect()[0][0] 
test = test.withColumn('Latitude', when(test.Latitude == np.nan,avg_lat).otherwise(test.Latitude))
test = test.withColumn('Longitude', when(test.Longitude == np.nan,avg_long).otherwise(test.Longitude))
a = avg(test.Longitude)
vecAssembler = VectorAssembler(inputCols=["Latitude", "Longitude"], outputCol="features")
new_df = vecAssembler.transform(test)
kmeans = KMeans().setK(4).setSeed(1)
model = kmeans.fit(new_df)
predictions = model.transform(new_df)
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))



