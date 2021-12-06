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
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

df = sqlContext.read.load("/home/pes2ug19cs418/Downloads/train.csv", 
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

#df = df.limit(10000)

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

hasher = VectorAssembler(inputCols=['pddis', 'dof','s[0]','s[1]','day','hour','minute','year'],
                       outputCol='features')

featurized = hasher.transform(data_df)
featurized.show()

#test train split
train,test,val=featurized.randomSplit([0.7, 0.15, 0.15],seed=50)


mlpc=MultilayerPerceptronClassifier( featuresCol='features',labelCol=outputCol,layers = [4,16,2],maxIter=1000,blockSize=8,seed=7,solver='gd')
ann = mlpc.fit(train)
pred = ann.transform(test)
evaluator = MulticlassClassificationEvaluator(labelCol='Class ',predictionCol=pred,metricName='f1')
ann_f1 = evaluator.evaluate(pred)
ann_f1


