from pyspark.sql import SparkSession
from pyspark.context import SparkContext
from pyspark.sql.functions import * 
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
import sys


sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

df = sqlContext.read.load("/opt/spark/bin/Project/train.csv", 
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

indexer = StringIndexer(inputCol="Category", outputCol="categoryIndex")
indexed = indexer.fit(df).transform(df)
indexed.show()
indexed.groupBy(['Category','categoryIndex']).count()
x.show(x.count(),False)