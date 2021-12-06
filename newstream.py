import pyspark
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.session import SparkSession


sc=SparkContext('local[2]',appName="crime")
ss=SparkSession(sc)
  
ssc=StreamingContext(sc,2)
dataStream=ssc.socketTextStream('localhost',6100)
words=dataStream.flatMap(lambda line : line.split(","))

def j(rdd):
    df=rdd.collect()  
    print(df)
    
    
words.foreachRDD(lambda x:j(x))

ssc.start()
ssc.awaitTermination()
ssc.stop()
