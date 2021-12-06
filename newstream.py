import findspark
findspark.init()
import time
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row,SQLContext
import sys
import requests
import json

#def aggregate_tweets_conf

conf=SparkConf()
conf.setAppName("BigData")
sc=SparkContext(conf=conf)

ssc=StreamingContext(sc,2)
ssc.checkpoint("checkpoint_BIGDATA")

dataStream=ssc.socketTextStream("localhost",6100)

words = dataStream.flatMap(lambda line: line.split(" "))
pairs = words.map(lambda word: (word, 1))
wordCounts = pairs.reduceByKey(lambda x, y: x + y)

# Print the first ten elements of each RDD generated in this DStream to the console
wordCounts.pprint()


#dataStream.pprint()



def j(rd):
    if not rd.isEmpty():
        df = spark.createDataFrame(rd) 
        df.show()
        #d=spark.read.json(rd)
        #d = json.loads()
        #dictList= lambda x: d[x]
        #df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
        #dictList= lambda x: d[x]
        #df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
        #df.show()
        #print(d)

'''
dataStream.foreachRDD(x => {
    my_json=x.map(_.value.toString)
    d = json.loads(my_json)
    dictList= lambda x: d[x]
    df=sc.parallelize(list(map(dictList,d))).map(convert_to_row).toDF(['Dates','Category','Descript','DayOfWeek','PdDistrict','Resolution','Address','X','Y'])
})
'''

#dataStream.foreachRDD(lambda x: j(x))
#dataStream.pprint()

#x=dataStream.map(lambda w:(w.split(';')[0],1))
#x.pprint()

ssc.start()
ssc.awaitTermination()
ssc.stop()


