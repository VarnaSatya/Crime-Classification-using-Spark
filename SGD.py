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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from pyspark.ml.classification import MultilayerPerceptronClassifier

sc = SparkContext.getOrCreate()
sqlContext = SQLContext(sc)

df = sqlContext.read.load("/home/pes2ug19cs418/Downloads/train.csv", 
                      format='com.databricks.spark.csv', 
                      header='true', 
                      inferSchema='true')

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

#['Dates','pddis', 'dof','s[0]','s[1]','day','hour','minute','year']

#columns with the most correlation with label
data_df=data_df.select('pddis','s[0]','s[1]','hour','minute','year','label')

hasher = FeatureHasher(inputCols=['pddis','s[0]','s[1]','hour','minute','year'],
                       outputCol="features")

featurized = hasher.transform(data_df)
featurized.show()

X, Y = datasets.make_classification(n_samples=32000, n_features=30, n_informative=20, n_classes=2)
X, Y = datasets.make_classification(n_samples=32000, n_features=30, n_informative=20, n_classes=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.9, random_state=123)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

X_train, X_test = X_train.reshape(-1,32,30), X_test.reshape(-1,32,30)
Y_train, Y_test = Y_train.reshape(-1,32), Y_test.reshape(-1,32)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=123)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

#SGD Classifier

classifier = SGDClassifier(random_state=123)
classifier = Perceptron(tol=1e-3, random_state=0)

epochs = 10

for k in range(epochs):
    for i in range(X_train.shape[0]):
        X_batch, Y_batch = X_train[i], Y_train[i]
        classifier.partial_fit(X_batch, Y_batch, classes=list(range(2))) ## Partially fitting data in batches
        
Y_test_preds = []
for j in range(X_test.shape[0]): ## Looping through test batches for making predictions
    Y_preds = classifier.predict(X_test[j])
    Y_test_preds.extend(Y_preds.tolist())

print("Test Accuracy      : {}".format(accuracy_score(Y_test.reshape(-1), Y_test_preds)))
