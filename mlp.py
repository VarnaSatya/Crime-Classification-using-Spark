from sklearn.neural_network import MLPClassifier
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
from stream1.py import mlp_fg

def mlp(df):
    X_train=list(df.select('features').collect())
    y_train=list(df.select('label').collect())

    #mlp_gs = MLPClassifier(max_iter=10)
    '''
    parameter_space = {
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive'],
    }
    from sklearn.model_selection import GridSearchCV
    '''

    #clf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
    #clf.partial_fit(X_train, y_train) # X is train samples and y is the corresponding labels
    #print('Best parameters found:\n', clf.best_params_)

    l=list(range(39))
    classes=np.array(l)
    mlp_gs.partial_fit(X_train,y_train,classes=classes)

    y_true, y_pred = y_val , clf.predict(X_val)
    from sklearn.metrics import classification_report
    print('Results on the test set:')
    print(classification_report(y_true, y_pred))
