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
from telnetlib import Telnet
import time
import json
import pickle
import socket
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row
from collections import OrderedDict

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(X, y)

partial_fit(X, y)
