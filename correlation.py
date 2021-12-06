from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
import seaborn as sn
import matplotlib.pyplot as plt

# convert to vector column first
#data_df from after preprocessing

df=data_df.select('pddis', 'dof','s[0]','s[1]','day','hour','minute','year','month','label')
vector_col = "features"
assembler = VectorAssembler(inputCols=df.columns, outputCol=vector_col)
df_vector = assembler.transform(df).select(vector_col)

# get correlation matrix
matrix = Correlation.corr(df_vector, vector_col)

matrix.collect()[0]["pearson({})".format(vector_col)].values

pearsonCorr = Correlation.corr(df_vector, 'features', method='spearman').collect()[0][0]
print(str(pearsonCorr).replace('nan', 'NaN'))

l=pearsonCorr.toArray().tolist()

x_axis_labels = ['pddis', 'dof','s[0]','s[1]','day','hour','minute','year','month','label'] # labels for x-axis
y_axis_labels = ['pddis', 'dof','s[0]','s[1]','day','hour','minute','year','month','label'] # labels for y-axis

hm = sn.heatmap(data = l,xticklabels=x_axis_labels, yticklabels=y_axis_labels)
  
# displaying the plotted heatmap
plt.show()

#highest correlation
#'pddis','s[0]','s[1]','hour','minute','year','label' are on