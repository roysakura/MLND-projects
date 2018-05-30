# Impute missing values with mean attribute values
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
import pandas
import numpy
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]
X[X == 0] = numpy.nan
imputer = Imputer(missing_values='NaN', strategy='mean')
imputedX = imputer.fit_transform(X)
print(imputedX)
