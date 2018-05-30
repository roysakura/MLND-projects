# Identify Features with Low Variance
from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
# load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
# feature selection
threshold = 0.8 * (1 - 0.8)
test = VarianceThreshold(threshold)
fit = test.fit(X)
print(fit.variances_)
features = fit.transform(X)
print(features)

