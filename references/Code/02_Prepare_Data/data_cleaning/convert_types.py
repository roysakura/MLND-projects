# Convert data types
import pandas
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset = pandas.read_csv(url, names=names)
print(dataset.dtypes)
dataset = dataset.astype(float)
print(dataset.dtypes)
