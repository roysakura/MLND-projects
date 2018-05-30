# Correlation Matrix Plot

import matplotlib.pyplot as plt
import matplotlib
import pandas
import seaborn

matplotlib.style.use('ggplot')
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pandas.read_csv(url, names=names)
correlations = data.corr()
seaborn.heatmap(correlations)
plt.show()