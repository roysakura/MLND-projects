# Delete rows with NaN values
import pandas
import numpy
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['Code', 'Clump-Thickness', 'Cell-Size', 'Cell-Shape', 'Adhesion', 'Single-Cell-Size', 'Bare-Nuclei', 'Chromatin', 'Nucleoli', 'Mitoses', 'Class']
dataset = pandas.read_csv(url, names=names)
print(dataset.shape)
dataset[['Bare-Nuclei']] = dataset[['Bare-Nuclei']].replace('?', numpy.NaN)
dataset.dropna(axis=0, how='any', inplace=True)
print(dataset.shape)
