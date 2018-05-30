# Load CSV from URL using NumPy
import numpy
import urllib
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = numpy.loadtxt(raw_data, delimiter=",")
print(dataset.shape)
