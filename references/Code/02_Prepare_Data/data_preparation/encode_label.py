# Convert a string class label to an integer
import pandas
from sklearn.preprocessing import LabelEncoder
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
dataset = pandas.read_csv(url, header=None)
array = dataset.values
y = array[:, 60]
encoder = LabelEncoder()
encoder.fit(y)
print(encoder.classes_)
encoded_y = encoder.transform(y)
print(encoded_y)
