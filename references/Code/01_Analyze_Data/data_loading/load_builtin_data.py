# Built-in datasets

# Boston house prices dataset (13x506, reals, regression)
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.data.shape)
print(boston.DESCR)

# Iris flower dataset (4x150, reals, multi-label classification)
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape)
print(iris.DESCR)

# Diabetes dataset (10x442, reals, regression)
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
print(diabetes.data.shape)

# Hand-written digit dataset (64x1797, multi-label classification)
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data.shape)

# Linnerud psychological and exercise dataset (3x20,3x20 multivariate regression)
from sklearn.datasets import load_linnerud
linnerud = load_linnerud()
print(linnerud.data.shape)
