import sys
import pandas as pd
import matplotlib 
import numpy as np
import scipy as sp
import IPython
import sklearn
import mglearn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier


from sklearn.datasets import load_iris
iris_dataset= load_iris()

print ("keys of iris dataset would be \n{}".format(iris_dataset.keys()))
print("Target names: {}".format(iris_dataset['target_names']))

print("Feature names: \n{}".format(iris_dataset['feature_names']))

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), marker='o',
hist_kwds={'bins': 20}, s=60, alpha=.8)
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,Y_train)
X_new = np.array([[8, 3, 5, 2]])
print("X_new.shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
iris_dataset['target_names'][prediction]))
y_pred = knn.predict(X_test)
print("{}".format(knn.score(X_test,Y_test)))
X,y= mglearn.datasets.make_wave(n_samples = 40)
plt.plot(X,y,'')




