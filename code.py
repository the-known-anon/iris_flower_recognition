# import all the libraries
import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the data set
iris_dataset = load_iris()

print("keys of iris dataset would be \n{}".format(
    iris_dataset.keys()), end='\n\n')

print("Target names: {}".format(iris_dataset['target_names']), end='\n\n')

print("Feature names: \n{}".format(
    iris_dataset['feature_names']), end='\n\n')

# Now we will split the dataset, into test and train datasets
X_train, X_test, Y_train, Y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("Y_train shape: {}".format(Y_train.shape))

# Create Pandas DataFrame
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# Create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), marker='o',
                                 hist_kwds={'bins': 20}, s=60, alpha=.8)

# We will be using the KNN model for training on the dataset
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)

# Predict a new iris flower
X_new = np.array([[8, 3, 5, 2]])
print("X_new.shape: {}".format(X_new.shape), end='\n\n')
prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction[0]))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction][0]), end='\n\n')

# Predict the test dataset
y_pred = knn.predict(X_test)
print("Model accuracy score: {}".format(knn.score(X_test, Y_test)))

# Plot the test data, according to the predictions made
X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, '')

# Plot KNN decision boundaries

h = .02  # step size in the mesh
X = iris_dataset.data[:, :2]
y = iris_dataset.target

clf = KNeighborsClassifier()
clf.fit(X, y)

# Create color maps
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks([])
plt.yticks([])
plt.title("KNN decision boundaries")
plt.show()
