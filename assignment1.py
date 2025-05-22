import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Only the first two features (sepal length and width)
y = iris.target

# Set the mesh step size
h = 0.02

# Create color maps
cmap_light = plt.cm.Pastel1
cmap_bold = plt.cm.Set1

for n_neighbors in [1, 5]:
    # Create an instance of k-NN classifier and fit the data
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)

    # Create a mesh of points to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict the class for each point in the mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and training points
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"3-Class classification (k = {n_neighbors}, weights = 'uniform')")

plt.show()
