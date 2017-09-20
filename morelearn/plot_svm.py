import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from .plot_helpers import cm2, cm3, discrete_scatter
from IPython.display import set_matplotlib_formats, display, HTML
from .plot_helpers import discrete_scatter
from .plot_classifiers import plot_classifiers
from .plot_2d_separator import plot_2d_classification, plot_2d_separator
from .plot_interactive_tree import plot_tree_progressive, plot_tree_partition
from .tools import heatmap
from .datasets import make_forge, make_blobs
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, make_moons
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz

def plot_kernelize():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    linear_svm = LinearSVC().fit(X, y)

    plot_2d_separator(linear_svm, X)
    discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1");

def plot_kernelize2():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2

    # add the squared first feature
    X_new = np.hstack([X, X[:, 1:] ** 2])


    from mpl_toolkits.mplot3d import Axes3D, axes3d
    figure = plt.figure()
    # visualize in 3D
    ax = Axes3D(figure, elev=-152, azim=-26)
    # plot first all the points with y==0, then all with y == 1
    mask = y == 0
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
               cmap=cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
               cmap=cm2, s=60)
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 ** 2");

def plot_kernelize3():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    X_new = np.hstack([X, X[:, 1:] ** 2])
    mask = y == 0

    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # show linear decision boundary
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
               cmap=cm2, s=60)
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
               cmap=cm2, s=60)

    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 ** 2")

def plot_kernelize4():
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    X_new = np.hstack([X, X[:, 1:] ** 2])
    mask = y == 0

    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # show decision boundary
    figure = plt.figure()
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = YY ** 2
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
                 cmap=cm2, alpha=0.5)
    discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1");
