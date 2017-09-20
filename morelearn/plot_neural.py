import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
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
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import export_graphviz

def plot_neural_predictions(activation='tanh', hidden_layer_sizes=[10, 10]):
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=42)

    mlp = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=hidden_layer_sizes, activation=activation).fit(X_train, y_train)
    plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")

def contoursteps(x1, x2, zs, steps=None):
    fig = plt.figure(figsize=(6,6))
    cp = plt.contour(x1, x2, zs, 10)
    plt.clabel(cp, inline=1, fontsize=10)
    if steps is not None:
        steps = np.matrix(steps)
        plt.plot(steps[:,0], steps[:,1], '-o')
    fig.show()

def f(x, A, b, c):
    return float(0.5 * x.T * A * x - b.T * x + c)

def bowl(A, b, c):
    fig = plt.figure(figsize=(10,8))
    qf = fig.gca(projection='3d')
    size = 20
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)
    qf.plot_surface(x1, x2, zs, rstride=1, cstride=1, linewidth=0)
    fig.show()
    return x1, x2, zs

def plot_gradient_descent_surface():
    # Some data
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])  # we will use the convention that a vector is a column vector
    c = 0.0

    x1, x2, zs = bowl(A, b, c)

def plot_gradient_descent(alpha=0.12):
    # Some data
    A = np.matrix([[3.0, 2.0], [2.0, 6.0]])
    b = np.matrix([[2.0], [-8.0]])  # we will use the convention that a vector is a column vector
    c = 0.0

    # Response surface
    size = 20
    x1 = list(np.linspace(-6, 6, size))
    x2 = list(np.linspace(-6, 6, size))
    x1, x2 = np.meshgrid(x1, x2)
    zs = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = np.matrix([[x1[i,j]], [x2[i,j]]])
            zs[i,j] = f(x, A, b, c)

    x = np.matrix([[-2.0],[-2.0]])
    steps = [(-2.0, -2.0)]
    i = 0
    imax = 10000
    eps = 0.01
    alpha = alpha  # 0.12 play with this value to see how it affects the optimization process, try 0.03, 0.2, 0.27
    r = b - A * x
    delta = r.T * r
    delta0 = delta
    while i < imax and delta > eps**2 * delta0:
        x = x + alpha * r
        steps.append((x[0,0], x[1,0]))  # store steps for future drawing
        r = b - A * x
        delta = r.T * r
        i += 1

    contoursteps(x1, x2, zs, steps)
