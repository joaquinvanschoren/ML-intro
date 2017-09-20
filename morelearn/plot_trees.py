import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats, display, HTML
from .plot_helpers import discrete_scatter
from .plot_classifiers import plot_classifiers
from .plot_2d_separator import plot_2d_classification, plot_2d_separator
from .plot_interactive_tree import plot_tree_progressive, plot_tree_partition
from .tools import heatmap
from .datasets import make_forge, make_blobs
from sklearn.datasets import load_digits, load_iris, load_breast_cancer, make_moons
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz

def plot_tree_extrapolate(data):
    # Use historical data to forecast prices after the year 2000
    data_train = data[data.date < 2000]
    data_test = data[data.date >= 2000]

    # predict prices based on date:
    X_train = data_train.date[:, np.newaxis]
    # we use a log-transform to get a simpler relationship of data to target
    y_train = np.log(data_train.price)

    tree = DecisionTreeRegressor().fit(X_train, y_train)
    linear_reg = LinearRegression().fit(X_train, y_train)

    # predict on all data
    X_all = data.date[:, np.newaxis]

    pred_tree = tree.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    # undo log-transform
    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)

    plt.rcParams['lines.linewidth'] = 2
    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(data.date, price_tree, label="Tree prediction")
    plt.semilogy(data.date, price_lr, label="Linear prediction")
    plt.legend();

def plot_random_forest():
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        random_state=42)

    forest = RandomForestClassifier(n_estimators=5, random_state=2)
    forest.fit(X_train, y_train)

    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        plot_tree_partition(X_train, y_train, tree, ax=ax)

    plot_2d_separator(forest, X_train, fill=True, ax=axes[-1, -1],
                                    alpha=.4)
    axes[-1, -1].set_title("Random Forest")
    discrete_scatter(X_train[:, 0], X_train[:, 1], y_train);

def plot_decision_tree_regression(regr_1, regr_2,depth1, depth2):
    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure(figsize=(8,6))
    plt.scatter(X, y, c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth="+str(depth1), linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth="+str(depth2), linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

def plot_regression_tree(depth1, depth2):
    regr_1 = DecisionTreeRegressor(max_depth=depth1)
    regr_2 = DecisionTreeRegressor(max_depth=depth2)

    plot_decision_tree_regression(regr_1,regr_2,depth1, depth2)

def plot_tree(dataset, class_names):
    # Build tree
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, stratify=dataset.target, random_state=42)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)

    # Creates a .dot file
    export_graphviz(tree, out_file="tree.dot", class_names=class_names,
                    feature_names=dataset.feature_names, impurity=False, filled=True)
    # Open and display


    import graphviz
    with open("tree.dot") as f:
        dot_graph = f.read()
    display(graphviz.Source(dot_graph))

def plot_heuristics():
    def gini(p):
       return (p)*(1 - (p)) + (1 - p)*(1 - (1-p))

    def entropy(p):
       return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

    def classification_error(p):
       return 1 - np.max([p, 1 - p])

    x = np.arange(0.0, 1.0, 0.01)
    ent = [entropy(p) if p != 0 else None for p in x]
    scaled_ent = [e*0.5 if e else None for e in ent]
    c_err = [classification_error(i) for i in x]

    fig = plt.figure()
    ax = plt.subplot(111)

    for j, lab, ls, c, in zip(
          [ent, scaled_ent, gini(x), c_err],
          ['Entropy', 'Entropy (scaled)', 'Gini Impurity', 'Misclassification Error'],
          ['-', '-', '--', '-.'],
          ['lightgray', 'red', 'green', 'blue']):
       line = ax.plot(x, j, label=lab, linestyle=ls, lw=1, color=c)

    ax.legend(loc='upper left', bbox_to_anchor=(0.01, 0.85),
             ncol=1, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')

    plt.ylim([0, 1.1])
    plt.xlabel('p(j=1)')
    plt.ylabel('Impurity Index')
    plt.show()
