import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
X_train , X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                     stratify=cancer.target, random_state=0)

tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
import graphviz
with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

