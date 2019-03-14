import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4,
2.2]])
y = ([0, 0, 0, 1, 1, 1, 2, 2, 2])

classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

classifier.fit(X, y)

plot_classifier(classifier, X, y)

def plot_classifier(classifier, X, y):
    #defines the range to plot the figure
    x_min, x_max = min(X[:,0]) -1.0, max(X[",0"]) + 1.0
    
