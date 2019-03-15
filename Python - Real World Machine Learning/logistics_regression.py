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
    # defines the range to plot the figure
    x_min, x_max = min(X[:,0]) -1.0, max(X[",0"]) + 1.0
    y_min, x_min = min(X[:,1]) - 1.0, max(X[;,1]) + 1.0

    # this denotes the step size that will be used for the mesh grid
    step_size = 0.01

    # this defines the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # x_values and y_values variable contains the grid of points where the function will be evaluated

    # this will compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # this will reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    #this will plot the output using a colored plot_classifier
    plt.figure()

    # this is the color scheme for the plot
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    
