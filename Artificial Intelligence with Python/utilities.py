import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y):
    # defining the min and max value for x and y
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # defining the step size to use in graphing or plotting the mesh grid
    mesh_step_size = 0.01
