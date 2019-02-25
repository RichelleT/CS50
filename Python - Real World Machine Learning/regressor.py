import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

filename = sys.argv[1]
x = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        x.append(xt)
        y.append(yt)
# loaded input data x and y, where x refers to data and y refers to labels

# we need a way to validate and check if the model is performing at a satisfactory level
# to do so, we need to seperate the data into 2 groups
# The training dataset will be used to build the model,
# and the testing dataset will be used to see how this trained model performs on unknown data.

# split data into testing and training datasets
num_training = int(0.8 * len(x))
num_test = len(x) - num_training

# training data
x_train = np.array(x[:num_training]).reshape((num_training, 1))
y_train = np.array(y[:num_training])

# test data
x_test = np.array(x[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])

# create linear regression object
linear_regressor = linear_model.LinearRegression()

# train the model using the training sets
linear_regressor.fit(x_train, y_train)

# the fit method takes the input data and trains the model

y_train_pred = linear_regressor.predict(x_train)
plt.figure()
plt.scatter(x_train, y_train_pred, color='black', linewidth=4)
plt.title('Training Data')
plt.show()

y_test_pred = linear_regressor.predict(x_test)
plt.scatter(x_test, y_test, color='green')
plt.plot(x_test, y_test_pred, color='black', linewidth=4)
plt.title('Test Data')
plt.show()
