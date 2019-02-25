import sys
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import _pickle as pickle
from sklearn.preprocessing import PolynomialFeatures

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

# new / regressor accuracy

output_model_file = 'saved_model.pkl'
with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)
with open(output_model_file, 'r') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(x_test)
print("\nNew mean absolute error=", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
# the alpha parameter controls the complexity
# As alpha gets closer to 0, the ridge regressor tends to become more like a linear regressor with ordinary least squares. So, if you want to make it robust against outliers, you need to as- sign a higher value to alpha. We considered a value of 0.01, which is moderate.
ridge_regressor.fit(x_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(x_test)
print("Mean absolute error=", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print("Mean squared error=", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print("Median absolute error=", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print("Explained variance score=", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print("R2 score=", round(sm.r2_score(y_test, y_test_pred_ridge), 2))

polynomial = PolynomialFeatures(degree=3)
x_train_transformed = polynomial.fit_transform(x_train)

datapoint = [0.39, 2.78, 7.11]
poly_datapoint = polynomial.fit_transform(datapoint)
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(x_train_transformed, y_train)
print("\nLinear regression=", linear_regressor.predict(datapoint)[0])
