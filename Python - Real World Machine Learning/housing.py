import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

housing_data = datasets.load_boston()

x, y = shuffle(housing_data.data, housing_data.target, random_state=7)

num_training = int(0.8 * len(x))
x_train, y_train = x[:num_training], y[:num_training]
x_test, y_test = x[num_training:], y[num_training:]

dt_regressor = DecisionTreeRegressor(max_depth=4)
dt_regressor.fit(x_train, y_train)

ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
ab_regressor.fit(x_train, y_train)

y_pred_dt = dt_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred_dt)
evs = explained_variance_score(y_test, y_pred_dt)

print("\n### Decision Tree performance")
print("Mean squared error=", round(mse, 2))
print("Explained variance score=", round(evs, 2))

y_pred_ab = ab_regressor.predict(x_test)
mse = mean_squared_error(y_test, y_pred_ab)
evs = explained_variance_score(y_test, y_pred_ab)

print("\n### AdaBoost performance")
print("Mean squared error=", round(mse, 2))
print("Explained variance score=", round(evs, 2))
