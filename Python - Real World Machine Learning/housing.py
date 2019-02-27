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


def plot_feature_importances(feature_importances, title, feature_names):


feature_importances = 100.0 * (feature_importances / max(feature_importances))
index_sorted = np.flipud(np.argsort(feature_importances))
pos = np.arange(index_sorted.shape[0]) + 0.5


plot_feature_importances(dt_regressor.feature_importances_, 'Decision Tree Regressor', housing_data.feature_names)
plot_feature_importances(ab_regressor.feature_importances_, 'Adaboost regressor', housing_data.feature_names)

plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title(title)
plt.show()
