import csv
from sklearn.ensemble import RandomForestRegressor
from housing import plot_feature_importances


def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])

        feature_names = np.array(X[0])

        return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

    X, y, feature_names = load.dataset(sys.argv[1])
    X, y = shuffle(X, y, random_state=7)

    num_training = int(0.9 * len(X))

    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=1)
    rf_regressor.fit(X_train, y_train)

    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    print("\n### Random Forest Regressor performance ###")

    print("Mean squared error = ", round(mse, 2))
    print("Explained variance score = ", round(evs, 2))

    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest Regressor', feature_names)
