import sklearn.metrics as sm

print("Mean absolute error=", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error=", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error=", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score=", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score=", round(sm.r2_score(y_test, y_test_pred), 2))
