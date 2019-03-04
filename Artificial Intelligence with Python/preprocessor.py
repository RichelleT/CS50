import numpy as np
from sklearn import preprocessing

# defining some sample/dummy data
input_data = np.array([[5.1, -2.0, 3.3], [-1.2, 7.8, -6.1], [3.9, -9.9, 2.1], [7.3, -9.9, -4.5]])

# binarizing data
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized Data:\n", data_binarized)

# print function for mean
print("\nBEFORE:")
print("Mean =", input_data.mean(axis=0))

# print function for standard deviation
print("Standard Deviation = ", input_data.std(axis=0))

# remove mean
# defining data_scaler
data_scaled = preprocessing.scale(input_data)

print("\nAFTER:")
print("Mean =", data_scaled.mean(axis=0))
print("Standard Deviation =", data_scaled.std(axis=0))

# min and max scaling
data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)

# print function for min max scaling
print("\nMinimum and Maximum scaled data =\n", data_scaled_minmax)

# normalizing data
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')

# print function for normalizing data
print("\nL1 Normalized data:\n", data_normalized_l1)
print("\nL2 Normalized Data:\n", data_normalized_l2)
