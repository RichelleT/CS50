import numpy as np
from sklearn import preprocessing

data = np.array([[3, -1.5, 2, -5.4], [0, 4, -0.3, 2.1], [1, 3.3, -1.9, -4.3]])

# Mean Removal
data_standardized = preprocessing.scale(data)
print("\nMean = ", data_standardized.mean(axis=0))
print("Std deviation =", data_standardized.std(axis=0))

# Scaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data =", data_scaled)

# Normalization
data_normalized = preprocessing.normalize(data, norm='l1')
print("\nl1 normalized data =", data_normalized)

# Binarization
data_binarized = preprocessing.binarizer(threshold=1.4).transform(data)
print("\nBinarized data=", data_binarized)

# One Hot Encoding
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print("\nEncoded Vector=", encoded_vector)
