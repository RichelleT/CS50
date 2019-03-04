import numpy as np
from sklearn import preprocessing

# defining some sample/dummy data
input_data = np.array([[5.1, -2.0, 3.3], [-1.2, 7.8, -6.1], [3.9, -9.9, 2.1], [7.3, -9.9, -4.5]])

# binarizing data
data_binarized = preprocessing.Binarizer(threshold=2.1).transform(input_data)
print("\nBinarized Data:\n", data_binarized)
