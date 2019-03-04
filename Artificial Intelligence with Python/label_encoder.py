import numpy as np
from sklearn import preprocessing

# defining sample/dummy data
input_label = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# create label encoder to fit the labels
# defining encoder
encoder = preprocessing.LabelEncoder()
encoder.fit(input_label)

# print function for mapping
print("\nLabel Mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# encoding a set of labels using the encoder
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
# print function for labels and encoded values
print("\nLabels = ", test_labels)
print("Encoded Values = ", list(encoded_values))

# decoding sets of values using the encoder
# defining encoded values
encoded_values = [3, 0, 4, 1]
# defining decoded list
decoded_list = encoder.inverse_transform(encoded_values)

# print function for encoded values and decoded values
print("\nEncoded Values =", encoded_values)
print("\nDecoded Values =", list(decoded_list))
