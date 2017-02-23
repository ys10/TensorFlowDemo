import numpy, h5py
import os

# File storing group name.
group_file_name = '../../tmp/data/train_speechorder_timit.txt'
# HDF5 file as training data set.
training_data_file_name = '../../tmp/data/train-timit.hdf5'
# Current path.
path = os.path.abspath('.')
print(path)
# Read group data.
groups = open(group_file_name);
# Read training data file.
training_data = h5py.File(training_data_file_name, 'r')

'''
# Traverse all groups
while 1:
    lines = training_data.readlines(10000);
    if not lines:
        break
    for line in lines:
        x = training_data['source/'+line]
        y = training_data['target/'+line]

'''

print(training_data.keys())
# Tensors as input data.
X = training_data['source/fbcg1_si982_5']
print(X.shape)
print(X.dtype)
for i in range(0, X.shape[0], 1):
    print(X[i])
# Label as expected classification result.
Y = training_data['target/fbcg1_si982_5']
print(Y.shape)
print(Y.dtype)
print(Y)
