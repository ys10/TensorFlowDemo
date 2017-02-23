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
groups = open(group_file_name, 'r');
# Read training data file.
training_data = h5py.File(training_data_file_name, 'r')

# Get a group.
print(training_data.keys())
# Tensors as input data.
X = training_data['source/fbcg1_si982_5']
print(X.shape)
print(X.dtype)
# Label as expected classification result.
Y = training_data['target/fbcg1_si982_5']
print(Y.shape)
print(Y.dtype)

# Traverse all groups
while 1:
    lines = groups.readlines(2);
    if not lines:
        break
    for line in lines:
        # Get training data by group name without line break.
        X = training_data['source/'+line.strip('\n')]
        Y = training_data['target/'+line.strip('\n')]
        for i in range(0, X.shape[0], 1):
            print(X[i])
        for i in range(0, Y.shape[0], 1):
            print(Y[i])
    break
