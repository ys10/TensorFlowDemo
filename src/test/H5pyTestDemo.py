import numpy, h5py
import os

# Current path.
path = os.path.abspath('.')
print(path)
# Read HDF5 file as data set.
f = h5py.File('../../tmp/data/train-timit.hdf5', 'r')
print(f.keys())
# Tensors as input data.
x = f['source/fbcg1_si982_5']
print(x.shape)
print(x.dtype)
# Label as expected classification result.
y = f['target/fbcg1_si982_5']
print(y.shape)
print(y.dtype)
