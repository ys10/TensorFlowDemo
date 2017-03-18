import h5py
import os
from math import ceil
from numpy import array
import tensorflow as tf
from src.lstm.utils import tensor_to_array

# File storing group name.
group_file_name = '../../tmp/data/train_speechorder_timit.txt'
# HDF5 file as training data set.
training_data_file_name = '../../tmp/result/2017-03-18-22-12-22.hdf5'
# Current path.
path = os.path.abspath('.')
print(path)
# Read group data.
groups = open(group_file_name, 'r');
# Read training data file.
training_data = h5py.File(training_data_file_name, 'r')

output = training_data['iter0/lstm_output/fbcg1_si982_5']
print(output.shape)
print(output.dtype)
print(output.value)
#
#
# # Get a group.
# print("keys: "+ str(training_data.keys()))
# # Tensors as input data.
# X = training_data['source/faem0_si1392']
# print(X.shape)
# print(X.dtype)
# # Label as expected classification result.
# Y = training_data['target/faem0_si1392']
# print(Y.shape)
# print(Y.dtype)
#
# seq_len = training_data['size/fajw0_sx183']
# print(seq_len.shape)
# print(seq_len.dtype)
#
#
# batch_x = []
# batch_x.append(X)
# batch_x.append(X)
#
# print("batch_x:")
# print(batch_x)
#
# batch_x = array(batch_x)
# print("batch_x:")
# print(batch_x.shape)
#
# batch_y = []
# batch_y.append(Y)
# # batch_y.append(Y)
#
# print("batch_y:")
# print(batch_y)
#
# batch_y = array(batch_y)
# print("batch_y:")
# print(batch_y.shape)

# ##########
# # Output file name.
# outpout_data_file_name = '../../tmp/data/test.hdf5'
# if not os.path.exists(outpout_data_file_name):
#     f = open(outpout_data_file_name, 'w')
#     f.close()
# # Output files.
# outpout_data_file = h5py.File(outpout_data_file_name, 'w')
# #
# lstm_grp = outpout_data_file.create_group("lstm_output")
# linear_grp = outpout_data_file.create_group("linear_output")
# #
# states = tf.zeros(shape=[1,2])
# variable = tf.Variable(states)
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# variable = sess.run(variable)
# print(len(variable.shape))
#
# #
# array = tensor_to_array(variable)
# print(array)
# linear_grp.create_dataset("test", shape = [1,2], data = array, dtype = 'f')
# ##############


# batch_size = 32;
# Traverse all groups
# while 1:
#     lines = groups.readlines();
#     print("len(lines):")
#     print(len(lines))
#     print(ceil(len(lines)/batch_size))
#     if not lines:
#         break
#     for line in lines:
#         # Get training data by group name without line break.
#         X = training_data['source/'+line.strip('\n')]
#         Y = training_data['target/'+line.strip('\n')]
#         # for i in range(0, X.shape[0], 1):
#         #     print(X[i])
#         for i in range(0, Y.shape[0], 1):
#             print(Y[i])
#     break
