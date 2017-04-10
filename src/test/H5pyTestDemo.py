import h5py
import os
import time
from math import ceil
import logging
from numpy import array
import tensorflow as tf
from src.lstm.utils import tensor_to_array

# File storing group name.
group_file_name = "../../tmp/data/ctc/train_speechorder_half-swbd_ctc.txt"
# Read group data.
groups = open(group_file_name, 'r');
# # Name of HDF5 file as training data set.
training_data_file_name = "../../tmp/data/ctc/train_half-swbd_ctc.hdf5"
# Read training data file.
training_data = h5py.File(training_data_file_name, 'r')

# output = training_data['fbcg1_si982_5target/fbcg1_si982_5']
# output = tf.reshape(output.value, [output.shape[1], output.shape[2]])
# print(output.shape)
# print(output.dtype)


# Get a group.
# print("keys: "+ str(training_data.keys()))
# # Tensors as input data.
# X = training_data['iter0/lstm_output/sw02466-B_026026-026936']
# print(X.shape)
# print(X.dtype)
# Label as expected classification result.
Y = training_data['source/sw02102-B_013523-015179'].value
print(Y.shape)
print(Y.dtype)
print(Y)
print(str(len(Y)))

# for i in range(0, X.shape[0], 1):
#     print(X[i])
# for i in range(0, Y.shape[0], 1):
#     print(Y[i])


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
# # Traverse all groups
# lines = groups.readlines();
# # print("len(lines):")
# # print(len(lines))
# # print(ceil(len(lines)/batch_size))
#
# for line in lines:
#     # Get training data by group name without line break.
#     # X = training_data['source/'+line.strip('\n')]
#     Y = training_data['target/'+line.strip('\n')]
#     # for i in range(0, X.shape[0], 1):
#     #     print(X[i])
#     for i in range(0, Y.shape[0], 1):
#         if Y[i].any() >=48:
#             print(line.strip('\n') + ":" +str(Y[i]))
#     # break


# logging.basicConfig(level = logging.DEBUG,
#                 format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filemode='w')
#
# # Import data set
# # Name of file storing trunk names.
# trunk_names_file_name = "../../tmp/data/ctc/train_speechorder_half-swbd_ctc.txt"
# # Name of HDF5 file as training data set.
# training_data_file_name = "../../tmp/data/ctc/train-swbd-ctc.hdf5"
# # Read trunk names.
# trunk_names_file = open(trunk_names_file_name, 'r')
# # Read training data set.
# training_data_file = h5py.File(training_data_file_name, 'r')
#
# batch_size = 16
# display_batch = 1
# training_iters = 1
#
# all_trunk_names = trunk_names_file.readlines()
# logging.debug("number of trunks:" + str(len(all_trunk_names)))
# # Calculate how many batches can the data set be divided into.
# n_batches = ceil(len(all_trunk_names) / batch_size)
# logging.debug("n_batches:" + str(n_batches))
# for iter in range(0, training_iters, 1):
#     train_cost = train_ler = 0
#     start = time.time()
#     # For each iteration.
#     logging.debug("Iter:" + str(iter))
#     # Break out of the training iteration while there is no trunk usable.
#     if not all_trunk_names:
#         break
#     # Train the RNN(LSTM) model by batch.
#     for batch in range(0, n_batches, 1):
#         # For each batch.
#         # Define two variables to store input data.
#         batch_x = []
#         batch_y = []
#         batch_seq_len = []
#         # Traverse all trunks of a batch.
#         for trunk in range(0, batch_size, 1):
#             # For each trunk in the batch.
#             # Calculate the index of current trunk in the whole data set.
#             trunk_name_index = batch * batch_size + trunk
#             logging.debug(trunk_name_index)
#             # There is a fact that if the number of all trunks
#             #   can not be divisible by batch size,
#             #   then the last batch can not get enough trunks of batch size.
#             # The above fact is equivalent to the fact
#             #   that there is at least a trunk
#             #   whose index is no less than the number of all trunks.
#             if (trunk_name_index >= len(all_trunk_names)):
#                 # So some used trunks should be add to the last batch when the "fact" happened.
#                 # Select the last trunk to be added into the last batch.
#                 trunk_name_index = len(all_trunk_names) - 1
#             # Get trunk name from all trunk names by trunk name index.
#             trunk_name = all_trunk_names[trunk_name_index]
#             logging.debug(trunk_name)