import time, os, math
from PIL import Image
import numpy as np
import h5py
import configparser
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
from src.lstm.utils import *

# Import configuration by config parser.
cp = configparser.ConfigParser()
cp.read('../../conf/ctc/show.ini')

# Config the logger.
# Output into log file.
log_file_name = cp.get('log', 'log_dir') + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+'.log'
if not os.path.exists(log_file_name):
    f = open(log_file_name, 'w')
    f.close()
logging.basicConfig(level = logging.DEBUG,
                format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=log_file_name,
                filemode='w')
# Output to the console.
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# File storing group name.
group_file_name = cp.get('data', 'group_file_name')
# Read group data.
groups = open(group_file_name, 'r');
# Name of HDF5 file as test data set.
test_data_file_name = cp.get('data', 'test_data_file_name')
# Read training data file.
test_data = h5py.File(test_data_file_name, 'r')
# Name of HDF5 file as result data set.
result_data_file_name = cp.get('data', 'result_data_file_name')
# Read result data file.
result_data = h5py.File(result_data_file_name, 'r')

# Parameters
batch_size = 1
display_batch = 1
training_iters = 1
n_steps = 777 # time steps
# n_classes = 50

batch_x = []
batch_y = []
batch_seq_len = []
batch_lstm_outputs = []
batch_linear_outputs = []
batch_decode = []
batch_beam_decode = []
batch_greedy_decode = []
# Traverse all trunks of a batch.
start = time.time()
trunk_name = "faem0_si1392"
# TODO
logging.debug("trunk_name: " + trunk_name)
# print("trunk_name: " + trunk_name)
# print("length:"+ len(trunk_name))
# Get trunk data by trunk name without line break character.
# sentence_x is a tensor of shape (n_steps, n_inputs)
sentence_x = test_data['source/' + trunk_name]
# sentence_y is a tensor of shape (None)
sentence_y = test_data['target/' + trunk_name]
# sentence_len is a tensor of shape (None)
sentence_len = test_data['size/' + trunk_name].value
# Add current trunk into the batch.
batch_x.append(sentence_x)
batch_y.append(sentence_y)
#
linear_outputs = result_data['iter0/linear_output/' + trunk_name].value[0]
lstm_outputs = result_data['iter0/lstm_output/' + trunk_name].value[0]
decode = result_data['iter0/decode/' + trunk_name].value[0]
# TODO
# greedy_decode = result_data['iter0/greedy_decode/' + trunk_name].value[0]
# beam_decode = result_data['iter0/beam_decode/' + trunk_name].value[0]
logging.debug("linear_outputs shape: " + str(linear_outputs.shape))
#
batch_decode.append(decode)
# TODO
batch_lstm_outputs.append(lstm_outputs)
batch_linear_outputs.append(linear_outputs)
# batch_beam_decode.append(beam_decode)
# batch_greedy_decode.append(greedy_decode)
# Process label & input.
batch_x, batch_seq_len = pad_sequences(batch_x, maxlen=n_steps)
batch_y = sparse_tuple_from(batch_y)
# Print the data info.
logging.debug("Trunk name: " + trunk_name)
logging.debug("linear_outputs: " + str(batch_linear_outputs))
# logging.debug("lstm_outputs: " + str(batch_lstm_outputs))
logging.debug("label: " + str(batch_y[1]))
logging.debug("label length: " + str(len(batch_y[1])))
logging.debug("decode: " + str(batch_decode[0]))
logging.debug("decode length: " + str(len(batch_decode[0])))
# TODO
# logging.debug("beam decode: " + str(batch_beam_decode[0]))
# logging.debug("beam decode length: " + str(len(batch_beam_decode[0])))
# logging.debug("greedy decode: " + str(batch_greedy_decode[0]))
# logging.debug("greedy decode length: " + str(len(batch_greedy_decode[0])))

max_prob_phome = []
for i in range(0, n_steps, 1):
    l = list(batch_linear_outputs[0][i])
    while l.index(max(l)) == 49:
        l.remove(max(l))
    index = l.index(max(l))
    # max_prob_phome.append(index)
    if not max_prob_phome:
        max_prob_phome.append(index)
    elif max_prob_phome[len(max_prob_phome) - 1] != index:
        max_prob_phome.append(index)

logging.debug("max prob phome in linear outputs: " +str(max_prob_phome))
logging.debug("max prob phome in linear outputs length: " +str(len(max_prob_phome)))


# show the result by picture.
linear_outputs_array = transpose_dimension(batch_linear_outputs[0])

#
all_phome = [i for i in range(0, 50)]
all_linear_outputs_array = transpose_dimension(linear_outputs_array)
print("decode_linear_outputs_array length:　" + str(len(all_linear_outputs_array)))
# draw picture
plt.figure(1)
ax = plt.subplot(111)
x = np.linspace(0, n_steps, n_steps)
plt.figure(1)
line = ax.plot(x, all_linear_outputs_array)
ax.legend(line, all_phome)
plt.show()
#
decode_phome = list(set(batch_beam_decode[0]))
# print("decode phome:　" + str(decode_phome))
decode_phome.sort()
# print("decode phome:　" + str(decode_phome))
decode_linear_outputs_array = [linear_outputs_array[i][:] for i in decode_phome]
decode_linear_outputs_array = transpose_dimension(decode_linear_outputs_array)
print("decode_linear_outputs_array length:　" + str(len(decode_linear_outputs_array)))
# draw picture
plt.figure(2)
ax = plt.subplot(111)
x = np.linspace(0, n_steps, n_steps)
plt.figure(2)
line = ax.plot(x, decode_linear_outputs_array)
ax.legend(line, decode_phome)
plt.show()

# show the result by picture.
# linear_outputs_array = transpose_dimension(batch_linear_outputs[0])
label_phome = list(set(batch_y[1]))
# print("label phome:　" + str(label_phome))
label_phome.sort()
# print("label phome:　" + str(label_phome))
label_linear_outputs_array = [linear_outputs_array[i][:] for i in label_phome]
label_linear_outputs_array = transpose_dimension(label_linear_outputs_array)
print("label_linear_outputs_array length:　" + str(len(label_linear_outputs_array)))
# draw picture
plt.figure(3)
ax = plt.subplot(111)
x = np.linspace(0, n_steps, n_steps)
plt.figure(3)
line = ax.plot(x, label_linear_outputs_array)
ax.legend(line, label_phome)
plt.show()
# Time
logging.debug("time: {:.3f}".format(time.time() - start))