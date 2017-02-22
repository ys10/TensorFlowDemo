import tensorflow as tf

# Node numbers of NN model.
input_size = 36
lstm_size = 384
output_size = 46

# Layer number of NN model.
layers_num = 2

# Construct LSTM layers.
lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=0.0, state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * layers_num, state_is_tuple=True)

# Data config.
batch_size = 10

# Initial states of LSTM.
initial_state = cell.zero_state(batch_size, tf.float32)

# Initialize the configuration of data set.
# TODO

# Train the Network with data.
# Initialize the loss.
loss = 0;
# Process training data by Trunk.
# TODO
















