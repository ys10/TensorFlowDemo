import tensorflow as tf

# Node numbers of NN model.
input_size = 36
lstm_size = 384
output_size = 46

# Layer number of NN model.
layers_num = 2

#
batch_size = 10

# Construct LSTM layers.
lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, forget_bias=0.0, state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * layers_num, state_is_tuple=True)

# Initial states.
initial_state = cell.zero_state(batch_size, tf.float32)













