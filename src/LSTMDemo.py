import tensorflow as tf

input_size = 36
lstm_size = 384
output_size = 46

lstm1 = tf.rnn.rnn_cell.BasicLSTMCell(lstm_size)
lstm2 = tf.rnn.rnn_cell.BasicLSTMCell(lstm_size)








