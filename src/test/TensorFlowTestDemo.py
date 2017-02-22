import tensorflow as tf

# Ensure you have install TensorFlow correctly.
hello = tf.constant('Hello, TensorFlow!')
states = tf.zeros(shape=[1,2])
variable = tf.Variable(states)
print(states)
# Start a TensorFlow session.
sess = tf.Session()
print(sess.run(hello))
print(sess.run(states))
# Initialize variable.
sess.run(tf.global_variables_initializer())
print(sess.run(variable))
