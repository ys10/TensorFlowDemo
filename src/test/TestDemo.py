import tensorflow as tf

# Ensure you have install TensorFlow correctly.
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
