import tensorflow as tf

# MSE function.
def mse(logits, outputs):
    mse = tf.reduce_mean(tf.pow(tf.sub(logits, outputs), 2.0))
    return mse