import tensorflow as tf

a = tf.multiply(2, 3)

with tf.Session() as sess:
    print(sess.run(a))
