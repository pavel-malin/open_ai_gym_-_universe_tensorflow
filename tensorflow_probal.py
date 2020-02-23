import tensorflow as tf


hello = tf.constant('Hello world')
sess = tf.Session()
print(sess.run(hello))


weights = tf.Variable(tf.random_normal([3, 2], stddev=0.1), name='weights')
print(weights)


x = tf.constant(13)
print(x)


x = tf.placeholder("float", shape=None)
print(x)


A = tf.multiply(8, 5)
B = tf.multiply(A, 1)
print(A, B)


A = tf.multiply(8, 5)
B = tf.multiply(4, 3)
print(A, B)
