import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!', tf.string)

sess = tf.Session()

print(hello)
print(sess.run(hello))