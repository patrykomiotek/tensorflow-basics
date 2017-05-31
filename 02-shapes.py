import tensorflow as tf

a = tf.constant(2, tf.int16)
print(a)

b = tf.constant([2, 3], tf.int16)
print(b)

c = tf.constant([2, 3, 4, 5], tf.int16, shape=[2, 2])
print(c)

d = tf.constant([2, 3, 4, 5, 6], tf.float16, shape=[3, 2])
print(d)

e = tf.constant([2, 3, 4, 5, 6], tf.float16, shape=[2, 3])
print(e)

with tf.Session() as sess:
    aR, bR, cR, dR, eR = sess.run([a, b, c, d, e])
    print(aR)
    print(bR)
    print(cR)
    print(dR)
    print(eR)

with tf.Session() as sess:
    mult = tf.matmul(d, e)
    print(mult)
    result = sess.run(mult)
    print(result)
