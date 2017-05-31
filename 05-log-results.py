import tensorflow as tf

a = tf.constant(2, tf.int16, name='a')
b = tf.constant(3, tf.int16, name='b')
c = tf.constant(4, tf.int16, name='c')

with tf.name_scope('add_function') as scope:
    add = tf.add(a, b)
    tf.summary.scalar("addition", add)

with tf.name_scope('mult_function') as scope:
    mul = tf.multiply(add, c)
    tf.summary.scalar("multiplication", mul)


# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/tmp/tensorflow/pom', graph=tf.get_default_graph())
    result, mulResult, summary_str = sess.run([add, mul, merged_summary_op])
    summary_writer.add_summary(summary_str, mulResult)
    print(result)