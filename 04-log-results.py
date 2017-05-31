import tensorflow as tf

a = tf.constant(2, tf.int16, name='a')
b = tf.constant(3, tf.int16, name='b')

with tf.name_scope('add_function') as scope:
    add = tf.add(a, b)
    tf.summary.scalar("addition", add)


# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/tmp/tensorflow/pom', graph=tf.get_default_graph())
    result, summary_str = sess.run([add, merged_summary_op])
    summary_writer.add_summary(summary_str, result)
    print(result)