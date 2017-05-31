import tensorflow as tf

# Model parameters
W = tf.Variable([.3], tf.float32, name='W')
b = tf.Variable([-.3], tf.float32, name='b')

# Model input and output
x = tf.placeholder(tf.float32, name='x-input')
linear_model = W * x + b
y = tf.placeholder(tf.float32, name='y-input')

# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
tf.summary.scalar("loss", loss)

# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

# Merge all summaries into a single operator
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('/tmp/tensorflow/pom-train', graph=tf.get_default_graph())

for step in range(1000):
  _, summary = sess.run([train, summary_op], {x: x_train, y: y_train})
  summary_writer.add_summary(summary, step)

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))
