import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

graph = tf.Graph()
sess = tf.Session(graph=graph)

const = np.arange(10)
with graph.as_default():
    y = tf.Variable(np.ones(10) * 5, dtype=tf.float32)

    update = tf.assign(y, y + const)

    init = tf.global_variables_initializer()

with sess.as_default():
    sess.run(init)
    sess.run(update)
    print(sess.run(y))
