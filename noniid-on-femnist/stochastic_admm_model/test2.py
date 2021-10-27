from numpy import dtype
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
# set value of tf.constant() 
# y = tf.constant(1, dtype=tf.int32)
# sess = tf.Session()
# y = y + 1
# with sess.as_default():
#     print(y.eval())

# set value of tf.variable()
y = tf.Variable(1, dtype=tf.int32)
init = tf.global_variables_initializer()
sess = tf.Session()
y = y + 1

with sess.as_default():
    sess.run(init)
    # sess.run(update)
    print(y.eval())