from numpy.core.fromnumeric import shape
import tensorflow.compat.v1 as tf

from model import Model

import numpy as np

tf.disable_eager_execution()
IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')

        # softmax_regression : input_layer(784, 1) -> output_layer(62, 1)
        logits = tf.layers.dense(inputs=features, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # TODO: Confirm that opt initialized once is ok?

        # stochastic-admm optimizer
        model = tf.trainable_variables()

        grad = tf.gradients(loss, model)
        lr = tf.constant(self.lr, dtype=tf.float32)
        
        update0 = tf.assign(model[0], model[0] - lr * grad[0])
        update1 = tf.assign(model[1], model[1] - lr * grad[1])
        train_op = tf.group(update0, update1)
        
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
