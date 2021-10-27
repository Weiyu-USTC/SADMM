from os import name
from numpy.core.fromnumeric import shape
import tensorflow.compat.v1 as tf

from model import Model
from baseline_constants import lamda, beta, graph
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
        eta_cur_0 = tf.placeholder(tf.float32, shape=[784,62], name='eta_cur_0')
        eta_cur_1 = tf.placeholder(tf.float32, shape=[62,], name='eta_cur_1')
        eta_pre_0 = tf.placeholder(tf.float32, shape=[784,62], name='eta_pre_0')
        eta_pre_1 = tf.placeholder(tf.float32, shape=[62,], name='eta_pre_1')
        lr_train = tf.placeholder(tf.float32, shape=[], name='lr_train')

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
        self.model_local = model.copy()

        grad = tf.gradients(loss, model)
        
        update0 = tf.assign(model[0], model[0] - lr_train * (grad[0] + 2 * eta_cur_0 - eta_pre_0))
        update1 = tf.assign(model[1], model[1] - lr_train * (grad[1] + 2 * eta_cur_1 - eta_pre_1))
        train_op = tf.group(update0, update1)
        
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, eta_cur_0, eta_cur_1, eta_pre_0,eta_pre_1, lr_train, train_op, eval_metric_ops, loss
    
    def update_eta(self, model_server):
        model_local = self.get_params()
        self.eta_pre = self.eta_cur.copy()
        for i in range(len(self.eta_cur)):
            self.eta_cur[i] = self.eta_cur[i] +  beta / 2 * (model_local[i] - model_server[i])
            for j in np.nditer(self.eta_cur[i], op_flags=['readwrite']):
                if j > lamda: j = lamda
                elif j < -lamda: j = -lamda
    
    def get_eta(self):
        return self.eta_cur, self.eta_pre
    
    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)
