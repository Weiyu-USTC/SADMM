import math
from re import S
import numpy as np
import tensorflow.compat.v1 as tf
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY,  graph, byzantine, regular

class Server:
    
    def __init__(self, client_model, lr):
        self.client_model = client_model
        self.lr = lr
        self.selected_clients = []
        self.updates = []
        with graph.as_default():
           self.model = [np.zeros((784, 62)), np.zeros(62)] 

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.
        
        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the server can select.
            num_clients: Number of clients to select; default 20
        Return:
            list of (num_train_samples, num_test_samples)
        """
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

        return [(c.num_train_samples, c.num_test_samples) for c in self.selected_clients]

    def train_model(self, num_epochs=1, batch_size=10, minibatch=None, clients=None):
        """Trains self.model on given clients.
        
        Trains model on self.selected_clients if clients=None;
        each client's data is trained with the given number of epochs
        and batches.

        Args:
            clients: list of Client objects.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        
        num_clients = len(clients)
        grad_list = [np.zeros((num_clients, 784, 62)), np.zeros((num_clients, 62))]
        
        for i in range(num_clients):
            if i in byzantine:
                c = clients[regular[0]]
            elif i in regular:
                c = clients[i]
            c.model.set_params(self.model)
            grad = c.train(num_epochs, batch_size, minibatch)
            grad_list[0][i] = grad[0]
            grad_list[1][i] = grad[1]

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = 0

        self.update_model(grad_list)
        return sys_metrics

    # Do not consider the unbalance data (fit for the Byzantine case)
    def update_model(self, grad_list):
        for i in range(len(self.model)):
            guess = np.mean(grad_list[i], axis=0)
            for _ in range(80):
                res1 = np.zeros_like(guess)
                res2 = 0
                for j in range(grad_list[i].shape[0]):
                    dist = np.linalg.norm(grad_list[i][j] - guess, 2)
                    if dist == 0: dist = 1
                    res1 += grad_list[i][j] / dist
                    res2 += 1 / dist
                guess_next = res1 / res2
                guess_move = np.linalg.norm(guess - guess_next, 2)
                guess = guess_next

                if guess_move < 1e-5:
                    break
            
            self.model[i] = self.model[i] - self.lr * guess


    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, groups, num_samples

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        model_sess =  self.client_model.sess
        tf.disable_v2_behavior()
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()
