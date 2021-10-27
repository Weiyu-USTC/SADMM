import math
from re import S
import numpy as np
import tensorflow.compat.v1 as tf
from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY, beta, lamda, lr, graph, byzantine, regular

class Server:
    
    def __init__(self, client_model, lr):
        self.client_model = client_model
        self.lr = lr
        self.selected_clients = []
        self.updates = []
        self.round_numer = 0
        with graph.as_default():
           self.model = [np.zeros((784, 62)), np.zeros(62)] 

    def set_round_number(self, round_number):
        self.round_numer = round_number
    
    def set_lr_server(self):
        num_clients = len(self.selected_clients)
        self.lr =  1 / (lr * math.sqrt(self.round_numer) + num_clients * beta)

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

        for c in clients:
            # c.model.set_params(self.model)
            c.model.set_lr_client(self.round_numer)
            comp, num_samples, update = c.train(num_epochs, batch_size, minibatch)
            eta_cur, eta_pre = c.model.get_eta()

            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((eta_cur, eta_pre))
            c.model.update_eta(self.model)

        return sys_metrics

    def update_model(self):
        base = [np.zeros((784, 62)), np.zeros(62)] 
        for i in range(len(self.updates)):
            if i in byzantine:
                eta_cur, eta_pre = self.updates[regular[0]]
            else:
                eta_cur, eta_pre = self.updates[i]
            for j in range(len(base)):
                base[j] += 2 * eta_cur[j] - eta_pre[j]
        self.set_lr_server()

        for i in range(len(self.model)):
            self.model[i] = self.model[i] + self.lr * base[i]

        self.updates = []

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
            # model_client = client.model.get_params()
            # client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
            # client.model.set_params(model_client)


        # for i in range(len(clients_to_test)):
        #     if i in regular:
        #         client = clients_to_test[i]
        #         c_metrics = client.test(set_to_use)
        #         metrics[client.id] = c_metrics
        
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
