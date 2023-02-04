import copy
import numpy as np
import torch
from baseline_constants import (
    BYTES_WRITTEN_KEY,
    BYTES_READ_KEY,
    CLIENT_PARAMS_KEY,
    CLIENT_GRAD_KEY,
    CLIENT_TASK_KEY,
)


class FedMdServer:
    def __init__(self, client_models, public_client_models):
        self.client_models = [
            copy.deepcopy(client_model) for client_model in client_models
        ]
        self.public_client_models = [
            copy.deepcopy(client_model) for client_model in public_client_models
        ]
        self.devices = [client_model.device for client_model in client_models]
        self.total_grad = 0
        self.selected_clients = []
        self.updates = []
        self.momentum = 0
        self.swa_model = None

    #################### METHODS FOR FEDERATED ALGORITHM ####################

    def select_clients(self, my_round, possible_clients, num_clients=20):
        """Selects num_clients clients randomly from possible_clients.

        Note that within function, num_clients is set to
            min(num_clients, len(possible_clients)).

        Args:
            possible_clients: Clients from which the servers can select.
            num_clients: Number of clients to select; default 20.
            my_round: Current round.
        Return:
            list of (num_train_samples, num_test_samples)
        """

        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(
            possible_clients, num_clients, replace=False
        )
        return [
            (c.num_train_samples, c.num_test_samples) for c in self.selected_clients
        ]

    def train_model(
        self, num_epochs=1, batch_size=10, minibatch=None, clients=None, public=False
    ):
        """Trains self.models on given clients.

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
            bytes_written: number of bytes written by each client to servers
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from servers
                dictionary with client ids as keys and integer values.
        """
        if clients is None:
            clients = self.selected_clients
        sys_metrics = {
            c.id: {
                BYTES_WRITTEN_KEY: 0,
                BYTES_READ_KEY: 0,
                CLIENT_PARAMS_KEY: 0,
                CLIENT_GRAD_KEY: 0,
                CLIENT_TASK_KEY: {},
            }
            for c in clients
        }

        for c in clients:
            num_samples, update = c.train(num_epochs, batch_size, minibatch)
            sys_metrics = self._update_sys_metrics(c, sys_metrics)
            self.updates.append((num_samples, copy.deepcopy(update)))

        return sys_metrics

    def _update_sys_metrics(self, c, sys_metrics):
        if isinstance(c.model, torch.nn.DataParallel):
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.module.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.module.size
        else:
            sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size

        sys_metrics[c.id][CLIENT_PARAMS_KEY] = c.params_norm()
        sys_metrics[c.id][CLIENT_GRAD_KEY] = c.total_grad_norm()
        sys_metrics[c.id][CLIENT_TASK_KEY] = c.get_task_info()
        return sys_metrics

    def test_model(self, clients_to_test, batch_size, set_to_use="test"):
        """Tests models.

        Tests model on self.selected_clients if clients_to_test=None.
        For each client, the current servers model is loaded before testing it.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            model = self.client_models[client.model_index]
            # TODO implement load_state_dict
            # if self.swa_model is None:
            #     client.model.load_state_dict(model)
            # else:
            #     client.model.load_state_dict(self.swa_model.state_dict())
            c_metrics = client.test(batch_size, set_to_use)
            metrics[client.id] = c_metrics

        return metrics

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        # Average updates (fedavg)
        # todo implement update

        self.models = [
            copy.deepcopy(client_model.state_dict())
            for client_model in self.client_models
        ]
        self.updates = []
        return

    def update_clients_lr(self, lr, clients=None):
        if clients is None:
            clients = self.selected_clients
        for c in clients:
            c.update_lr(lr)

    #################### METHODS FOR ANALYSIS ####################

    def set_num_clients(self, n):
        """Sets the number of total clients"""
        self.num_clients = n

    def set_num_public_clients(self, n):
        """Sets the number of total clients"""
        self.public_num_clients = n

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.
        Returns info about self.selected_clients if clients=None;
        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        num_samples = {c.id: c.num_samples for c in clients}
        return ids, num_samples

    def num_parameters(self, params):
        """Number of model parameters requiring training"""
        return sum(p.numel() for p in params if p.requires_grad)

    def get_model_params_norm(self, client_model):
        """Returns:
        total_params_norm: L2-norm of the model parameters"""
        total_norm = 0
        for p in client_model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_params_norm = total_norm**0.5
        return total_params_norm

    def get_model_grad(self):
        """Returns:
        self.total_grad: total gradient of the model (zero in case of FedAvg, where the gradient is never stored)"""
        return self.total_grad

    def get_model_grad_by_param(self, client_model):
        """Returns:
        params_grad: dictionary containing the L2-norm of the gradient for each trainable parameter of the network
                    (zero in case of FedAvg where the gradient is never stored)"""
        params_grad = {}
        for name, p in client_model.named_parameters():
            if p.requires_grad:
                try:
                    param_norm = p.grad.data.norm(2)
                    params_grad[name] = param_norm
                except Exception:
                    # this param had no grad
                    pass
        return params_grad

    #################### METHODS FOR SAVING CHECKPOINTS ####################

    def save_model(self, round, ckpt_path, swa_n=None):
        """Saves the servers model on checkpoints/dataset/model_name.ckpt."""
        # Save servers model
        save_info = {
            "model_state_dict": [model for model in self.models],
            "round": round,
        }
        if self.swa_model is not None:
            save_info["swa_model"] = self.swa_model.state_dict()
        if swa_n is not None:
            save_info["swa_n"] = swa_n
        torch.save(save_info, ckpt_path)
        return ckpt_path
