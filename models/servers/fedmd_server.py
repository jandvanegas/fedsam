import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split

from baseline_constants import (BYTES_READ_KEY, BYTES_WRITTEN_KEY,
                                CLIENT_GRAD_KEY, CLIENT_PARAMS_KEY,
                                CLIENT_TASK_KEY)


class FedMdServer:
    def __init__(self, client_models, public_client_models, public_data, PublicDataset, batch_size, num_workers, device):
        self.client_models = [
            copy.deepcopy(client_model) for client_model in client_models
        ]
        self.public_client_models = [
            copy.deepcopy(client_model) for client_model in public_client_models
        ]
        self.devices = [client_model.device for client_model in self.client_models]
        self.public_devices = [
            client_model.device for client_model in self.public_client_models
        ]
        self.total_grad = 0
        self.selected_clients = []
        self.updates = []
        self.momentum = 0
        self.swa_model = None
        self.public_data = public_data
        self.public_data_size = len(public_data["x"])
        self.PublicDataset = PublicDataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

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
        clients_by_model = self.get_clients_by_model(possible_clients)
        self.selected_clients = []
        num_models = 5
        for model_index, model_clients in clients_by_model.items():
            this_num_clients = min(num_clients // num_models, len(model_clients))
            np.random.seed(my_round)
            self.selected_clients.extend(np.random.choice(model_clients, this_num_clients, replace=False))
        return [
            (c.num_train_samples, c.num_test_samples) for c in self.selected_clients
        ]

    def train_model(
        self,
        num_epochs=1,
        batch_size=10,
        minibatch=None,
        clients=None,
        pre_training=False,
        consensus=False
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
        if pre_training:
            print(f"{'*'*10} Pre-training public started {'*'*10}")
            pre_trained_models = {}
            all_models_trained = False
            loader = self.get_loader(5000)
            for c in clients:
                if len(pre_trained_models) == 5:
                    all_models_trained = True
                    break
                if c.model_index not in pre_trained_models:
                    model_without_last_layer, losses = c.pre_train(
                        loader=loader, num_epochs=num_epochs
                    )
                    pre_trained_models[c.model_index] = model_without_last_layer
            assert all_models_trained, "Not all models are trained"
            print(f"{'*'*10} Updating clients {'*'*10}")
            for c in clients:
                c.update_pretrained_model(pre_trained_models[c.model_index])
            print(f"{'*'*10} Pre-training public ended {'*'*10}")

        if consensus:
            print(f"{'*'*10} Training on consensus {'*'*10}")
            pre_trained_models = {}
            all_models_trained = False
            loader = self.consensus
            for c in clients:
                model_without_last_layer, losses = c.pre_train(
                    loader=loader, num_epochs=num_epochs
                )
                pre_trained_models[c.id] = model_without_last_layer
            print(f"{'*'*10} Updating clients {'*'*10}")
            for c in clients:
                c.update_pretrained_model(pre_trained_models[c.id])
            print(f"{'*'*10} Pre-training public ended {'*'*10}")
        print(f"{'*'*10} Private Training started {'*'*10}")

        for c in clients:
            num_samples, update = c.train(num_epochs)
            sys_metrics = self._update_sys_metrics(c, sys_metrics)
            self.updates.append((num_samples, copy.deepcopy(update)))
        print(f"{'*'*10} Private Training ended {'*'*10}")
        return sys_metrics

    def get_loader(self, size):
        public_loader = self.PublicDataset(
            self.public_data,
        )
        public_data_indexes = list(range(self.public_data_size))
        np.random.shuffle(public_data_indexes)
        # loader = Subset(public_loader, public_data_indexes[:size])
        public_sampler = SubsetRandomSampler(public_data_indexes[:size])

        return DataLoader(
            dataset=public_loader,
            sampler=public_sampler,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

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

    def test_model(self, clients_to_test, set_to_use="test"):
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
        loader = None
        if set_to_use == "public":
            loader = self.get_loader(5000)
        for client in clients_to_test:
            client.model_index
            c_metrics = client.test(set_to_use=set_to_use, loader=loader)
            metrics[client.id] = c_metrics

        return metrics

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        clients = self.selected_clients
        common_data = list(self.get_loader(5000))
        predictions = [torch.zeros(len(common_data[i][1])).to(self.device) for i in range(len(common_data))]
        for c in clients:
            c_predictions = c.test(set_to_use="public", loader=common_data)['predictions']
            for j, prediction_batch in enumerate(c_predictions):
               predictions[j] = torch.add(predictions[j], prediction_batch) 
        consensus = [[common_data[j][0], torch.div(prediction, len(clients)).long()] for j, prediction in enumerate(predictions)]
        self.consensus = consensus
        return consensus

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

    def get_model_grad(self, client_model):
        """Returns:
        self.total_grad: total gradient of the model (zero in case of FedAvg, where the gradient is never stored)"""
        # todo store gradient
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

    def save_all(self, clients, path):
        models = {
            f"client_{client.model_index}_model": client.model.state_dict()
            for client in clients
        }
        torch.save(models, path)

    def load_all(self, clients, path):
        models = torch.load(path)
        for client in clients:
            client.model.load_state_dict(models[f"client_{client.model_index}_model"])

    def get_clients_by_model(self, clients):
        clients_by_model = {}
        for client in clients:
            if client.model_index in clients_by_model.keys():
                clients_by_model[client.model_index].append(client)
            else:
                clients_by_model[client.model_index] = [client]
        return clients_by_model
