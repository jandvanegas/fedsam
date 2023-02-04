import torch
import random
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from cifar10.dataloader import ClientDataset as PublicDataset
from cifar100.dataloader import ClientDataset as PrivateDataset
from .resnet20 import ResNet20
from .model_build import build_model


def train_models(device, models, train_clients, train_dataset, lr, optimizer, epochs):
    """
    Train an array of models on the same dataset.
    """
    private_model_public_dataset_train_losses = []
    for n, model in enumerate(models):
        print("Training model ", n)
        print("Mdeol Type index", modelsindex[n])
        model.to(device)
        model.train()
        if optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.5)
        elif optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        trainloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        criterion = torch.nn.NLLLoss().to(device)
        train_epoch_losses = []
        print("Begin Public Training")
        for epoch in range(epochs):
            train_batch_losses = []
            for batch_idx, (images, labels) in enumerate(trainloader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    print(
                        "Local Model {} Type {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            n,
                            modelsindex[n],
                            epoch + 1,
                            batch_idx * len(images),
                            len(trainloader.dataset),
                            100.0 * batch_idx / len(trainloader),
                            loss.item(),
                        )
                    )
                train_batch_losses.append(loss.item())
            loss_avg = sum(train_batch_losses) / len(train_batch_losses)
            train_epoch_losses.append(loss_avg)

        torch.save(
            model.state_dict(),
            "Src/Model/LocalModel{}Type{}Epoch{}.pkl".format(n, modelsindex[n], epochs),
        )
        private_model_public_dataset_train_losses.append(train_epoch_losses)


from .option import args_parser

if __name__ == "__main__":

    args = args_parser()
    device = "cuda" if args.gpu else "cpu"

    from utils.model_utils import read_dir

    train_clients, _, train_data = read_dir("../data/train_data", args.alpha)

    models = {
        "resnet_20": ResNet20,
        "CNN_3_128_128_192": build_model(
            num_classes=10, n1=128, n2=128, n3=192, softmax=True
        ),
        "CNN_3_64_64_64": build_model(
            num_classes=10, n1=64, n2=64, n3=64, softmax=True
        ),
        "CNN_3_128_64_64": build_model(
            num_classes=10, n1=128, n2=64, n3=64, softmax=True
        ),
        "CNN_3_64_64_128": build_model(
            num_classes=10, n1=64, n2=64, n3=128, softmax=True
        ),
    }
    modelsindex = [
        "resnet_20",
        "CNN_3_128_128_192",
        "CNN_3_64_64_64",
        "CNN_3_128_64_64",
        "CNN_3_64_64_128",
    ]

    train_models(
        device,
        models,
        train_clients,
        train_data,
        args.lr,
        args.optimizer,
        args.epoch,
    )
