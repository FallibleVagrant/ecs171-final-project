import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import CustomDataSet
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score

#tuning
from ray import train,tune
from ray.train import Checkpoint
import os
import tempfile
from torch.utils.data import random_split
import numpy as np
from ray.tune.schedulers import ASHAScheduler


class MLP(nn.Module):

    def __init__(self, num_of_params, num_of_labels, configuration=0, l1=0, l2=0, l3=0, l4=0):
        super(MLP, self).__init__()
        self.num_of_params = num_of_params
        self.num_of_labels = num_of_labels
        self.model_configuration = configuration
        
        if configuration == 0:
            # coniguration 0, 55% accuracy 1.49 CEL | 0.03 lr | 100 epochs
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, 128),
                nn.ReLU(),
                nn.Linear(128, 84),
                nn.ReLU(),
                nn.Linear(84, self.num_of_labels),
                nn.Softmax(dim=1),
            )
        elif configuration == 1:
            # configuration 1, 50% acc 1.4 CEL | 0.01 lr | 100 epochs
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, 100),
                nn.ReLU(),
                nn.Linear(100, 56),
                nn.ReLU(),
                nn.Linear(56, 30),
                nn.ReLU(),
                nn.Linear(30, self.num_of_labels),
                nn.ReLU(),
                # nn.Softmax(dim=1),   
            )
        elif configuration == 2:
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, 40),
                nn.ReLU(),
                nn.Linear(40, 80),
                nn.ReLU(),
                nn.Linear(80, 100),
                nn.ReLU(),
                nn.Linear(100, 36),
                nn.ReLU(),
                nn.Linear(36, self.num_of_labels),
                nn.ReLU(),    
            )
        elif configuration == 3:
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, 20),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(20, self.num_of_labels),
                nn.Sigmoid(),
            )
        elif configuration == 4:
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, 20),
                nn.ReLU(),
                nn.Linear(20, 40),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(40, self.num_of_labels),
                nn.Sigmoid(),             
            )
        elif configuration == 5:
            
            # didn't really work, started from basic 37% accuracy, 100 epochs 0.03 lr
            self.model = nn.Sequential(
                nn.Linear(self.num_of_params, self.num_of_labels),
                nn.Sigmoid(),
            )
        elif configuration == 6:
            # 41
            # 43
            # 43 | 1.56
            self.model = nn.Sequential(
                
                nn.Linear(self.num_of_params, 300),
                nn.ReLU(),
                nn.Linear(300, self.num_of_labels),
                nn.Sigmoid(),
            )
        elif configuration == 7:
            # 44 xd
            self.model = nn.Sequential(
                # nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, l1),
                nn.ReLU(),
                nn.Linear(l1, l2),
                nn.ReLU(),
                nn.Linear(l2, self.num_of_labels),
                nn.Softmax(dim=1),
            )
        elif configuration == 8:
            # 53% 60e .01 lr
            self.model = nn.Sequential(
                nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, l1), # 720
                nn.ReLU(),
                nn.Linear(l1, l2), # 256
                nn.ReLU(),
                nn.Linear(l2, l3), # 84
                nn.ReLU(),
                nn.Linear(l3, l4),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(l4, self.num_of_labels),
                nn.Softmax(dim=1)
            )
        elif configuration == 9:
            self.model = nn.Sequential(
                # nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, 400),
                nn.ReLU(),
                nn.Linear(400, 200),
                nn.ReLU(),
                nn.Linear(200, 132),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(132, 84),
                nn.ReLU(),
                nn.Linear(84, 60),
                nn.ReLU(),
                nn.Linear(60, self.num_of_labels),
                nn.Softmax(dim=1),   
            )
        elif configuration == 10:
            # 2 layers with best parameters
            self.model = nn.Sequential(
                nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_of_labels),
                nn.ReLU(),
                nn.Softmax(dim=1)
            )
        elif configuration == 11:
            # 4 layers with the best parameters
            self.model = nn.Sequential(
                nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(256, self.num_of_labels),
                nn.Softmax(dim=1)
            )
        elif configuration == 12:
            # 3 layers with the best parameters
            # nn.BatchNorm1d(self.num_of_params),
            self.model = nn.Sequential(
                nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.ReLU(),
                nn.Linear(16, 32), 
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(32, self.num_of_labels),
                nn.Softmax(dim=1)
            )
            
             
            
            
        elif configuration == 13:
            self.model = nn.Sequential(
                nn.BatchNorm1d(self.num_of_params),
                nn.Linear(self.num_of_params, 720),
                nn.ReLU(),
                nn.Linear(720, 256), # 256
                nn.ReLU(),
                nn.Linear(256, 84), # 84
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(84, self.num_of_labels),
                nn.Softmax(dim=1)
            )
            
    def forward(self, x):
        return self.model(x)


def train_model(data_loader, model, optimizer, loss_function):
    model.train()
    avg_loss = 0
    for _, (X, y) in enumerate(data_loader):
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        avg_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return avg_loss / len(data_loader)


def test_model(data_loader, model, loss_function, mode="Test"):
    model.eval()
    num_of_batches = len(data_loader)
    loss, correct = 0, 0
    precision, recall = 0, 0
    with torch.no_grad():

        for _, (X, y) in enumerate(data_loader):
            prediction = model(X)
            loss += loss_function(prediction, y).item()
            y_true = y.argmax(1)
            y_pred = prediction.argmax(1)
            
            correct += (
                (prediction.argmax(1) == y.argmax(1)).type(torch.float32).sum().item()
            )
            precision += precision_score(
                y_true, y_pred, average="macro", zero_division=True
            )
            recall += recall_score(
                y_true, y_pred, average="macro", labels=range(2), zero_division=True
            )

    accuracy = 100 * correct / len(data_loader.dataset)
    recall /= num_of_batches / 100
    precision /= num_of_batches / 100
    f1 = 2 * precision * recall / (precision + recall)
    loss /= num_of_batches

    # print(
    #     f"""
    #     Metrics for model MLP on {mode} data, configured {model.model_configuration}:
    #     -- Accuracy: {accuracy:>.2f} %
    #     -- Loss: {loss:>.2f}
    #     """
    # )

    print(
        f"""
        Metrics for model MLP on {mode} data {model.model_configuration}:
        -- Accuracy: {accuracy:>.2f} %
        -- Recall: {recall:>.2f} %
        -- Precision: {precision:>.2f} %
        -- F1: {f1:>.2f} %
        -- Loss: {loss:>.2f}
        """
    )
    return accuracy

def run_model(loss_fn, lr=0.001, epochs=100, model_conf=0, batch_size=32, disable_msg=False):
    train_data = CustomDataSet(train=True)
    test_data = CustomDataSet(train=False)

    batch_size = len(train_data) if batch_size == "full" else batch_size

    train_data = DataLoader(train_data, batch_size=batch_size, drop_last=(batch_size!="full"))
    test_data = DataLoader(test_data, batch_size=len(test_data))

    model = MLP(train_data.dataset.num_of_params, train_data.dataset.num_of_labels, configuration=model_conf)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loss = []
    
    for i in range(epochs):
        if not disable_msg: 
            print(f"Epoch {i+1} ---------------------")
        train_loss.append(train_model(train_data, model, optimizer, loss_fn))
        if not disable_msg:
            print(f"Loss: {train_loss[i]}")

    test_model(test_data, model, loss_fn)
    test_model(train_data, model, loss_fn, mode="Train")

    plt.plot(train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss (cross entropy)")
    plt.show()


# tuning

def tuning_train_mlp(config):
    
    trainset = CustomDataSet(train=True)
    net = MLP(trainset.num_of_params, trainset.num_of_labels, 8, config["l1"], config["l2"], config["l3"], config["l4"])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    device = "cpu"

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)


    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs])

    trainloader = DataLoader(
        train_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers =  8,
    )
    
    valloader = DataLoader(
        val_subset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=8,
    )

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print(f"{predicted=}; {labels=}")
                correct += (predicted == labels.argmax(1)).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (net.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def tuning_test_best_model(best_result):
    dataset = CustomDataSet(train=True)
    best_trained_model = MLP(dataset.num_of_params, dataset.num_of_labels, 8, best_result.config["l1"], best_result.config["l2"],
                             best_result.config["l3"], best_result.config["l4"])
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_result.checkpoint.to_directory(), "checkpoint.pt")

    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    testset = CustomDataSet(train=False)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = best_trained_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.argmax(1)).sum().item()


    print("Best trial test set accuracy: {}".format(correct / total))


def run_tuning(num_samples, max_num_epochs, gpus_per_trial):
    
    config = {
        "l1": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
        "l2": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
        "l3": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
        "l4": tune.sample_from(lambda _: 2**np.random.randint(2, 10)),
        "lr": tune.loguniform(1e-5, 1e-1),
        "batch_size": tune.choice([4, 8, 16, 32, 64])
    }
    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(tuning_train_mlp),
            resources={"cpu": 8, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
    )
    results = tuner.fit()
    
    best_result = results.get_best_result("loss", "min")

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(
        best_result.metrics["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_result.metrics["accuracy"]))

    tuning_test_best_model(best_result )



if __name__ == "__main__":

    run_tuning(num_samples=15, max_num_epochs=50, gpus_per_trial=0)
    
    # run_model(nn.CrossEntropyLoss(), lr=0.00001, epochs=100, model_conf=9, batch_size=128)

    # epochs = 1000
    # lr = 1e-3

    # data_train = CustomDataSet(train=True)
    # data_test = CustomDataSet(train=False)
    # data_train = DataLoader(data_train, batch_size=len(data_train))
    # data_test = DataLoader(data_test, batch_size=len(data_test))

    # m = MLP(data_train.dataset.num_of_params, data_train.dataset.num_of_labels, configuration=4)

    # l = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(m.parameters(), lr=lr)
    
    # train_loss = []
    # for i in range(epochs):
    #     print(f"Epochs {i+1} -----")
    #     train_loss.append(train(data_test, m, opt, l))
    #     print(f"Train Loss: {train_loss[i]}")

    # test(data_test, m, l )
    # test(data_train, m, l,  mode="Train")



    # plt.plot(train_loss)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss (cross entropy)")
    # plt.show()


