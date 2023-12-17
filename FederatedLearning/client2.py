import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from network import Net
import socket
import pandas as pd
import json
import threading
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import requests
from time import sleep

PORT = 5000
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)






class Client():

    def __init__(self, host, port, username, password, optimizer, dataLoader):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.network = None
        self.optimizer = optimizer
        self.dataLoader = dataLoader


    def trainModel(self):
        """Train the model with local data"""
        self.network.train()
        for batch_idx, (data, target) in enumerate(self.dataLoader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        requests.post(f"https://{self.host}:{self.port}/trainClient",
                      auth=(self.username, self.password),
                      json={"model": self.network.state_dict()},
                      verify="server.crt")
        print("[MODEL TRAINED]", self.username)


    def standBy(self):
        """Ping the server until training is ready to begin. This gives control over when to train to the server."""
        print("Waiting for server to initiate training\n")
        # Register the client with the server
        requests.post(f"https://{self.host}:{self.port}/registerClient",
                      auth=(self.username, self.password),
                      verify="server.crt")
        while True:
            response = requests.get(f"https://{self.host}:{self.port}/readyToTrain",
                                    auth=(self.username, self.password),
                                    verify="server.crt")
            # If we have looped through all epochs
            print(response)
            if response.json()["ready"] == "Ready":
                # Take the new global model
                state = json.loads(response.json()["model"], object_pairs_hook=OrderedDict)
                self.network.load_state_dict(torch.Tensor(state))
                self.trainModel()
            elif response.json()["ready"] == "Finished":
                break
            sleep(1)  # ping the server once a second per thread



def createClient(username, password, optimizer, dataloader):
    """Create a client and register it with the server"""
    runningClient = Client(SERVER, PORT, username, password, optimizer, dataloader)
    runningClient.standBy()

dataset = torchvision.datasets.MNIST('dataset/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

# Split the training data into 3 groups
train_loader1 = torch.utils.data.DataLoader(data_utils.Subset(dataset, range(0, 20000)),
                                            batch_size=batch_size_train,
                                            shuffle=True)
train_loader2 = torch.utils.data.DataLoader(data_utils.Subset(dataset, range(20000, 40000)),
                                            batch_size=batch_size_train,
                                            shuffle=True)
train_loader3 = torch.utils.data.DataLoader(data_utils.Subset(dataset, range(40000, 60000)),
                                            batch_size=batch_size_train,
                                            shuffle=True)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

# Create 3 clients and have them on standby
thread1 = threading.Thread(target=createClient, args=("user1", "pass1", optimizer, train_loader1))
thread1.start()
thread2 = threading.Thread(target=createClient, args=("user2", "pass2", optimizer, train_loader2))
thread2.start()
thread3 = threading.Thread(target=createClient, args=("user3", "pass3", optimizer, train_loader3))
thread3.start()
thread1.join()
thread2.join()
thread3.join()
