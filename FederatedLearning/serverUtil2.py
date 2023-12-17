import copy
import socket
import threading
from sklearn.linear_model import LogisticRegression
from time import sleep
import pandas as pd
import pickle
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"

class FLOrchestrator():
    """An orchestrator class that will manage the training and evaluation of the model over time.
    """

    def __init__(self, minClients, nEpochs, dataLoader, network, optimizer):
        self.minClients = minClients
        self.clients = []
        self.clientModels = []
        self.globalAccuracy = -1
        self.dataLoader = dataLoader
        self.network = network
        self.optimizer = optimizer
        self.nEpochs = nEpochs
        self.epoch = 1



    def registerClient(self, name):
        """Register a client with the server"""
        self.clients.append(name)
        print(f"[NEW CONNECTION] {name} connected.")
        print(f"[ACTIVE CLIENTS] {self.clients}")

    def globalTrain(self):
        """Trigger aggregation of client models"""
        avgWeights = copy.deepcopy(self.clientModels[0])
        for key in avgWeights.keys():
            for i in range(1, len(self.clientModels)):
                avgWeights[key] += self.clientModels[i][key]
            avgWeights[key] = torch.div(avgWeights[key], len(self.clientModels))
        self.network.load_state_dict(avgWeights)
        self.clientModels = []  # Reset client models
        self.scoreModel()
        self.epoch += 1



    def scoreModel(self):
        """Score the model with the global weights"""
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.dataLoader:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(self.dataLoader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.dataLoader.dataset),
            100. * correct / len(self.dataLoader.dataset)))
        self.globalAccuracy = 100. * correct / len(self.dataLoader.dataset)

