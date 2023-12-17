import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from network import Net
import torch.utils.data as data_utils
from flask import Flask
from flask import request
from serverUtil2 import FLOrchestrator
import json
import numpy as np

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


app = Flask(__name__)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('dataset/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
orchestrator = FLOrchestrator(3, 3, test_loader, network, optimizer)
f = open("users.json", "r")
users = json.load(f)


def valid_user(user, password):
    if user in users:
        if users[user] == password:
            return True
    return False


@app.route('/readyToTrain', methods=['GET'])
def readyToTrain():
    if valid_user(request.authorization.username, request.authorization.password):
        if orchestrator.epoch <= orchestrator.nEpochs:
            print("Valid User")
            print(orchestrator.network.state_dict())
            return {"ready": "Ready", "model": orchestrator.network.state_dict()}
        else:
            return {"ready": "Finished", "model": orchestrator.network.state_dict()}
    else:
        print("Invalid User")
        return "Failure"



@app.route('/registerClient', methods=['POST'])
def registerClient():
    if valid_user(request.authorization.username, request.authorization.password):
        orchestrator.registerClient(request.authorization.username)
        return 'Success'
    else:
        print("Invalid Client Register")
    return 'Failure'


@app.route('/trainClient', methods=['POST'])
def trainClient():
    """Train a single client, and save their accuracy and model coefficients and intercept,
    then trigger aggregation of client models if all clients have trained"""
    if valid_user(request.authorization.username, request.authorization.password):
        orchestrator.clientModels.append(request.json['model'])
    else:
        print("Invalid Client Train")

    if len(orchestrator.clientModels) >= len(orchestrator.clients):
        print("All Clients Trained")
        orchestrator.globalTrain()
    return 'Success'

if __name__ == '__main__':
    context = ('server.crt', 'server.key')  # certificate and key files
    app.run(debug=False, ssl_context=context)