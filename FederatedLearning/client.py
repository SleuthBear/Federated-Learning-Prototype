import socket
import pandas as pd
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np
import requests

PORT = 5051
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"


class Client():

    def __init__(self, host, port, dataPath):
        self.host = host
        self.port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.connect((host, port))
        self.data = pd.read_csv(dataPath)


    def trainModel(self):
        """Train the model with local data"""
        xTrain, xTest, yTrain, yTest = train_test_split(self.data.iloc[:, :-1], self.data.iloc[:, -1])
        model = LogisticRegression()

        model.fit(xTrain, yTrain)
        accuracy = model.score(xTest, yTest)
        coef = np.array(model.coef_)
        intercept = np.array(model.intercept_)
        metrics = pickle.dumps([accuracy, coef, intercept])
        self.client.send(metrics)
        print("[MODEL TRAINED] Accuracy: ", accuracy)

    def standBy(self):
        """Wait for the server to initiate training"""
        print("Waiting for server to initiate training\n")
        msg = self.client.recv(64).decode(FORMAT)
        print(msg)
        self.trainModel()



def createClient(dataPath):
    """Create a client and register it with the server"""
    runningClient = Client(SERVER, PORT, dataPath)
    runningClient.standBy()


path1 = "dataset/iris1.csv"
path2 = "dataset/iris2.csv"
path3 = "dataset/iris3.csv"

# Create 3 clients and have them on standby
thread = threading.Thread(target=createClient, args=(path1,))
thread.start()
thread = threading.Thread(target=createClient, args=(path2,))
thread.start()
thread = threading.Thread(target=createClient, args=(path3,))
thread.start()