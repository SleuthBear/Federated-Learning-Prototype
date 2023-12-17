import socket
import threading
from sklearn.linear_model import LogisticRegression
from time import sleep
import pandas as pd
import pickle
import numpy as np

PORT = 5050
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"

class FLOrchestrator():
    """An orchestrator class that will manage the training and evaluation of the model over time.
    """

    def __init__(self, minClients, dataPath="dataset/iris4.csv"):
        self.minClients = minClients
        self.clients = []
        self.clientAccuracies = []
        self.clientCoefficients = []
        self.clientIntercepts = []
        self.dataPath = dataPath
        self.globalAccuracy = -1

        self.model = LogisticRegression()
        dummyData = pd.DataFrame([[1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [3, 3, 3, 3]], columns=["0", "1", "2", "3"])
        self.model.fit(dummyData, [0, 1, 2])



    def registerClient(self, name):
        """Register a client with the server"""
        self.clients.append(name)
        print(f"[NEW CONNECTION] {name} connected.")
        print(f"[ACTIVE CLIENTS] {[c[1] for c in self.clients]}")

    def globalTrain(self):
        """Trigger aggregation of client models"""
        print("Aggregating Coefficients")

        # Averaging the coefficients over the results
        globalCoef = self.clientCoefficients[0]
        for i in range(1, len(self.clientCoefficients)):
            globalCoef += self.clientCoefficients[i]
        globalCoef = globalCoef / len(self.clientCoefficients)

        # Averaging the intercepts over the results
        globalIntercept = self.clientIntercepts[0]
        for i in range(1, len(self.clientIntercepts)):
            globalIntercept += self.clientIntercepts[i]
        globalIntercept = globalIntercept / len(self.clientIntercepts)

        self.model.coef_ = globalCoef
        self.model.intercept_ = globalIntercept

    def scoreModel(self):
        """Score the model with the global coefficients"""
        data = pd.read_csv(self.dataPath)
        xTest, yTest = data.iloc[:, :-1], data.iloc[:, -1]
        self.globalAccuracy = self.model.score(xTest, yTest)
