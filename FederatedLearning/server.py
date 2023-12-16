import socket
import threading
from sklearn.linear_model import LogisticRegression
from time import sleep
import pandas as pd
import pickle
import numpy as np

PORT = 5051
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"

class FLOrchestrator():
    """An orchestrator class that will manage the training and evaluation of the model over time.
    """

    def __init__(self, minClients, dataPath="dataset/iris4.csv"):
        self.minClients = minClients
        self.clients = []
        self.clientAccuracy = []
        self.clientCoef = []
        self.clientIntercept = []
        self.dataPath = dataPath

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(ADDR)

        self.model = LogisticRegression()
        dummyData = pd.DataFrame([[1, 1, 1, 1],
                        [2, 2, 2, 2],
                        [3, 3, 3, 3]], columns=["0", "1", "2", "3"])
        self.model.fit(dummyData, [0, 1, 2])



    def registerClient(self, conn, addr):
        """Register a client with the server"""
        self.clients.append((conn, addr))
        print(f"[NEW CONNECTION] {addr} connected.")
        print(f"[ACTIVE CLIENTS] {[c[1] for c in self.clients]}")



    def collectClients(self):
        """Collect client information until minClients has been reached"""
        self.server.listen()
        while True:
            conn, addr = self.server.accept()
            self.registerClient(conn, addr)
            if len(self.clients) >= self.minClients:
                break

    def trainClient(self, conn, addr):
        """Train a single client, and save their accuracy and model coefficients and intercept"""
        conn.send("START".encode(FORMAT))
        accuracy, coef, intercept = pickle.loads(conn.recv(10000))
        self.clientAccuracy.append(accuracy)
        self.clientCoef.append(coef)
        self.clientIntercept.append(intercept)
        print(f"[TRAINING] {addr} trained with accuracy {accuracy}")


    def globalTrain(self):
        """Trigger training for all clients and collect evaluation scores and model coefficients"""
        print("Training Clients")
        for client in self.clients:
            self.trainClient(client[0], client[1])

        print("Aggregating Coefficients")

        # Averaging the coefficients over the results
        globalCoef = self.clientCoef[0]
        for i in range(1, len(self.clientCoef)):
            globalCoef += self.clientCoef[i]
        globalCoef = globalCoef / len(self.clientCoef)
        globalIntercept = self.clientIntercept[0]
        for i in range(1, len(self.clientIntercept)):
            globalIntercept += self.clientIntercept[i]
        globalIntercept = globalIntercept / len(self.clientIntercept)

        self.model.coef_ = globalCoef
        self.model.intercept_ = globalIntercept

    def scoreModel(self):
        """Score the model with the global coefficients"""
        data = pd.read_csv(self.dataPath)
        xTest, yTest = data.iloc[:, :-1], data.iloc[:, -1]
        accuracy = self.model.score(xTest, yTest)
        print(accuracy)

    def run(self):
        self.collectClients()
        self.globalTrain()
        print("Global Model Coefficients")
        print(self.model.coef_)
        print("Global Model Intercept")
        print(self.model.intercept_)
        print("Global Model Accuracy")
        self.scoreModel()
        self.server.close()


orchestrator = FLOrchestrator(3)
orchestrator.run()
