import socket
import pandas as pd
import threading
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import requests
from time import sleep

PORT = 5000
SERVER = socket.gethostbyname(socket.gethostname())
ADDR = (SERVER, PORT)
FORMAT = "utf-8"


class Client():

    def __init__(self, host, port, dataPath, username, password):
        self.host = host
        self.port = port
        self.data = pd.read_csv(dataPath)
        self.username = username
        self.password = password

    def trainModel(self):
        """Train the model with local data"""
        xTrain, xTest, yTrain, yTest = train_test_split(self.data.iloc[:, :-1], self.data.iloc[:, -1])
        model = LogisticRegression()

        model.fit(xTrain, yTrain)
        accuracy = model.score(xTest, yTest)
        # convert to list so they can be serialized
        coefficients = model.coef_.tolist()
        intercept = model.intercept_.tolist()
        requests.post(f"https://{self.host}:{self.port}/trainClient",
                      auth=(self.username, self.password),
                      json={"accuracy": accuracy, "coefficients": coefficients, "intercepts": intercept},
                      verify="server.crt")
        print("[MODEL TRAINED] Accuracy: ", accuracy)

    def standBy(self):
        """Ping the server until training is ready to begin. This gives control over when to train to the server."""
        print("Waiting for server to initiate training\n")
        # Register the client with the server
        requests.post(f"https://{self.host}:{self.port}/registerClient",
                      auth=(self.username, self.password),
                      verify="server.crt")
        while True:
            r = requests.get(f"https://{self.host}:{self.port}/readyToTrain",
                                    auth=(self.username, self.password),
                                    verify="server.crt")
            print(r.json())
            if r.json()["ready"]:
                self.trainModel()
                break
            sleep(1)  # ping the server once a second per thread


def createClient(dataPath, username, password):
    """Create a client and register it with the server"""
    runningClient = Client(SERVER, PORT, dataPath, username, password)
    runningClient.standBy()


# Create 3 clients and have them on standby
thread1 = threading.Thread(target=createClient, args=("dataset/iris1.csv", "user1", "pass1"))
thread1.start()
thread2 = threading.Thread(target=createClient, args=("dataset/iris2.csv", "user2", "pass2"))
thread2.start()
thread3 = threading.Thread(target=createClient, args=("dataset/iris3.csv", "user3", "pass3"))
thread3.start()
thread1.join()
thread2.join()
thread3.join()

# Now request the global metrics
r = requests.get(f"https://{SERVER}:{PORT}/getGlobalMetrics",
                        auth=("admin", "admin"),
                        verify="server.crt")
print("Global Metrics: ", r.json())
