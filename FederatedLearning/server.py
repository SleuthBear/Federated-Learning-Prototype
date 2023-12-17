from flask import Flask
from flask import request
from serverUtil import FLOrchestrator
import json
import numpy as np

app = Flask(__name__)

orchestrator = FLOrchestrator(3)
f = open("users.json", "r")
users = json.load(f)


def valid_user(user, password):
    if user in users:
        if users[user] == password:
            return True
    return False


@app.route('/postModelInfo', methods=['POST'])
def postModelInfo():
    if valid_user(request.authorization.username, request.authorization.password):
        print("Valid User")
        orchestrator.clientAccuracy.append(float(request.json['accuracy']))
        print(orchestrator.clientAccuracy)
        return 'Success'
    else:
        print("Invalid User")
    return 'Failure'


@app.route('/readyToTrain', methods=['GET'])
def readyToTrain():
    if valid_user(request.authorization.username, request.authorization.password):
        print("Valid User")
        return json.dumps({"ready": len(orchestrator.clients) >= orchestrator.minClients})
    else:
        print("Invalid User")
    return json.dumps({"ready": False})


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
        orchestrator.clientIntercepts.append(np.asarray(request.json['intercepts']))
        orchestrator.clientCoefficients.append(np.asarray(request.json['coefficients']))
        orchestrator.clientAccuracies.append(request.json['accuracy'])
        print(f"Client {request.authorization.username} trained. Accuracy {request.json['accuracy']}")
    else:
        print("Invalid Client Train")

    if len(orchestrator.clientAccuracies) >= len(orchestrator.clients):
        print("All Clients Trained")
        orchestrator.globalTrain()
    return 'Success'

@app.route('/getGlobalMetrics', methods=['GET'])
def getGlobalMetrics():
    if valid_user(request.authorization.username, request.authorization.password):
        orchestrator.scoreModel()
        return json.dumps({"accuracy": orchestrator.globalAccuracy,
                           "coefficients": orchestrator.model.coef_.tolist(),
                           "intercepts": orchestrator.model.intercept_.tolist()})
    else:
        print("Invalid Client Metrics")
    return json.dumps({"accuracy": -1})

if __name__ == '__main__':
    context = ('server.crt', 'server.key')  # certificate and key files
    app.run(debug=False, ssl_context=context)