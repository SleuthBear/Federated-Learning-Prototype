# Federated Learning Project
 An example federated learning model
 
SSL SETUP:

 - Install openssl
 - Generate a private key and a certificate signing request (CSR) with the following command:

>> openssl genrsa -out ca.key 2048
>> openssl req -new -x509 -days 365 -key ca.key -subj "/C=AU/ST=VICTORIA/L=MELBOURNE" -out ca.crt
>> openssl req -newkey rsa:2048 -nodes -keyout server.key -subj "/C=AU/ST=VICTORIA/L=MELBOURNE" -out server.csr
>> openssl x509 -req -in server.csr -extfile <(printf "subjectAltName=IP:127.0.0.1") -CA ca.crt -CAkey ca.key \
        -CAcreateserial -out server.crt -days 365

Python Environment Setup:
 - Make sure python is installed.
 - create a virtual environment with the following command:
>> python3 -m venv venv
 - Activate the virtual environment with the following command:
>> source venv/bin/activate
 - Install the required packages with the following command:
>> pip install -r requirements.txt
 
How to Run:
 - run dataDownload.py. this will download the iris data onto your system.
 - run server.py. This will intialize the flask app, so clients can make api calls
 - run client.py. This will create 3 simulated client computers, and register them with the server.
   The server will then trigger training to start, collect the trained models, aggregate and test them.

