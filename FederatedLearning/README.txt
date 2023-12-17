SSL SETUP:
*1. Install openssl

*2. Generate a private key and a certificate signing request (CSR) with the following command:

>> openssl genrsa -out ca.key 2048
>> openssl req -new -x509 -days 365 -key ca.key -subj "/C=AU/ST=VICTORIA/L=MELBOURNE" -out ca.crt
>> openssl req -newkey rsa:2048 -nodes -keyout server.key -subj "/C=AU/ST=VICTORIA/L=MELBOURNE" -out server.csr
>> openssl x509 -req -in server.csr -extfile <(printf "subjectAltName=IP:127.0.0.1") -CA ca.crt -CAkey ca.key \
        -CAcreateserial -out server.crt -days 365

Python Environment Setup:
*1. Make sure python is installed.
*2. create a virtual environment with the following command:
>> python3 -m venv venv
*3. Activate the virtual environment with the following command:
>> source venv/bin/activate
*4. Install the required packages with the following command:
>> pip install -r requirements.txt

