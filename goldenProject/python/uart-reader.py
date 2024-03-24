# -*- coding: utf-8 -*-
"""
uart-reader.py
ELEC PROJECT - 210x
"""
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import numpy as np
import serial
from serial.tools import list_ports
import pickle
import socket

import requests
import json

hostname = "http://lelec210x.sipr.ucl.ac.be"
key = "jc5jE0qHTmt1l-0EYOYJ3HzxEB8vIb6qtNm6dI3w"



from classification.utils.plots import plot_specgram

# creating the socket
host = socket.gethostname()
port = 5006
server_socket = socket.socket() 

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

result_filename = "predicted_class.csv"

dt = np.dtype(np.uint16).newbyteorder("<")

model_dir = "models/" # where to save the models
filename = 'KNN.pickle'
model = pickle.load(open(model_dir + filename, 'rb'))


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def reader(port=None):
    ser = serial.Serial(port=port, baudrate=115200)
    while True:
        line = ""
        while not line.endswith("\n"):
            line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                "ascii"
            )
            print(line)
        line = line.strip()
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array

def reader_socket(): 

    server_socket.bind((host, port))

    server_socket.listen(1)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))

    while True:
        line = conn.recv(16 * N_MELVECS * MELVEC_LENGTH).decode('ascii')
        line = line.strip()

        print(line)
        
        buffer = parse_buffer(line)
        if buffer is not None:
            buffer_array = np.frombuffer(buffer, dtype=dt)

            yield buffer_array
                  
if __name__ == "__main__":
    try:
        # file = open(result_filename, 'w')
        # file.close()

        argParser = argparse.ArgumentParser()
        argParser.add_argument("-p", "--port", help="Port for serial communication")
        args = argParser.parse_args()
        print("uart-reader launched...\n")

        if args.port is None:
            print(
                "No port specified, here is a list of serial communication port available"
            )
            print("================")
            port = list(list_ports.comports())
            for p in port:
                print(p.device)
            print("================")
            print("Launch this script with [-p PORT_REF] to access the communication port")

        else:
            if args.port == "socket":
                input_stream = reader_socket()
            else:
                input_stream = reader(port=args.port)

            msg_counter = 0

            for melvec in input_stream:
                msg_counter += 1

                print("MEL Spectrogram #{}".format(msg_counter))

                melvec = np.reshape(melvec, (1, N_MELVECS * MELVEC_LENGTH))
                melvec_normalized = melvec / np.linalg.norm(melvec, keepdims=True)

                plt.figure()
                plot_specgram(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, ax=plt.gca(), is_mel=True, title="")
                plt.draw()
                plt.pause(0.001)
                plt.show()

                y_predict = model.predict(melvec_normalized)

                print(f'predicted class: {y_predict[0]}')

                try:
                    response = requests.post(f"{hostname}/lelec210x/leaderboard/submit/{key}/{y_predict[0]}", timeout=1)
                    
                    # All responses are JSON dictionaries
                    response_as_dict = json.loads(response.text)
                    print(f'server response : {response_as_dict}')
                    
                except Exception as error:
                    print(error)

    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Shutting down server")
        server_socket.close()
        exit(0)
