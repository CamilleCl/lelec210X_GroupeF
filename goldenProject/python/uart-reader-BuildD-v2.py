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
from numpy import random
import sounddevice as sd
import soundfile as sf

import time

from classification.datasets import Dataset, get_cls_from_path
from classification.utils.plots import plot_audio, plot_specgram

import threading
from threading import Thread
import queue

# creating the socket
host = socket.gethostname()
port = 5002

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

result_filename = "predicted_class.csv"
melvec_dir = "melvectors/"

dt = np.dtype(np.uint16).newbyteorder("<")

model_dir = "../../classification/data/models/" # where to save the models
filename = 'model.pickle'
model = pickle.load(open(model_dir + filename, 'rb'))


dataset = Dataset()
classes = dataset.list_classes()
print(classes)
SoundPerClasse = 20
classe = 'birds'


data_lock = threading.Lock()


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        print(line)
        return None


def read_serial(port,data_queue):
    with serial.Serial(port=port, baudrate=115200) as ser:
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
                with data_lock:    
                    data_queue.put(buffer_array)
                #yield buffer_array


def write_serial(port):
    with serial.Serial(port=port, baudrate=115200) as ser:
        while True:
            for i in range(SoundPerClasse):
                sound = dataset[classe, i]
                x, fs = sf.read(sound)
                # target_dB = 25
                # x /= np.linalg.norm(x) * 10 ** (-target_dB / 20)
                print(f'Playing a "{get_cls_from_path(sound)}"')
                sd.play(x, fs)

                sleeptime = random.uniform(0, 4)
                print("sleeping for:", sleeptime, "seconds")
                time.sleep(sleeptime)

                with data_lock:
                    ser.write(bytearray('s','ascii'))
                print("message sent to uart")

                time.sleep(7-sleeptime)


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
            input_queue = queue.Queue()

            reader_thread = Thread(target=read_serial, args=(args.port, input_queue))
            writer_thread = Thread(target=write_serial, args=(args.port,))

            reader_thread.start()
            writer_thread.start()

            msg_counter = 0

            while True:
                try:
                    with data_lock:
                        melvec = input_queue.get(block=False)
                    msg_counter += 1

                    print("MEL vector #{}".format(msg_counter))

                    melvec = np.reshape(melvec, (1, N_MELVECS * MELVEC_LENGTH))
                    melvec_normalized = melvec / np.linalg.norm(melvec, keepdims=True)

                    #enregistrement des melvecs de la vraie chaine de communication
                    filename = "melvec_{}".format(msg_counter)
                    pickle.dump(melvec_normalized, open(melvec_dir+filename, 'wb'))

                except queue.Empty:
                    pass
        
    except KeyboardInterrupt:
        exit(0)
