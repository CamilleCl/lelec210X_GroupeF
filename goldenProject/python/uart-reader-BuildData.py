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

import librosa  # For audio signal computations as MFCC
from numpy import random
import sounddevice as sd
import soundfile as sf
from scipy import signal
from scipy.fftpack import dct

import time

from classification.datasets import Dataset, get_cls_from_path
from classification.utils.plots import plot_audio, plot_specgram


# creating the socket
host = socket.gethostname()
port = 5002
server_socket = socket.socket() 

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


def parse_buffer(line):
    line = line.strip()
    if line.startswith(PRINT_PREFIX):
        return bytes.fromhex(line[len(PRINT_PREFIX) :])
    else:
        #print(line)
        return None


def reader(ser):
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



def playing_sound(ser, SoundPerClasse, classes = None):
    classe = 'birds'
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

            ser.write(bytearray('s','ascii'))
            print("message sent to uart")
            #ser.close()

            time.sleep(7-sleeptime)
                  
if __name__ == "__main__":

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

        ser = serial.Serial(port = args.port, baudrate = 115200)
        dataset = Dataset()
        classes = dataset.list_classes()
        msg_counter = 0
        input_stream = reader(ser)
        for classe in classes:
            #classe = 'helicopter'
            SoundPerClasse = 5
            for i in range(SoundPerClasse):

                ###### envoi du son ######
                sound = dataset[classe, i]
                x, fs = sf.read(sound)
                # target_dB = 25
                # x /= np.linalg.norm(x) * 10 ** (-target_dB / 20)
                print(f'Playing a "{get_cls_from_path(sound)}"')
                sd.play(x, fs)

                sleeptime = random.uniform(0, 4)
                print("sleeping for:", sleeptime, "seconds")
                time.sleep(sleeptime)

                ser.write(bytearray('s','ascii'))
                print("message sent to uart")

                ###### recevoir du son ######
                buffer = None
                while buffer == None:
                    line = ""
                    while not line.endswith("\n"):
                        line += ser.read_until(b"\n", size=2 * N_MELVECS * MELVEC_LENGTH).decode(
                            "ascii"
                        )
                    print(line)
                    line = line.strip()
                    buffer = parse_buffer(line)
                    if buffer is not None:
                        melvec = np.frombuffer(buffer, dtype=dt)

                        
                #melvec = next(input_stream)
                msg_counter += 1

                print("MEL Spectrogram #{}".format(msg_counter))
                print(melvec.shape)

                melvec = np.reshape(melvec, (1, N_MELVECS * MELVEC_LENGTH))
                

                #enregistrement des melvecs de la vraie chaine de communication
                filename = "{}_{}".format(classe, msg_counter)
                pickle.dump(melvec, open(melvec_dir+filename, 'wb'))

                ##### plot #####
                # plt.figure()
                # plot_specgram(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, ax=plt.gca(), is_mel=True, title="MEL Spectrogram #{} \n Predicted class: {}".format(msg_counter, "glucie"), xlabel="Mel vector")
                # plt.draw()
                # plt.pause(0.001)
                # plt.show()

                time.sleep(7-sleeptime)


        ser.close()

            
            # plt.figure()
            # plot_specgram(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, ax=plt.gca(), is_mel=True, title="MEL Spectrogram #{} \n Predicted class: {}".format(msg_counter, y_predict), xlabel="Mel vector")
            # plt.draw()
            # plt.pause(0.001)
            # plt.show()
