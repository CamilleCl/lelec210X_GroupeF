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
from classification.utils.audio_student import AudioUtil

from sklearn.preprocessing import LabelEncoder

# creating the socket
host = socket.gethostname()
port = 5002
server_socket = socket.socket() 

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

result_filename = "predicted_class.csv"
melvec_dir = "bigbigDataset/"

dt = np.dtype(np.uint16).newbyteorder("<")

model_dir = "modelCNN/" # where to save the models
filename = 'CNN.pickle'
label_name = "label_encoder.pickle"
model = pickle.load(open(model_dir + filename, 'rb'))
label_encoder = pickle.load(open(model_dir + label_name, 'rb'))

predict_threshold = 0.5 #threshold for garbage class
past_predictions = [] #liste où on vient mettre les proba des anciennes predictions
 
classnames = ['birds','chainsaw','fire','handsaw','helicopter']
start = None #start for time threshold
time_threshold = 2.5 # max time between 2 melspecs


#choisir le mode qu'on veut: enregistrer un dataset et/ou faire une classification
create_data = True
classif = False
plot_fig = False
plot_sound_melvec = False


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

    file = open(result_filename, 'w')
    file.close()

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
        #classes = dataset.list_classes()
        classes = ["helicopter"]
        input_stream = reader(ser)
        for classe in classes:
            #classe = 'helicopter'
            SoundPerClasse = 250
            for i in range(SoundPerClasse):

                ###### envoi du son ######
                sound = dataset[classe, i]
                x, fs = sf.read(sound)
                # target_dB = 25
                # x /= np.linalg.norm(x) * 10 ** (-target_dB / 20)
                print(f'Playing a "{get_cls_from_path(sound)}"')
                sd.play(x, fs)

                sleeptime = 1 #random.uniform(0, 4)
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

                print("MEL Spectrogram #{}".format(i))
                print(melvec.shape)

                melvec = np.reshape(melvec, (1, N_MELVECS * MELVEC_LENGTH))
                #melvec = np.reshape(melvec, (1, N_MELVECS, MELVEC_LENGTH, 1))
                print(f"melvec reshaped:{melvec.shape}")

                if classif:
                    melvec_normalized = melvec / np.linalg.norm(melvec, keepdims=True)

                    proba = model.predict(melvec_normalized)
                    print(proba)
                    y_predict = np.argmax(proba, axis=1) # the most probable class
                    print(y_predict)
                    y_predict = label_encoder.inverse_transform(y_predict)
                    print(y_predict)
                    #proba = model.predict_proba(melvec_normalized)

                    #take past predictions into account
                    # if (start == None):
                    #     start = time.time() #begin the time counter
                    # else:
                    #     stop = time.time()
                    #     delay = stop - start
                    #     if(delay > time_threshold):
                    #       past_predictions = [] #clear array of predictions
                    #       print(f"too long :-( : {delay} sec")
                    #     start = stop 
                    # past_predictions.append(proba)
                    
                    # if (len(past_predictions) > 1): 
                    #     weights = np.ones(len(past_predictions)) #weights of the predictions
                    #     avg_proba = np.average(past_predictions, axis = 0, weights = weights) #avg proba of all columns with higher weight for the present proba
                    #     y_predict = classnames[np.argmax(avg_proba)]
                    #     print(f"avg proba: {avg_proba}, predicted class: {y_predict}")

                    # #decide if sound is garbage
                    # if (np.max(proba) < predict_threshold):
                    #     y_predict = 'garbage'
                    #print(f"probabilities:{classnames}\n {proba[0]}")

                    print(f'predicted class: {y_predict}')

                    file = open(result_filename, 'a')
                    file.write(f"{y_predict}\n")
                    file.close()
                
                if create_data:
                    #enregistrement des melvecs de la vraie chaine de communication
                    filename = "{}_{}.pickle".format(classe, i)
                    pickle.dump(melvec, open(melvec_dir+filename, 'wb'))

                ##### plot #####
                if plot_sound_melvec:
                    audio = AudioUtil.open(sound)
                    sound_melspec = AudioUtil.melspectrogram(audio)
                    print(f"melspec shape: {sound_melspec.shape}")
                    delay = int(sleeptime*10)
                    print(f"melspec delay: {delay}")
                    sound_melspec = sound_melspec[:, delay: delay+20]
                    print(f"melspec shape: {sound_melspec.shape}")
                    plt.figure()
                    plot_specgram(sound_melspec, ax=plt.gca(), is_mel=True, title="", xlabel="Mel vector")
                    plt.draw()
                    plt.pause(0.001)
                    #plt.savefig("ideal_{}.pdf".format(classe))
                    plt.show()

                if plot_fig:
                    title = "" #"MEL Spectrogram #{} \n Predicted class: {}".format(i, "glucie")
                    plt.figure()
                    plot_specgram(melvec.reshape((N_MELVECS, MELVEC_LENGTH)).T, ax=plt.gca(), is_mel=True, title=title, xlabel="Mel vector")
                    plt.draw()
                    plt.pause(0.001)
                    #plt.savefig("nonideal_micro_{}.pdf".format(classe))
                    plt.show()

                time.sleep(7-sleeptime)


        ser.close()
