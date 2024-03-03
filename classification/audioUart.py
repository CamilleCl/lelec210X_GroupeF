"""
For running

-> poetry run python audioUart.py in lelec210X_GroupeF/classifications folder
If no port specified : give a list of port available
then poetry run python audioUart.py -p PORT?
"""
import librosa  # For audio signal computations as MFCC
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import sounddevice as sd
import soundfile as sf
import serial
from serial.tools import list_ports
import argparse
from scipy import signal
from scipy.fftpack import dct

import time

from classification.datasets import Dataset, get_cls_from_path
from classification.utils.plots import plot_audio, plot_specgram

SoundPerClasse = 20

def playing_sound(SoundPerClasse, port = None, classes = None):
    classe = 'birds'
    for i in range(SoundPerClasse):
            sound = dataset[classe, i]
            x, fs = sf.read(sound)
            # target_dB = 25
            # x /= np.linalg.norm(x) * 10 ** (-target_dB / 20)
            print(f'Playing a "{get_cls_from_path(sound)}"')
            
            sleeptime = random.uniform(0, 4)
            print("sleeping for:", sleeptime, "seconds")
            sleep(sleeptime)

            ser = serial.Serial(port = port, baudrate = 115200)
            ser.write(bytearray('s','ascii'))
        
            sd.play(x, fs)

            time.sleep(7-sleeptime)

argParser = argparse.ArgumentParser()
argParser.add_argument("-p", "--port", help="Port for serial communication")
args = argParser.parse_args()
print("Audio uart launched...\n")

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
    dataset = Dataset()
    classes = dataset.list_classes()
    playing_sound(SoundPerClasse, port=args.port, classes = classes)