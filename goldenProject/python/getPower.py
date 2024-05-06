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
import time

import requests
import json

PRINT_PREFIX = "DF:HEX:"
FREQ_SAMPLING = 10200
MELVEC_LENGTH = 20
N_MELVECS = 20

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
            if "Power" in line:
                print(line)
                  
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
            input_stream = reader(port=args.port)

            msg_counter = 0

            for melvec in input_stream:
                continue
                
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Shutting down server")
        server_socket.close()
        exit(0)