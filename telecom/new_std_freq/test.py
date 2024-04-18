import serial
import os
import time

# to launch from the root of the project

ser = serial.Serial("/dev/ttyACM0",115200)

tx_power_list = list(range(-67, -55, 1))

repeat = 1

for tx_power in tx_power_list:
    for i in range(repeat):
        # launching gnuradio
        os.system(f"python3 ../hands_on_measurements/gr-fsk/apps/eval_limesdr_fpga.py >> Tx_{tx_power}dBm-{i}.txt &")

        # tx_power_pos = tx_power + 128
        tx_power_pos = tx_power + 128
        bytes_array = tx_power_pos.to_bytes(1, byteorder='big')

        # wait for the program to be ready
        time.sleep(15)

        # start sending packets at tx power lvl
        ser.write(bytes_array)

        # wait until all packets have been transmitted
        while("Packets sent" not in str(ser.readline())):
            continue

        # wait for the python program to end :(
        time.sleep(600)

    