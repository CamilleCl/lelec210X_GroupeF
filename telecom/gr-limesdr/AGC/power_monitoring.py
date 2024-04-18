import socket
import matplotlib.pyplot as plt
import numpy as np
import time
    
host = socket.gethostname()
port = 10020
server_socket = socket.socket() 

def reader_socket(server_socket, host, port): 

    server_socket.bind((host, port))

    server_socket.listen(1)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))

    while True:
        line = conn.recv(12).decode('ascii')
        time.sleep(0.1)
        yield line


input_stream = reader_socket(server_socket, host, port)


power_arr = np.ones(25) * 0.00001

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(power_arr)
ax.set_ylim(0, 40)
ax.set_ylabel("normalised RX power [dBm]")

for power in input_stream:
    print(power)
    s = power.split(".")
    power_val = float(s[0] + "." + s[1])
    # print(np.log10(float(power)/1e-3))
    power_arr[:-1] = power_arr[1:]
    power_arr[-1]  = power_val
    line.set_ydata(10 * np.log10(power_arr/1e-3))
    ax.set_title(f"mean RX power: {np.mean(10 * np.log10(power_arr/1e-3))}")
    fig.canvas.draw()
    fig.canvas.flush_events()

