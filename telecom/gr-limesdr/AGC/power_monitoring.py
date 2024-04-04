import socket
import matplotlib.pyplot as plt
import numpy as np
    
host = socket.gethostname()
port = 10004
server_socket = socket.socket() 

def reader_socket(server_socket, host, port): 

    server_socket.bind((host, port))

    server_socket.listen(1)
    conn, address = server_socket.accept()
    print("Connection from: " + str(address))

    while True:
        line = conn.recv(10).decode('ascii')
        
        yield line


input_stream = reader_socket(server_socket, host, port)


power_arr = np.ones(100) * 0.00001

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line, = ax.plot(power_arr)
ax.set_ylim(20, 30)
ax.set_ylabel("normalised RX power [dBm]")

for power in input_stream:
    print(np.log10(float(power)/1e-3))
    power_arr[:-1] = power_arr[1:]
    power_arr[-1]  = float(power)
    line.set_ydata(10 * np.log10(power_arr/1e-3))
    fig.canvas.draw()
    fig.canvas.flush_events()

