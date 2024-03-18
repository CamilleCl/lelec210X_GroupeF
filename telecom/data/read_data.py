import numpy as np
import matplotlib.pyplot as plt

def curve_averaging(data, start, stop, n, n_remove):
    x = []
    y = []
    db = np.linspace(start, stop, n)
    for i in range(len(db) - 1):
        errors = 0
        packets = 0
        for j in range(len(data[:, 0])):
            if db[i] <= data[j, 0] <= db[i+1]:
                errors  += (data[j,1] - n_remove)
                packets += (data[j, 2] - n_remove)

        if packets != 0:
            y.append(errors/packets)
            x.append((db[i] + db[i+1]) / 2)

    return x,y

data_base = np.loadtxt("data_base.csv", delimiter=',')
data_new  = np.loadtxt("data_new2.csv",  delimiter=',')

data_base_sto = np.loadtxt("data_base_sto.csv", delimiter=',')
data_new_sto_cfo_fin = np.loadtxt("data_new_sto_cfo_fin.csv", delimiter=',')
data_new_sto = np.loadtxt("data_new_sto.csv", delimiter=',')
data_sync = np.loadtxt("data_sync.csv", delimiter=',')

# plot 1
#plt.scatter(data_base[:,0], (data_base[:,1] - 5) / (data_base[:,2] - 5))

# plot 2
#plt.scatter(data_new[:,0], (data_new[:,1] - 5) / (data_new[:, 2] - 5))

# plot 3
#plt.scatter(data_base_sto[:,0], (data_base_sto[:,1] - 5) / (data_base_sto[:,2] - 5))

# plot 4
#plt.scatter(data_new_sto[:,0], (data_new_sto[:,1] - 5) / (data_new_sto[:, 2] - 5))
#plt.scatter(data_new_sto_cfo[:,0], (data_new_sto_cfo[:,1] - 5) / (data_new_sto_cfo[:, 2] - 5))

x_base, y_base = curve_averaging(data_base, -10, 15, 50, 5)
x_new, y_new = curve_averaging(data_new, -10, 15, 50, 5)
x_new_sync, y_new_sync = curve_averaging(data_sync, -10, 15, 50, 3)

plt.plot(x_base, y_base, label='baseline')
plt.plot(x_new, y_new, label='df=25kHz')
plt.plot(x_new_sync, y_new_sync, label='df=25kHz, new sync')

#sim_base = np.loadtxt('PER-sim-df12500.csv')
sim_new  = np.loadtxt('PER-sim-df25000.csv')

#plt.plot(sim_base[0], sim_base[1])
#plt.plot(sim_new[0], sim_new[1])

#np.savetxt('PER-df12500.csv', np.hstack([x_base, y_base]))
#np.savetxt('PER-df25000.csv', np.hstack([x_new, y_new]))

plt.xlabel("SNR [dB]")
plt.ylabel("PER [-]")

plt.yscale('log')
plt.xlim((-5, 15))
plt.legend()
plt.grid(alpha=0.5)
plt.show()