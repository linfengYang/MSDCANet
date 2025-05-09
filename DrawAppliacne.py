import numpy as np
import matplotlib.pyplot as plt


data1 = np.loadtxt('D:/NILMdataset/ukdale/house_1/channel_6.dat', delimiter=' ')[:100000, 1]
data2 = np.loadtxt('D:/NILMdataset/ukdale/house_2/channel_13.dat', delimiter=' ')[:100000, 1]
data3 = np.loadtxt('D:/NILMdataset/ukdale/house_5/channel_22.dat', delimiter=' ')[:100000, 1]
data4 = np.loadtxt('D:/NILMdataset/REDD_datasets/REDD/low_freq/house_1/channel_6.dat', delimiter=' ')[:100000, 1]
data5 = np.loadtxt('D:/NILMdataset/REDD_datasets/REDD/low_freq/house_2/channel_10.dat', delimiter=' ')[:100000, 1]
data6 = np.loadtxt('D:/NILMdataset/REDD_datasets/REDD/low_freq/house_3/channel_9.dat', delimiter=' ')[:100000, 1]

plt.figure(figsize=(10, 6))
plt.plot(data1[:100000], label='File 1')
plt.plot(data2[:100000], label='File 2')
plt.plot(data3[:100000], label='File 3')
plt.plot(data4[:100000], label='File 4')
plt.plot(data5[:100000], label='File 5')
plt.plot(data6[:100000], label='File 6')


plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Second Column Data from .dat Files')
plt.legend()
plt.show()
