import numpy as np
import matplotlib.pyplot as plt

plot, ax = plt.subplots(1,1)
n = 5000000
a1 = np.random.rand(n) + 1
a2 = np.random.rand(n) + 1
s1 = np.fft.rfft(a1)
s2 = np.fft.rfft(a2)
s1[n//20:] = 0
s2[n//20:] = 0
b1 = np.fft.irfft(s1)
b2 = np.fft.irfft(s2)
b1 = (b1 - np.mean(b1)) / np.std(b1) * 150 + 960
b2 = (b2 - np.mean(b2)) / np.std(b2) * 80 + 600
ax.set_xlim(0,1920)
ax.set_ylim(0,1200)
line = ax.plot([])
b = np.concatenate((b1.reshape(-1,1),b2.reshape(-1,1)), 1).astype('int')
print(b.shape)
np.save('points.npy',b)


#for i in range(n):
#    if i <110:
#        continue 
#    line[0].set_xdata(b1[i-100:i])
#    line[0].set_ydata(b2[i-100:i])
#    plt.pause(0.030)
#
