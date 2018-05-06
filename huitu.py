import numpy as np
import matplotlib.pyplot as plt

rnn = np.fromfile("rnn.bin", dtype=np.float32)
adam = np.fromfile("adam.bin", dtype=np.float32)
x = range(len(adam))
print(x)
y= range(len(rnn))
plt.plot(y,rnn,'b')
plt.plot(x,adam,'r')
plt.savefig("contrast.jpg")
#plt.plot(rnn,'b')
#plt.plot(adam,'r')
#plt.show()