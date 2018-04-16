import numpy as np
import matplotlib.pyplot as plt

a = np.fromfile("aa.bin",dtype=np.float32)
plt.plot(a)
plt.show()