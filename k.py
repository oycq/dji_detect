import numpy as np
a = np.array([[[2,5]]])
b = np.array([[20,50]])
print(a.shape)
print(b.shape)
print(a-b)
print(np.mean(a-b).abs())
