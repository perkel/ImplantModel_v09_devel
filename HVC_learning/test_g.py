import numpy as np
import matplotlib.pyplot as plt
def g(t, n, tau):
    temp0 = np.power(t, n)
    temp1 = np.exp(-t/tau)
    return temp0*temp1


times = np.linspace(0, 500, num=501)
vals = g(times, 5, 5)


a = np.ones((5, 3))
b = np.zeros(3) + [1, 2, 4]

c = np.matmul(a, b)
d = np.multiply(a, b)

print (c, d)