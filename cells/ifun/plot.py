# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:12:01 2013

@author: chris
"""
import _ifun
import numpy as np
import matplotlib.pyplot as plt

I = np.ones(2000)*1
I[1000:1010] = 2

np.random.seed(1)
r = np.random.rand(1000*1000)

print r

z = _ifun.ifun(r, 1000, 2000, 0.1, 100, 0.8, I)

z = np.array(z).reshape((2000,1000))
print np.shape(z)

plt.subplot(2,1,1)
plt.plot(z[:,1], 'r')
plt.plot(z[:,2], 'b')
plt.plot(z[:,3], 'g')
plt.subplot(2,1,2)
plt.plot(z)
plt.show()