# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:00:02 2022

@author: danie
"""
import skimage.io
import matplotlib.pyplot as plt
import numpy as np

I = skimage.io.imread('C:/Users/hujo8/OneDrive/Advanced image analysis/Old exam/circly.png').astype(float)
#I = np.array([[1,2,6,4,10,8],[4,1,3,5,9,6],[5,2,3,5,4,7]])
mu = np.array([70, 120, 180], dtype=float)
beta = 100

U = (I.reshape(I.shape+(1,)) - mu.reshape(1,1,-1))**2
S0 = np.argmin(U, axis=2)

prior = beta * ((S0[1:,:]!=S0[:-1,:]).sum() + (S0[:,1:]!=S0[:,:-1]).sum()) 
likelihood = int(((mu[S0]-I)**2).sum())

fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap=plt.cm.gray, vmin=0, vmax=255)
ax[1].imshow(S0, cmap=plt.cm.jet, vmin=0, vmax=2)
print(f'Answer: {prior}')