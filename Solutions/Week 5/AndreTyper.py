# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:53:44 2022

@author: danie
"""

import skimage.io
import matplotlib.pyplot as plt
import numpy as np

#I = skimage.io.imread('C:/Users/danie/OneDrive/Desktop/02506/circly.png').astype(float)
I = np.array([[14,40,30,45,32,40],[33,32,40,50,23,20],[30,35,34,45,25,21],[31,25,40,52,20,27],[30,38,32,40,25,20]])
mu = np.array([30, 20], dtype=float)
beta = 125

S0 = np.array([[1,0,0,0,1,1],[1,0,0,0,1,1],[0,0,0,1,1,1],[0,0,0,0,1,1],[0,0,0,1,1,1]])


prior = beta * ((S0[1:,:]!=S0[:-1,:]).sum() + (S0[:,1:]!=S0[:,:-1]).sum()) 
likelihood = int(((mu[S0]-I)**2).sum())


fig, ax = plt.subplots(1,2)
ax[0].imshow(I, cmap=plt.cm.gray, vmin=0, vmax=255)
ax[1].imshow(S0, cmap=plt.cm.jet, vmin=0, vmax=2)
print(f'Answer: {(prior + likelihood)}')