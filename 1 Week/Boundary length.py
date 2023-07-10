# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:21:07 2022

@author: hujo8
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt

def boundary_length(S):
    lx = S[1:,:]!=S[:-1,:]
    ly = S[:,1:]!=S[:,:-1]
    L = np.sum(lx)+np.sum(ly)
    return L

fig, ax = plt.subplots(1,3)
path = 'fuel_cells/'

for i in range(3):
    I = skimage.io.imread(f'{path}fuel_cell_{i+1}.tif')
    L = boundary_length(I)
    ax[i].imshow(I)
    ax[i].set_title(f'L={L}')
