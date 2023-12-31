#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:07:32 2021

@author: vand
"""

import skimage.io
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt


def segmentation_energy(S, I, mu, beta):
    
    # likelihood energy
    V1 = ((mu[S]-I)**2).sum()
    
    # prior energy
    V2x = S[1:,:]!=S[:-1,:]
    V2y = S[:,1:]!=S[:,:-1]
    V2 = beta * (V2x.sum() + V2y.sum())
    
    return(V1,V2)

def segmentation_histogram(ax, I, S, edges=None):
    '''
    Histogram for data and each segmentation label.
    '''
    if edges is None:
        edges = np.linspace(I.min(), I.max(), 100)
    ax.hist(I.ravel(), bins=edges, color = 'k')
    centers = 0.5*(edges[:-1] + edges[1:]);
    for k in range(S.max()+1):
        ax.plot(centers, np.histogram(I[S==k].ravel(), edges)[0])
        

path = 'C:/Users/hujo8/OneDrive/Advanced image analysis/5 Week/'
I = skimage.io.imread(path + 'noisy_circles.png').astype(np.float)

segmentations = [] # list where I'll place different segmentations
GT = skimage.io.imread(path + 'noise_free_circles.png')

(mu, S_gt) = np.unique(GT, return_inverse=True)
S_gt = S_gt.reshape(I.shape)

segmentations += [S_gt]

#%% finding some configurations (segmentations) using conventional methods
S_t = np.zeros(I.shape, dtype=int) + (I>100) + (I>160) # thresholded
segmentations += [S_t]

D_s = scipy.ndimage.gaussian_filter(I, sigma=1, truncate=3, mode='nearest')
S_g = np.zeros(I.shape, dtype=int) + (D_s>100) + (D_s>160) # thresholded
segmentations += [S_g]

D_m = scipy.ndimage.median_filter(I, size=(5,5), mode='reflect');
S_t = np.zeros(I.shape, dtype=int) + (D_m>100) + (D_m>160) # thresholded
segmentations += [S_t]


#%% visualizattion
fig, ax = plt.subplots()
ax.imshow(I, vmin=0, vmax=255, cmap=plt.cm.gray)


N = len(segmentations)
fig, ax = plt.subplots(3,N)
beta = 100

for i in range(N):
    ax[0][i].imshow(segmentations[i])
    V1, V2 = segmentation_energy(segmentations[i], I, mu, beta)
    ax[0][i].set_title(f'likelihood: {int(V1)}\nprior: {V2}\nposterior: {int(V1+V2)}')
    
    segmentation_histogram(ax[1][i], I, segmentations[i])
    ax[1][i].set_xlabel('intensity')
    ax[1][i].set_ylabel('count')
    
    err = S_gt - segmentations[i]
    ax[2][i].imshow(err, vmin=-2, vmax=2, cmap=plt.cm.bwr)
    ax[2][i].set_title(f'error: {(err>0).sum()}')
    
    
    