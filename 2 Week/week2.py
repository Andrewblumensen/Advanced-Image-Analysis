# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:41:51 2022

@author: hujo8
"""

import numpy as np
import scipy.ndimage
import skimage.feature
import skimage.io
import matplotlib.pyplot as plt
import cv2


def gauss_kernels(sigma, size=4):
    s = np.ceil(np.max([sigma*size, size]))
    x = np.arange(-s,s+1)
    x = x.reshape(x.shape + (1,))
    g = np.exp(-x**2/(2*sigma*sigma))
    g /= np.sum(g)
    dg = -x/(sigma*sigma)*g
    ddg = -1/(sigma*sigma)*g -x/(sigma*sigma)*dg
    return g, dg, ddg

im = skimage.io.imread('CT_lab_high_res.png').astype(np.float)
fig, ax = plt.subplots(1)
ax.imshow(im)



#%% 2.1.1

im = skimage.io.imread('test_blob_uniform.png').astype(np.float)

fig, ax = plt.subplots(1)
ax.imshow(im)



#%% 2.1.2

im = skimage.io.imread('test_blob_uniform.png').astype(np.float)

t = 325

sigma = np.sqrt(t)
g, dg, ddg = gauss_kernels(sigma)


Lxx = cv2.filter2D(cv2.filter2D(im, -1 , g), -1, ddg.T)
Lyy = cv2.filter2D(cv2.filter2D(im, -1 , ddg), -1, g.T)

L_blob = t*(Lxx + Lyy)

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
pos = ax.imshow(L_blob)
fig.colorbar(pos)


magthres = 50

coordpos = skimage.feature.peak_local_max(L_blob, threshold_abs=magthres)
coordneg = skimage.feature.peak_local_max(-L_blob, threshold_abs=magthres)
coord = np.r_[coordpos, coordneg]


theta = np.arange(0, 2*np.pi, step=np.pi/100)
thetax = np.append(theta,0)
circ = np.array((np.cos(theta),np.sin(theta)))
n = coord.shape[0]
m = coord.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
plt.plot(coord[:,1],coord[:,0], '.r')

M = 1000
angle = np.exp(1j * 2 * np.pi / M)
angles = np.cumprod(np.ones(M + 1) * angle)


x = []
y = []

for i in range(n): 

        x.append(np.sqrt(2*t)*(np.real(angles))+coord[i,1])
        y.append(np.sqrt(2*t)*(np.imag(angles))+coord[i,0])
        plt.plot(x[i], y[i], 'r')

#%% 2.1.3


# im = skimage.io.imread('test_blob_uniform.png').astype(np.float)
# im = skimage.io.imread('test_blob_varying.png').astype(np.float)
im = skimage.io.imread('CT_lab_high_res.png').astype(np.float)

t = 0.5

sigma = np.sqrt(t)
g, dg, ddg = gauss_kernels(sigma)

im = cv2.filter2D(cv2.filter2D(im,-1,g), -1, g.T) 


t = 15

sigma = np.sqrt(t)
g, dg, ddg = gauss_kernels(sigma)

r,c = im.shape

n = 10

L_blob_vol = np.zeros((r,c,n+2))
tstep = np.zeros(n)

Lg = im

for i in range(0,n):
    tstep[i] = t*(i+1)
    L_blob_vol[:,:,i+2] = t*(i+1)*(cv2.filter2D(cv2.filter2D(Lg,-1,g), -1, ddg.T) + (cv2.filter2D(cv2.filter2D(Lg,-1,ddg), -1, g.T)))
    Lg = cv2.filter2D(cv2.filter2D(Lg,-1,g), -1, g.T)                              

thres = 8000.0


coordpos = skimage.feature.peak_local_max(L_blob_vol, min_distance = 2, threshold_abs=thres)
coordneg = skimage.feature.peak_local_max(-L_blob_vol, min_distance = 2, threshold_abs=thres)
coord = np.r_[coordpos, coordneg]


n = coord.shape[0]
m = coord.shape[1]

fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
ax.imshow(im)
plt.plot(coord[:,1],coord[:,0], '.r')


M = 1000
angle = np.exp(1j * 2 * np.pi / M)
angles = np.cumprod(np.ones(M + 1) * angle)


scale = tstep[coord[:,2]-2]

x = []
y = []
rad = 0
for i in range(n): 

        x.append(np.sqrt(2*scale[i])*(np.real(angles))+coord[i,1])
        y.append(np.sqrt(2*scale[i])*(np.imag(angles))+coord[i,0])
        rad = np.sqrt(2*scale[i]) + rad
        plt.plot(x[i], y[i], 'r')
        
print(rad/n)        

print(n)




