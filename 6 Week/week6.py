# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import skimage.io
import skimage.draw
import numpy as np
import matplotlib.pyplot as plt
# import easy_snake as es

#%% EXPLAINING THE BASIC PRINCIPLE OF WORKING WITH SNAKE

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('C:/Users/hujo8/OneDrive/Gamle kurser/Advanced image analysis/6 Week/plusplus.png')   
gray = rgb2gray(img)  
   
plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.show()

temp = 0
inx = 0

for i in range(588):
    for j in range(588):
        
        if (i - 294)**2 + (j - 294)**2 <= 180**2:
            temp = temp + gray[i,j]
            inx = inx + 1
        
gind = temp/inx


omegaind = 0

for i in range(588):
    for j in range(588):
        
        if (i - 294)**2 + (j - 294)**2 <= 180**2:
            omegaind = omegaind + (gray[i,j]-gind)**2


temp = 0
inx = 0

for i in range(588):
    for j in range(588):
        
        if (i - 294)**2 + (j - 294)**2 > 180**2:
            temp = temp + gray[i,j]
            inx = inx + 1
        
gud = temp/inx


omegaud = 0

for i in range(588):
    for j in range(588):
        
        if (i - 294)**2 + (j - 294)**2 > 180**2:
            omegaud = omegaud + (gray[i,j]-gud)**2

ext = omegaind + omegaud





# coord = np.array([[10,10]])

# t = 40

# theta = np.arange(0, 2*np.pi, step=np.pi/100)
# theta = np.append(theta, 0)
# circ = np.array((np.cos(theta),np.sin(theta)))
# n = coord.shape[0]
# m = circ.shape[1]


# circ_y = t*np.reshape(circ[0,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,0],(-1,1)).T
# circ_x = t*np.reshape(circ[1,:],(1,-1)).T*np.ones((1,n)) + np.ones((m,1))*np.reshape(coord[:,1],(-1,1)).T


# for j in range(10):
#     G = [circ_y[1]-circ_y[200]]
    
#     M = [circ_x[1]-circ_x[200]]
    
    
#     for i in range(np.size(theta)-2):
    
#         G.append(circ_y[i+2]-circ_y[i])
    
#         M.append(circ_x[i+2]-circ_x[i])
    
#     G.append(circ_y[0]-circ_y[199])
    
#     M.append(circ_x[0]-circ_x[199])
    
#     G = np.array(G)*-1
#     M = np.array(M)
    
#     N = np.column_stack((G,M))
    
#     for i in range(np.size(N,0)):
    
#         N[i,:] = N[i,:]/np.sqrt(N[i,0]**2+N[i,1]**2)
    
#     circ_y = circ_y + N[:,1]
#     circ_x = circ_x + N[:,0]
#     fig, ax = plt.subplots(1,1,figsize=(10,10),sharex=True,sharey=True)
#     plt.plot(coord[:,1], coord[:,0], '.r')
#     plt.plot(circ_x, circ_y, 'r')




