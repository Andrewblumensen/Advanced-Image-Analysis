# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 12:53:26 2022

@author: hujo8
"""

import cv2 as cv
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries, find_boundaries
from scipy.linalg import circulant

## Opg 1


myImage = cv.imread(r'C:\Users\hujo8\OneDrive\Advanced image analysis\noisy_number.png',0)
# img1 = Image.fromarray(myImage)
# img1.show()



# Initializing value of x-axis and y-axis
# in the range -1 to 1
x, y = np.meshgrid(np.linspace(-1,1,13), np.linspace(-1,1,13))
dst = (x*x+y*y)
 
# Initializing sigma and muu
sigma = 1
 
# Calculating Gaussian array
gauss = 1/(2*sigma*np.pi) * np.exp(-(dst) / ( 2.0 * sigma ) )
 
# gauss = gauss * 1/(np.sum(gauss))



g = ndimage.convolve(myImage,gauss)

# img = Image.fromarray(g)
# img.show()

#ans 13


## Opg 2

myImage = cv.imread(r'C:\Users\hujo8\OneDrive\Advanced image analysis\fuel_cells\fuel_cell_1.tif',0)
# img = Image.fromarray(myImage)
# img.show()

imgb = find_boundaries(myImage, mode='outer').astype(np.uint8)

print(np.sum(imgb))

# imgb = Image.fromarray(imgb*255)
# imgb.show()

#ans 21847

## Opg 3

with open('curves/dino_noisy.txt') as f:
    array = []
    for line in f: # read rest of lines
        array.append([float(x) for x in line.split()])

X = np.reshape(array,[200,2]) 

fig1, ax1 = plt.subplots()

plt.plot(X[:,0], X[:,1], color='black')


I = np.identity(len(X))

Lambda = 0.25


g = np.zeros(200)

g[0] = -2
g[1] = 1
g[199] = 1

L = circulant(g)

Xnew = np.dot((I + Lambda * L),X)

fig2, ax1 = plt.subplots()

plt.plot(Xnew[:,0], Xnew[:,1], color='black')

temp = np.sqrt((Xnew[199,0]-Xnew[0,0])**2+(Xnew[199,1]-Xnew[0,1])**2)

for i in range(199):
    temp = np.sqrt((Xnew[i,0]-Xnew[i+1,0])**2+(Xnew[i,1]-Xnew[i+1,1])**2) + temp

print(temp)




