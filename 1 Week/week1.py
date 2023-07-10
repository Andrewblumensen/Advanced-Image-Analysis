# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 15:56:16 2022

@author: hujo8
"""

import cv2 as cv
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



#1.1
 
myImage = cv.imread(r'C:\Users\hujo8\OneDrive\Advanced image analysis\fibres_xcth.png',0)
# img1 = Image.fromarray(myImage)
# img1.show()



# Initializing value of x-axis and y-axis
# in the range -1 to 1
x, y = np.meshgrid(np.linspace(-1,1,6), np.linspace(-1,1,6))
dst = (x*x+y*y)
 
# Initializing sigma and muu
sigma = 1
 
# Calculating Gaussian array
gauss = 1/(2*sigma*np.pi) * np.exp(-(dst) / ( 2.0 * sigma ) )
 
gauss = gauss * 1/(np.sum(gauss))



g = ndimage.convolve(myImage,gauss)

# img = Image.fromarray(g)
# img.show()




# Initializing value of x-axis and y-axis
# in the range -1 to 1
x = np.linspace(-1,1,6)
dst = (x*x)
 
# Initializing sigma and muu
sigma = 1
 
# Calculating Gaussian array
gaussx = 1/np.sqrt(2*sigma*np.pi) * np.exp(-(dst) / ( 2.0 * sigma ) )
 
gaussx = np.reshape(gaussx * 1/(np.sum(gaussx)),[1,6])

gaussy = np.reshape(gaussx,[6,1])


g1d = ndimage.convolve(myImage,gaussx)
g1d = ndimage.convolve(g1d,gaussy)

# img = Image.fromarray(g1d)
# img.show()

diff = g-g1d

print(np.sum(diff))

# img = Image.fromarray(diff)
# img.show()



#1.2

div = np.reshape([0.5,0,-0.5],[1,3])

g = ndimage.convolve(myImage,div)

g = ndimage.convolve(g,gaussx)

img = Image.fromarray(g)
img.show()



x = np.linspace(-1,1,6)
dst = (np.power(x, 2))
 
# Initializing sigma and muu
sigma = 1
 
# Calculating div Gaussian array
gaussd = -(x) / (sigma**3*np.sqrt(2*np.pi)) * np.exp(-(dst) / ( 2.0 * sigma**2 ) )


g = ndimage.convolve(myImage,np.reshape(gaussd, [1,6]))


img = Image.fromarray(g)
img.show()




#1.3
 
# Initializing value of x-axis and y-axis
# in the range -1 to 1
x, y = np.meshgrid(np.linspace(-1,1,6), np.linspace(-1,1,6))
dst = (x*x+y*y)

# Initializing sigma and muu
sigma = 20
 
# Calculating Gaussian array
gauss20 = 1/(2*sigma*np.pi) * np.exp(-(x*x+y*y) / ( 2.0 * sigma ) )
 
gauss20 = gauss20 * 1/(np.sum(gauss20))


g = ndimage.convolve(myImage,np.reshape(gauss20,[6,6]))

# img = Image.fromarray(g)
# img.show()


# Initializing value of x-axis and y-axis
# in the range -1 to 1
x, y = np.meshgrid(np.linspace(-1,1,6), np.linspace(-1,1,6))
dst = (x*x+y*y)

sigma = 2
 
# Calculating Gaussian array
gauss2 = 1/(2*sigma*np.pi) * np.exp(-(x*x+y*y) / ( 2.0 * sigma ) )
 
gauss2 = gauss2 * 1/(np.sum(gauss2))

gauss2 = np.reshape(gauss2,[6,6])

temp = myImage 

for i in range(10):
    temp = ndimage.convolve(temp,gauss2)

# img = Image.fromarray(temp)
# img.show()



g = gaussian_filter(myImage, sigma=20)

# img = Image.fromarray(g)
# img.show()


temp = myImage 

for i in range(10):
    temp = gaussian_filter(temp, sigma=2)

# img = Image.fromarray(temp)
# img.show()
























