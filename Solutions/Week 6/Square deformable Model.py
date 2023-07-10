# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:12:20 2022

@author: danie
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis
import scipy.interpolate
import scipy.linalg

#%% QUESTION 14 *
# Solution based on
# https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week06/week06_pycode/quiz_solution.py
# and
# https://lab.compute.dtu.dk/aia02506/weekly-solutions/-/blob/master/Week06/week06_pycode/plusplus_segmentation.py
I = skimage.io.imread('C:/Users/hujo8/OneDrive/Advanced image analysis/Old exam/frame.png').astype(float)/255
mask = np.zeros(I.shape, dtype=bool)
mask[I.shape[0]//2-40:I.shape[0]//2+40, I.shape[1]//2-40:I.shape[1]//2+40] = 1

m_in = np.mean(I[mask])
m_out = np.mean(I[~mask])

p = [I.shape[0]/2+39.5, I.shape[1]/2-40.5]
I_p = I[int(p[0]),int(p[1])]

Ein = np.sum((I[mask]-m_in)**2)
Eout = np.sum((I[~mask]-m_out)**2)

Eext = Ein + Eout

#force = 0.5*(m_in-m_out)*(2*I_p - (m_in+m_out))
force = (m_in - m_out) * (2*I_p - m_in - m_out)

# Visualization
fig, ax = plt.subplots()
rgb = 0.5*(np.stack((I,I,I), axis=2) + np.stack((mask,mask,0*mask), axis=2))
ax.imshow(rgb)
ax.plot(p[1], p[0], 'co',markersize=10)
print(f'Question 14: {force}')
