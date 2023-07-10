# -*- coding: utf-8 -*-
"""
Created on Sat May 21 14:56:24 2022

@author: danie
"""

import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import simple_snake as sis
import scipy.interpolate
import scipy.linalg


filename = 'C:/Users/hujo8/OneDrive/Advanced image analysis/6 Week/plusplus.png'
I = skimage.io.imread(filename).astype(np.float)
I = np.mean(I,axis=2)/255

nr_points = 100
nr_iter = 1
step_size = 5
alpha = 0.01
beta = 0.1

center = np.array(I.shape)/2
radius = 180


snake = sis.make_circular_snake(nr_points, center, radius)
B = sis.regularization_matrix(nr_points, alpha, beta)

mask = skimage.draw.polygon2mask(I.shape, snake.T)
m_in = np.mean(I[mask])
m_out = np.mean(I[~mask])

Ein = np.sum((I[mask]-m_in)**2)
Eout = np.sum((I[~mask]-m_out)**2)

Eext = Ein + Eout

f = scipy.interpolate.RectBivariateSpline(np.arange(I.shape[0]), np.arange(I.shape[1]), I)
val = f(snake[0],snake[1], grid=False)
    # val = I[snake[0].astype(int), snake[1].astype(int)] # simpler variant without interpolation
#force = 0.5*(m_in-m_out)*(2*val - (m_in+m_out))
force = (m_in - m_out) * (2*val - m_in - m_out)
fig, ax = plt.subplots()
ax.imshow(I, cmap=plt.cm.gray)
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
ax.set_title('Initialization')

snake = sis.evolve_snake(snake, I, B, step_size)
ax.clear()
ax.imshow(I, cmap=plt.cm.gray)
ax.plot(np.r_[snake[1],snake[1,0]],np.r_[snake[0],snake[0,0]],'r-')
ax.set_title(f'iteration 1')