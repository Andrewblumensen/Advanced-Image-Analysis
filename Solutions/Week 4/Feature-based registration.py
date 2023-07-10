# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:35:13 2022

@author: danie
"""
#import matplotlib.pyplot as plt
import numpy as np
#%% QUESTION 6 *
s = 1.7
t = np.array([[36, 13]]).T
theta = 140/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])

#p = np.loadtxt('C:/Users/danie/OneDrive/Desktop/02506/points_p.txt')
#q = np.loadtxt('C:/Users/danie/OneDrive/Desktop/02506/points_q.txt')

p = [[1],[2]]
q =[[7],[8]]

p_ = R.T@(q-t)/s

# visualization
#fig,ax = plt.subplots(1,2)
#ax[0].plot(p[0], p[1], 'r.', q[0], q [1], 'b.')
#ax[0].plot(np.stack((p[0], q[0])), np.stack((p[1], q[1])), 'k', linewidth = 0.3)
#ax[0].set_aspect('equal')
#ax[1].plot(p[0], p[1], 'r.', p_[0], p_[1], 'b.')
#ax[1].plot(np.stack((p[0], p_[0])), np.stack((p[1],p_[1])),  'k', linewidth = 0.3)
#ax[1].set_aspect('equal')

d = np.sqrt(np.sum((p - p_)**2,axis=0))
print(f'Question 6: {np.sum(d>2)}')