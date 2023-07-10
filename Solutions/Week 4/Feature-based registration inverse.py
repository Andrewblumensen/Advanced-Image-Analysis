# -*- coding: utf-8 -*-
"""
Created on Tue May 24 12:03:54 2022

@author: hujo8
"""

# -*- coding: utf-8 -*-
"""
Anders Bjorholm Dahl, abda@dtu.dk

Script to develop function to estimate transformation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt


#%% Generate and plot point sets

n = 30
p = np.loadtxt('C:/Users/hujo8/OneDrive/Advanced image analysis/Old exam/points_p.txt')

angle = 140
theta = angle/180*np.pi
R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
t = np.array([[36],[13]])
s = 1.7

q = s*R@p + t

qc = np.loadtxt('C:/Users/hujo8/OneDrive/Advanced image analysis/Old exam/points_q.txt')



d = np.sqrt(np.sum((qc-q)**2,axis=0))


fig,ax = plt.subplots(1)
ax.plot(p[0],p[1],'r.')
ax.plot(q[0],q[1],'b.')
ax.plot(np.c_[p[0],q[0]].T,np.c_[p[1],q[1]].T,'g',linewidth=0.8)
ax.set_aspect('equal')

#%% Compute parameters

m_p = np.mean(p,axis=1,keepdims=True)
m_q = np.mean(q,axis=1,keepdims=True)
s1 = np.linalg.norm(q-m_q)/np.linalg.norm(p-m_p)
np.linalg.norm(q-m_q,ord=2)/np.linalg.norm(p-m_p,ord=2)
print(s1)

C = (q-m_q)@(p-m_p).T
U,S,V = np.linalg.svd(C)

R_ = U@V
R1 = R_@np.array([[1,0],[0,np.linalg.det(R_)]])
print(R1)

t1 = m_q - s1*R1@m_p
print(t1)
