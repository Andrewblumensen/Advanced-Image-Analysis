# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 14:32:24 2022

@author: hujo8
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.transform
import cv2

#%% READ IN IMAGES

random.seed(10)
n = 20

x = []
y = []
for i in range(0,n):
    g = random.randint(0,100)
    x.append(g)

for i in range(0,n):
    g = random.randint(0,100)
    y.append(g)


theta = (45/180)*np.pi
t = np.array([100,100])
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
s = 2
P = np.concatenate((x, y))

P = np.reshape(P, (2, n))
          
Q = []         
for i in range(0,n):

    g = (s*(np.dot(R,P[:,i])))+t
    Q.append(g)

Q = np.array(Q)



cq = np.array([[np.sum(Q[:,0])/n],[np.sum(Q[:,1])/n]])

cp = np.array([[np.sum(P[0,:])/n],[np.sum(P[1,:])/n]])



ax = plt.gca()

ax.scatter(P[1,:],P[0,:], color="b")
ax.scatter(Q[:,1],Q[:,0], color="r")
ax.scatter(cq[1],cq[0], color="g")
ax.scatter(cp[1],cp[0], color="g")

#%% READ IN IMAGES
P = P.astype(np.float)

distp = np.linalg.norm(P-cp)


distq = np.linalg.norm(Q-cq.T)


sm =  distq / distp


C = (Q.T-cq)@(P-cp).T

U,sig,Vt = np.linalg.svd(C)

Rh = U@Vt

D = np.array([[1,0],[0,np.linalg.det(Rh)]])

Rm = Rh@D

tm = cq - sm*Rm@cp





#%% Read in image

in_dir = 'c:/Users/hujo8/OneDrive/Advanced image analysis/4 Week/'

im1 = skimage.io.imread (in_dir + 'quiz_image_1.png')
im2 = skimage.io.imread (in_dir + 'quiz_image_2.png')

fig, ax = plt.subplots(1,2)
ax[0].imshow(im1,cmap='gray')
ax[1].imshow(im2,cmap='gray')


#%% Detect keypoints and show
# Initiate SIFT detector

sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(im1,None)
kp2,des2 = sift.detectAndCompute(im2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
for m,n in matches:
    # Apply the Lowe criterion best should be closer than second best
    if m.distance/(n.distance + 10e-10) < 0.6:
        good.append([m])
        

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

#%%
thres = 0.1
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(im1,None)


print(f'Coordinates: {kp1[0].pt}')
print(f'Response: {kp1[0].response}')
print(f'Size: {kp1[0].size}')

#%% Plot keypoints

# show keypoints
pts_im1 = np.asarray([kp1[i].pt for i in range(len(kp1))])
fig,ax = plt.subplots(1)
ax.imshow(im1, cmap='gray')
ax.plot(pts_im1[:,0],pts_im1[:,1],'r.')

#% Match keypoints and show
thres = 0.6
# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good_matches = []

for m,n in matches:
    if m.distance/(n.distance+10e-10) < thres:
        good_matches.append([m])

# Find coordinates
pts_im1 = [kp1 [m[0].queryIdx].pt for m in good_matches]
pts_im1 = np.array(pts_im1, dtype=np.float32).T
pts_im2 = [kp2 [m [0].trainIdx].pt for m in good_matches]
pts_im2 = np.array(pts_im2, dtype=np.float32).T

r1, c1 = im1.shape
r2, c2 = im2.shape
n_row = np.maximum(r1,r2)
n_col = c1 + c2
im_comp = np.zeros((n_row,n_col))
im_comp[:r1,:c1] = im1
im_comp[:r2,c1:(c1+c2)] = im2

fig,ax = plt.subplots(1)
ax.imshow(im_comp, cmap='gray')
ax.plot(pts_im1[0],pts_im1[1],'.r')
ax.plot(pts_im2[0]+c1,pts_im2[1],'.b')
ax.plot(np.c_[pts_im1[0],pts_im2[0]+c1].T,np.c_[pts_im1[1],pts_im2[1]].T,'w',linewidth = 0.5)


#%%

P = pts_im1 

Q = pts_im2

P = P.astype(np.float)


cq = np.array([[np.sum(Q[:,0])/n],[np.sum(Q[:,1])/n]])

cp = np.array([[np.sum(P[0,:])/n],[np.sum(P[1,:])/n]])

distp = np.linalg.norm(P-cp)


distq = np.linalg.norm(Q-cq.T)


sm =  distq / distp


C = (Q.T-cq)@(P-cp).T

U,sig,Vt = np.linalg.svd(C)

Rh = U@Vt

D = np.array([[1,0],[0,np.linalg.det(Rh)]])

Rm = Rh@D

tm = cq - sm*Rm@cp


#%% Quiz


Pq = pts_im1 

Qq = pts_im2

n = 653

cqq = np.array([[np.sum(Qq[0,:])/n],[np.sum(Qq[1,:])/n]])

cpq = np.array([[np.sum(Pq[0,:])/n],[np.sum(Pq[1,:])/n]])



ax = plt.gca()

ax.scatter(Pq[1,:],Pq[0,:], color="b")
ax.scatter(Qq[1,:],Qq[0,:], color="r")
ax.scatter(cqq[1],cqq[0], color="g")
ax.scatter(cpq[1],cpq[0], color="g")

Pq = Pq.astype(np.float)

distpq = np.linalg.norm(Pq-cpq)


distqq = np.linalg.norm(Qq-cqq)


smq =  distqq / distpq


Cq = (Qq-cqq)@(Pq-cpq).T

Uq,sigq,Vtq = np.linalg.svd(Cq)

Rhq = Uq@Vtq

Dq = np.array([[1,0],[0,np.linalg.det(Rhq)]])

Rmq = Rhq@Dq

tmq = cqq - smq*Rmq@cpq





