"""
Anders Bjorholm Dahl, abda@dtu.dk

Script to match images based on SIFT features.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.ndimage
import transform

#%% Read two images to test the matching properties (default from documentation)
path = 'C:/Users/hujo8/OneDrive/Advanced image analysis/4 Week/' # Replace with your own path

#im1 = cv2.imread(path + 'CT_lab_high_res.png',cv2.IMREAD_GRAYSCALE) # queryImage
#im2 = cv2.imread(path + 'CT_lab_low_res.png',cv2.IMREAD_GRAYSCALE) # trainImage

im1 = cv2.imread(path + 'quiz_image_1.png',cv2.IMREAD_GRAYSCALE) # queryImage
im2 = cv2.imread(path + 'quiz_image_2.png',cv2.IMREAD_GRAYSCALE) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(im1,None)
kp2, des2 = sift.detectAndCompute(im2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    # Apply the Lowe criterion best should be closer than second best
    if m.distance/(n.distance + 10e-10) < 0.6:
        good.append([m])

# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(im1,kp1,im2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


#%% Match SIFT - function to extract the coordinates of matching points

# Parameters that can be extracted from keypoints
# keypoint.pt[0],
# keypoint.pt[1],
# keypoint.size,
# keypoint.angle,
# keypoint.response,
# keypoint.octave,
# keypoint.class_id,

def match_SIFT(im1, im2, thres = 0.6):
    
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
    pts_im1 = [kp1[m[0].queryIdx].pt for m in good_matches]
    pts_im1 = np.array(pts_im1, dtype=np.float32).T
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good_matches]
    pts_im2 = np.array(pts_im2, dtype=np.float32).T
    return pts_im1, pts_im2


#%% Create a test image - Rotate, scale and crop image

ang = 67
sc = 0.6
imr = scipy.ndimage.rotate(scipy.ndimage.zoom(im1,sc),ang,reshape=False)[50:-50,50:-50]
plt.imshow(imr)

#%% Plot the keypoints with very low threshold (0.1) to see what is going on

pts_im1, pts_im2 = match_SIFT(im1, imr, 0.1)

def plot_matching_keypoints(im1, im2, pts_im1, pts_im2):
    r1,c1 = im1.shape
    r2,c2 = im2.shape
    n_row = np.maximum(r1, r2)
    n_col = c1 + c2
    im_comp = np.zeros((n_row,n_col))
    im_comp[:r1,:c1] = im1
    im_comp[:r2,c1:(c1+c2)] = im2
    
    fig,ax = plt.subplots(1)
    ax.imshow(im_comp, cmap='gray')
    ax.plot(pts_im1[0],pts_im1[1],'.r')
    ax.plot(pts_im2[0]+c1,pts_im2[1],'.b')
    ax.plot(np.c_[pts_im1[0],pts_im2[0]+c1].T,np.c_[pts_im1[1],pts_im2[1]].T,'w',linewidth = 0.5)

plot_matching_keypoints(im1,imr,pts_im1, pts_im2)

#%% Plot the keypoints between image 1 and image 2

pts_im1, pts_im2 = match_SIFT(im1, im2, 0.6)
plot_matching_keypoints(im1,im2,pts_im1, pts_im2)

#%% Compute the transformaions and plot them on top of each other

R,t,s = transform.get_transformation(pts_im2,pts_im1)

print(f'Transformations: Rotation:\n{R}\n\nTranslation:\n{t}\n\nScale: {s}\n\nAngle: {np.arccos(R[0,0])/np.pi*180}')
pts_im1_1 = s*R@pts_im2 + t

fig,ax = plt.subplots(1)
ax.imshow(im1)
ax.plot(pts_im1[0],pts_im1[1],'.r')
ax.plot(pts_im1_1[0],pts_im1_1[1],'.b')

#%% Robust transformation - kept points are shown in cyan and magenta

R,t,s,idx = transform.get_robust_transformation(pts_im2,pts_im1)

pts_im1_2 = s*R@pts_im2 + t

fig,ax = plt.subplots(1)
ax.imshow(im1, cmap = 'gray')
ax.plot(pts_im1[0],pts_im1[1],'.b')
ax.plot(pts_im1_2[0],pts_im1_2[1],'.r')
ax.plot(pts_im1[0,idx],pts_im1[1,idx],'.c')
ax.plot(pts_im1_2[0,idx],pts_im1_2[1,idx],'.m')





