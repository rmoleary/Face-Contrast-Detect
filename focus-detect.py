#!/usr/bin/env python

# ### First, detect the faces and eyes in an image.  
# Then calculate the contrast of the face and eyes to determine if the eyes are in focus.
# The face detection algorithm in opencv, the haar cascade, gives too many false positives.  Instead we are going to rewrite the script to use dlib's histogram of gradients (HOG) detector.  
# 
# We are going to follow dlib's public domain example algorithm to detect faces in the image.

# In[1]:

#get_ipython().magic(u'pylab inline')


# In[4]:
from pylab import *
import cv2
import numpy as np
import time
import dlib
import sys


def get_right_eye(shape):
    xpts = []
    ypts = []
    
    for i in arange(36,42):
        xpts.append(shape.part(i).x)
        ypts.append(shape.part(i).y)
    xpts = array(xpts)
    ypts = array(ypts)
    dx = xpts.max() - xpts.min()
    dy = ypts.max() - ypts.min()
    return xpts.min(), ypts.min(), dx,  dy

def get_left_eye(shape):
    xpts = []
    ypts = []
    
    for i in arange(42,48):
        xpts.append(shape.part(i).x)
        ypts.append(shape.part(i).y)
    xpts = array(xpts)
    ypts = array(ypts)
    dx = xpts.max() - xpts.min()
    dy = ypts.max() - ypts.min()
    return xpts.min(), ypts.min(), dx,dy


def get_contrast_peak(fn):
    img = cv2.imread(fn)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #

    
    faces = detector(gray)
    if(len(faces) == 0 ):
        #did not detect faces. This is slightly slower
        imgmp =  cv2.cvtColor(img, cv2.cv.CV_BGR2RGB) #this is to plot images using matplotlib
        faces = detector(imgmp,1)
    if(len(faces) == 0 ):
        #now try rotation
        print "#Try Rotation"
        NN = 5
        rows,cols = gray.shape
        for j,i in enumerate(linspace(360./NN,360*(NN-1)*1./NN,NN)):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),i,1)
            dst = cv2.warpAffine(img,M,(cols,rows))
            faces = detector(dst)
            if(len(faces)>0):
                img = dst
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                imgmp =  cv2.cvtColor(img, cv2.cv.CV_BGR2RGB) #this is to plot images using matplotlib
                break
    if(len(faces) == 0):
        #still found no faces
        return -1.0

    fshapes = []
    areas = zeros(len(faces))    
    for k, d in enumerate(faces):
        shape = predictor(gray, d)
        fshapes.append(shape)
        areas[k] = d.area()

    if(max(areas) < 50*50):
        #face too small
        return -1.0
    index = (areas).argmax()
    shape = fshapes[index]
    
    f1 = faces[index]
    shape = fshapes[index]
    eyes = ([get_left_eye(shape),get_right_eye(shape)])
    #calculate contrast inside face region
    #We use a Laplacian filter to estimate the amount of contrast in the image following http://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    
    faceimg = gray[f1.top():f1.bottom(), f1.left():f1.right()]
    lap_faceimg = cv2.Laplacian(faceimg, 8)
    eyestd = []
    eyeimgs = []
    for (ex,ey,ew,eh) in array(eyes):
        eyeimg = gray[ey-20:ey+eh+20, ex-20:ex+ew+20]
        eyeimgs.append(eyeimg)
        lap_eyeimg = cv2.Laplacian(eyeimg, 8)
        eyestd.append( lap_eyeimg.std())
        #print "Face std", lap_faceimg.std(), " and eye stds ", eyestd, per99, percentile(lap_faceimg,99.9)


    print "# Face contrast: ", lap_faceimg.std(), "eyes: ", eyestd
    #Occasionally a single eye has a flash or spot in it giving it a large contrast.  Only use if eyes are within 50%
    
    if ((ew*eh < 100) & (lap_faceimg.std() > max(eyestd))):
        return  lap_faceimg.std()
    if(max(eyestd) > 1.75 * min(eyestd)):
        #Occasionally a single eye has a flash or spot in it giving it a large contrast.  Only use if eyes are within 50%
        return lap_faceimg.std()
        
    return max(eyestd)








# Import the images and load grayscale version of it

if(len(sys.argv) < 2):
    print "run as ./focus-detect.py file1 file2 ..."
    quit()

files = sys.argv[1:]

detector = dlib.get_frontal_face_detector()
predictor_path =  'facepredictor/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)

stds = zeros(len(files))
for i,fn in enumerate(files):
#    try:
    print "#Analyzing ", fn
    stds[i] = get_contrast_peak(fn)
    print "#        Contrast:  ", stds[i]
 #   except:
  #      print "failure in file", fn

ngood = stds[stds>0].size
index_best = stds.argmax()

print "#Detected faces in ", ngood, " of ", stds.size, "faces."
print "#Highest contrast picture was \n", files[index_best]

        
