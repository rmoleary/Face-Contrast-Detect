
# coding: utf-8

# ### First, detect the faces and eyes in an image.  
# Then calculate the contrast of the face and eyes to determine if the eyes are in focus.
# We are going to follow the opencv documentation to find the eyes in the image:
# http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

# In[2]:

get_ipython().magic(u'pylab inline')


# In[10]:

import cv2
import numpy as np
import time


# XML classifiers downloaded from opencv.org

# In[4]:

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


# In[81]:

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')


# Import the image and load grayscale version of it

# In[117]:

img = cv2.imread('imgs/20160306_IMG_2791p.JPG')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgmp =  cv2.cvtColor(img, cv2.cv.CV_BGR2RGB) #this is to plot images using matplotlib


# In[118]:

t1 = time.time()
#detect faces
faces = face_cascade.detectMultiScale(gray, 1.2, 5)
#this could be sped up by rescaling large images
t2 = time.time()
print "It took ", t2-t1, "seconds to complete faces."
for (x,y,w,h) in faces:
    #detect eyes within the face
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray,1.2,5)
t3 = time.time()
print "It took ", t3-t2, "seconds to complete eyes.\n\n"
print "Found ", len(faces), " faces and ", len(eyes), " eyes."


# In[119]:

#find largest face
index = (faces[:,3]*faces[:,2]).argmax()


# In[120]:

imshow(imgmp)
x,y,w,h = faces[index]
fill_between([x,x+w],[y,y],[y+h,y+h],color='None',edgecolor='w')
for (ex,ey,ew,eh) in eyes:
    fill_between([x+ex,x+ex+ew],[y+ey,y+ey],[y+ey+eh,y+ey+eh],color='None',edgecolor='r')
#axis([1500,3000,2000,500])
axis([x-2.5*w,x+3.5*w,y+3.5*h,y-2.5*h])


# In[121]:

#plot all faces
imshow(imgmp)
for (x,y,w,h) in faces:
    fill_between([x,x+w],[y,y],[y+h,y+h],color='None',edgecolor='w')


# In[90]:

roi_gray.shape


# In[ ]:



