
# coding: utf-8

# ### First, detect the faces and eyes in an image.  
# Then calculate the contrast of the face and eyes to determine if the eyes are in focus.
# We are going to follow the opencv documentation to find the eyes in the image:
# http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0

# In[1]:

get_ipython().magic(u'pylab inline')


# In[5]:

import cv2
import numpy as np


# In[ ]:

face_cascade = cv2.CascadeClassifier()

