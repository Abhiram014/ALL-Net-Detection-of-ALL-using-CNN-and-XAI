#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model


# In[2]:


import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from lime import lime_image
from skimage.segmentation import mark_boundaries 


# In[3]:


# Load the trained model
model = load_model('cnn10 augmentation.h5')


# In[19]:


# Load an example image
img_path = 'ALL dataset without augmentation\Pro\WBC-Malignant-Pro-173.jpg'
img = cv2.imread(img_path)
img=  cv2.resize(img,(224,224))
img = img/255


# In[20]:


plt.imshow(img)
plt.show()


# In[21]:


np.shape(img)


# In[22]:


image = np.expand_dims(img, axis=0) 


# In[23]:


# Predict the class
prediction = model.predict(image)


# In[24]:


prediction


# In[25]:


predicted_class_index = np.argmax(model.predict(image))


# In[26]:


#Benign - 0, Early -1, Pre-2, Pro-3

predicted_class_index


# In[27]:


# Initialize LIME image explainer
explainer = lime_image.LimeImageExplainer()


# In[28]:


# Explanation
explanation = explainer.explain_instance(img, model.predict, top_labels=1, hide_color=0, num_samples=1000)


# In[29]:


# Display the image with most contributing regions
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp, mask,color=(0,0,0),mode="thick"))
plt.show()


# In[30]:


# Display the whole image 
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
plt.imshow(mark_boundaries(temp, mask,color=(0,0,0),mode="thick"))
plt.show()

