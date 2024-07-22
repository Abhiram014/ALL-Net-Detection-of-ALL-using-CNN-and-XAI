#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os

import numpy as np
import random
import cv2


# In[27]:


d1="ALL dataset without augmentation/Benign"
d2="ALL dataset without augmentation/Early"
d3="ALL dataset without augmentation/Pre"
d4="ALL dataset without augmentation/Pro"

j=0

for file in os.listdir(d1):
                file_path = os.path.join(d1, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    j+=1
print(j)

k=0

for file in os.listdir(d2):
                file_path = os.path.join(d2, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    k+=1
print(k)
                    
l=0

for file in os.listdir(d3):
                file_path = os.path.join(d3, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    l+=1
print(l)

m=0

for file in os.listdir(d4):
                file_path = os.path.join(d4, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    m+=1
print(m)


# In[28]:


categories = ["Benign","Early","Pre","Pro"]


# In[29]:


freq=[j,k,l,m]


# In[30]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.bar(categories, freq)
plt.xlabel('Classes')
plt.ylabel('Number of Images')


for i in range(len(categories)):
    plt.text(categories[i], freq[i] + 1, str(freq[i]), ha='center')

plt.savefig("Distribution of images without augmentation 1.png")

plt.show()


# In[31]:


d1="ALL dataset/Benign"
d2="ALL dataset/Early"
d3="ALL dataset/Pre"
d4="ALL dataset/Pro"

j=0

for file in os.listdir(d1):
                file_path = os.path.join(d1, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    j+=1
print(j)

k=0

for file in os.listdir(d2):
                file_path = os.path.join(d2, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    k+=1
print(k)
                    
l=0

for file in os.listdir(d3):
                file_path = os.path.join(d3, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    l+=1
print(l)

m=0

for file in os.listdir(d4):
                file_path = os.path.join(d4, file)
                if file_path.endswith('.jpg') or file_path.endswith('.png'):
                    m+=1
print(m)


# In[32]:


freq=[j,k,l,m]


# In[34]:


import matplotlib.pyplot as plt



plt.figure(figsize=(8, 6))
plt.bar(categories, freq)
plt.xlabel('Classes')
plt.ylabel('Number of Images')




for i in range(len(categories)):
    plt.text(categories[i], freq[i] + 1, str(freq[i]), ha='center')
plt.savefig("Distribution of images with augmentation.png")
plt.show()


# In[ ]:




