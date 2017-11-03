
# coding: utf-8

# In[20]:


import tensorflow_number_picture_function as tnpf
import math
import numpy as np
import h5py
from tf_utils import load_dataset,convert_to_one_hot, predict

get_ipython().magic('matplotlib inline')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[24]:


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
parameters = tnpf.model(X_train, Y_train, X_test, Y_test)

