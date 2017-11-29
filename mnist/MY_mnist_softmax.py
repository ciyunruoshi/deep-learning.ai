
# coding: utf-8

# In[22]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import input_data

import tensorflow as tf

import function_mnist as fm

get_ipython().magic('load_ext autoreload')
 
get_ipython().magic('autoreload 2')


# In[15]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
hidden1_units = 128
hidden2_units = 32
print(mnist.train.labels.shape)


# In[16]:


X,Y = fm.initialization_put();
logits = fm.forward_prop(X,hidden1_units,hidden2_units);
cost = fm.cost(logits,Y)
optim = fm.optimizer_cost(cost,learning_rate=0.5)
#accuray = fm.predict_accuary(logits,labels);


# In[ ]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# In[ ]:


for _ in range(1000):
    epoch_cost = 0;
    for i in range(550):
    
        batch_xs, batch_ys = mnist.train.next_batch(100); 
        _,batch_cost = sess.run([optim,cost], feed_dict={X: batch_xs, Y: batch_ys})
        epoch_cost+=batch_cost/550;
    print(epoch_cost)


# In[62]:





